const BACKEND_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws";
const DEBUG_UPDATE_MS = 250;
const TIMER_UPDATE_MS = 33;
const RUN_MODES = {
  stable: {
    fps: 36,
    label: "Stable 36",
    lightUi: true,
  },
  laptop: {
    fps: 24,
    label: "Laptop 24",
    lightUi: true,
  },
  competition: {
    fps: 60,
    label: "Competition 60",
    lightUi: false,
  },
};

const canvas = document.getElementById("gameCanvas");
const renderer = createFrameRenderer(canvas);
const packetDecoder = new TextDecoder();

const startBtn = document.getElementById("startBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resetBtn = document.getElementById("resetBtn");
const worldSelect = document.getElementById("worldSelect");
const stageSelect = document.getElementById("stageSelect");
const versionSelect = document.getElementById("versionSelect");
const ghostSelect = document.getElementById("ghostSelect");
const modeSelect = document.getElementById("modeSelect");
const statusText = document.getElementById("status");
const debugText = document.getElementById("debug");
const runTimerText = document.getElementById("runTimer");
const bestTimerText = document.getElementById("bestTimer");
const runInfoPanel = document.querySelector(".run-info");

let running = false;
let loopId = null;
let inFlight = false;
let socket = null;
let pendingSocketResponse = null;
let lastDebugUpdateMs = 0;

let timerStartMs = 0;
let elapsedBeforePauseMs = 0;
let timerRunning = false;
let timerAnimationId = null;
let currentLevelKey = "1-1-v0";
let currentRunMode = modeSelect.value;
let lastTimerDisplayMs = 0;
let gameStarted = false;

const keys = {
  right: false,
  left: false,
  jump: false,
  run: false,
};

function keyToInput(code) {
  if (code === "ArrowRight" || code === "KeyD") return "right";
  if (code === "ArrowLeft" || code === "KeyA") return "left";
  if (code === "Space" || code === "KeyZ" || code === "KeyJ" || code === "ArrowUp" || code === "KeyW") return "jump";
  if (code === "ShiftLeft" || code === "ShiftRight" || code === "KeyX" || code === "KeyK") return "run";
  return null;
}

function isEditingControl(target) {
  return ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target?.tagName);
}

function selectByOffset(select, offset) {
  const options = Array.from(select.options);
  if (!options.length) return;
  const currentIndex = Math.max(0, options.findIndex((option) => option.value === select.value));
  const nextIndex = (currentIndex + offset + options.length) % options.length;
  select.value = options[nextIndex].value;
  handleLevelSelectionChanged();
}

function setSelectValue(select, value) {
  if (select.value === value) return;
  select.value = value;
  handleLevelSelectionChanged();
}

function handleAppShortcut(event) {
  if (event.repeat || isEditingControl(event.target)) return false;

  if (event.code === "Enter") {
    event.preventDefault();
    startGame();
    return true;
  }

  if (event.code === "KeyP") {
    event.preventDefault();
    togglePause();
    return true;
  }

  if (event.code === "KeyR") {
    event.preventDefault();
    resetGame();
    return true;
  }

  if (event.code === "KeyL") {
    event.preventDefault();
    worldSelect.focus();
    return true;
  }

  if (event.code === "KeyM") {
    event.preventDefault();
    selectByOffset(modeSelect, 1);
    return true;
  }

  if (event.code === "KeyV") {
    event.preventDefault();
    selectByOffset(versionSelect, 1);
    return true;
  }

  if (event.code === "KeyG") {
    event.preventDefault();
    selectByOffset(ghostSelect, 1);
    return true;
  }

  if (event.code === "BracketLeft") {
    event.preventDefault();
    selectByOffset(stageSelect, -1);
    return true;
  }

  if (event.code === "BracketRight") {
    event.preventDefault();
    selectByOffset(stageSelect, 1);
    return true;
  }

  if (event.code === "Slash" && event.shiftKey) {
    event.preventDefault();
    if (runInfoPanel) runInfoPanel.open = !runInfoPanel.open;
    return true;
  }

  if (/^Digit[1-8]$/.test(event.code)) {
    event.preventDefault();
    setSelectValue(worldSelect, event.code.replace("Digit", ""));
    return true;
  }

  return false;
}

window.addEventListener("keydown", (event) => {
  if (handleAppShortcut(event)) return;

  const input = keyToInput(event.code);
  if (!input) return;
  event.preventDefault();
  keys[input] = true;
});

window.addEventListener("keyup", (event) => {
  const input = keyToInput(event.code);
  if (!input) return;
  event.preventDefault();
  keys[input] = false;
});

function connectSocket() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    return Promise.resolve(socket);
  }

  if (socket && socket.readyState === WebSocket.CONNECTING) {
    return new Promise((resolve, reject) => {
      socket.addEventListener("open", () => resolve(socket), { once: true });
      socket.addEventListener("error", reject, { once: true });
    });
  }

  return new Promise((resolve, reject) => {
    socket = new WebSocket(WS_URL);
    socket.binaryType = "arraybuffer";

    socket.addEventListener("open", () => resolve(socket), { once: true });
    socket.addEventListener("error", reject, { once: true });
    socket.addEventListener("message", handleSocketMessage);
    socket.addEventListener("close", () => {
      socket = null;
      if (pendingSocketResponse) {
        pendingSocketResponse.reject(new Error("Game socket closed."));
        pendingSocketResponse = null;
      }
    });
  });
}

async function sendGameMessage(message) {
  const activeSocket = await connectSocket();

  return new Promise((resolve, reject) => {
    if (pendingSocketResponse) {
      pendingSocketResponse.reject(new Error("Interrupted by a newer game frame request."));
    }

    pendingSocketResponse = { resolve, reject };
    activeSocket.send(JSON.stringify(message));
  });
}

function handleSocketMessage(event) {
  if (!(event.data instanceof ArrayBuffer)) return;

  try {
    const data = parseFramePacket(event.data);
    drawRawFrame(data.frame, data.meta.width, data.meta.height, data.meta.channels);

    if (pendingSocketResponse) {
      pendingSocketResponse.resolve(data.meta);
      pendingSocketResponse = null;
    }
  } catch (error) {
    if (pendingSocketResponse) {
      pendingSocketResponse.reject(error);
      pendingSocketResponse = null;
    }
  }
}

function parseFramePacket(buffer) {
  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, true);
  const headerBytes = new Uint8Array(buffer, 4, headerLength);
  const meta = JSON.parse(packetDecoder.decode(headerBytes));
  const frame = new Uint8Array(buffer, 4 + headerLength);
  return { meta, frame };
}


async function loadGhostOptions(preferredId = null) {
  const current = preferredId || ghostSelect.value || "none";

  try {
    const response = await fetch(`${BACKEND_URL}/ghosts`);
    if (!response.ok) return;
    const data = await response.json();

    ghostSelect.innerHTML = "";
    for (const ghost of data.ghosts || []) {
      const option = document.createElement("option");
      option.value = ghost.id;
      option.textContent = ghost.name;
      option.title = ghost.description || "";
      ghostSelect.appendChild(option);
    }

    const exists = Array.from(ghostSelect.options).some((option) => option.value === current);
    ghostSelect.value = exists ? current : "none";
  } catch (error) {
    console.warn("Could not load ghosts", error);
  }
}

async function startGame() {
  try {
    const world = Number(worldSelect.value);
    const stage = Number(stageSelect.value);
    const version = Number(versionSelect.value);
    const ghostId = ghostSelect.value;
    currentRunMode = modeSelect.value;

    running = false;
    gameStarted = false;
    stopLoop();
    inFlight = false;
    clearKeys();

    currentLevelKey = `${world}-${stage}-v${version}`;
    loadBestTime();
    resetTimer();

    applyRunMode();
    statusText.textContent = `Starting World ${world}-${stage} v${version} (${getRunMode().label})...`;
    const data = await sendGameMessage({
      type: "start",
      world,
      stage,
      version,
      ghost_id: ghostId,
      run_mode: currentRunMode,
    });
    updateDebug(data, true);

    gameStarted = true;
    running = true;
    pauseBtn.textContent = "Pause";
    startTimer();
    startLoop();
    statusText.textContent = `Running ${data.info?.env_id || `World ${world}-${stage} v${version}`} (${getRunMode().label}).`;
  } catch (error) {
    statusText.textContent = `Could not start game: ${error.message}`;
    console.error(error);
  }
}

async function resetGame() {
  try {
    running = false;
    gameStarted = false;
    stopLoop();
    inFlight = false;
    clearKeys();
    resetTimer();

    const data = await sendGameMessage({ type: "reset" });
    updateDebug(data, true);

    gameStarted = true;
    running = true;
    pauseBtn.textContent = "Pause";
    startTimer();
    startLoop();
    statusText.textContent = `Reset ${data.info?.env_id || "current level"}.`;
  } catch (error) {
    statusText.textContent = `Could not reset: ${error.message}`;
  }
}

function startLoop() {
  if (loopId !== null) return;

  const frameDelayMs = 1000 / getRunMode().fps;
  let lastFrameTime = 0;

  const loop = async (now) => {
    if (running && !inFlight && now - lastFrameTime >= frameDelayMs) {
      lastFrameTime = now;
      inFlight = true;

      try {
        const data = await sendGameMessage({
          type: "step",
          input: { ...keys },
        });

        updateDebug(data);

        if (data.done) {
          const finalTime = stopTimer();

          // Only count as a valid speedrun if the player reached the flag
          if (data.info && data.info.flag_get) {
            saveBestTime(finalTime);
            statusText.textContent = `Finished! ${formatTime(finalTime)}. Saved as a replay ghost.`;
            setTimeout(() => loadGhostOptions(), 400);
          } else {
            statusText.textContent = `Run ended (death). Time not saved.`;
          }

          running = false;
          gameStarted = false;
        }
      } catch (error) {
        statusText.textContent = `Step failed: ${error.message}`;
        running = false;
        gameStarted = false;
        pauseTimer();
      } finally {
        inFlight = false;
      }
    }

    if (running) {
      loopId = requestAnimationFrame(loop);
    } else {
      loopId = null;
    }
  };

  loopId = requestAnimationFrame(loop);
}

function stopLoop() {
  if (loopId !== null) {
    cancelAnimationFrame(loopId);
    loopId = null;
  }
}

function startTimer() {
  timerStartMs = performance.now();
  timerRunning = true;
  startTimerRenderLoop();
  updateTimerDisplay(true);
}

function pauseTimer() {
  if (!timerRunning) return;
  elapsedBeforePauseMs += performance.now() - timerStartMs;
  timerRunning = false;
  updateTimerDisplay(true);
}

function resumeTimer() {
  if (timerRunning) return;
  timerStartMs = performance.now();
  timerRunning = true;
  startTimerRenderLoop();
  updateTimerDisplay(true);
}

function stopTimer() {
  if (timerRunning) {
    elapsedBeforePauseMs += performance.now() - timerStartMs;
    timerRunning = false;
  }
  stopTimerRenderLoop();
  updateTimerDisplay(true);
  return elapsedBeforePauseMs;
}

function resetTimer() {
  timerStartMs = 0;
  elapsedBeforePauseMs = 0;
  timerRunning = false;
  stopTimerRenderLoop();
  updateTimerDisplay(true);
}

function getElapsedMs() {
  if (!timerRunning) return elapsedBeforePauseMs;
  return elapsedBeforePauseMs + (performance.now() - timerStartMs);
}

function updateTimerDisplay(force = false) {
  const now = performance.now();
  if (!force && now - lastTimerDisplayMs < TIMER_UPDATE_MS) return;
  lastTimerDisplayMs = now;
  runTimerText.textContent = formatTime(getElapsedMs());
}

function startTimerRenderLoop() {
  if (timerAnimationId !== null) return;

  const tick = () => {
    updateTimerDisplay();
    if (timerRunning) {
      timerAnimationId = requestAnimationFrame(tick);
    } else {
      timerAnimationId = null;
    }
  };

  timerAnimationId = requestAnimationFrame(tick);
}

function stopTimerRenderLoop() {
  if (timerAnimationId !== null) {
    cancelAnimationFrame(timerAnimationId);
    timerAnimationId = null;
  }
}

function formatTime(ms) {
  const totalMs = Math.max(0, Math.floor(ms));
  const minutes = Math.floor(totalMs / 60000);
  const seconds = Math.floor((totalMs % 60000) / 1000);
  const millis = totalMs % 1000;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function bestStorageKey() {
  return `mario-best-${currentLevelKey}-${currentRunMode}`;
}

function loadBestTime() {
  const best = localStorage.getItem(bestStorageKey());
  bestTimerText.textContent = best ? formatTime(Number(best)) : "--:--.---";
}

function saveBestTime(ms) {
  const key = bestStorageKey();
  const previous = localStorage.getItem(key);
  if (!previous || ms < Number(previous)) {
    localStorage.setItem(key, String(Math.floor(ms)));
    bestTimerText.textContent = formatTime(ms);
  }
}

function clearKeys() {
  keys.right = false;
  keys.left = false;
  keys.jump = false;
  keys.run = false;
}

function createFrameRenderer(targetCanvas) {
  const gl = targetCanvas.getContext("webgl", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    preserveDrawingBuffer: false,
    powerPreference: "high-performance",
  });

  if (gl) {
    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, `
      attribute vec2 a_position;
      attribute vec2 a_texCoord;
      varying vec2 v_texCoord;

      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
      }
    `);

    const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, `
      precision mediump float;
      varying vec2 v_texCoord;
      uniform sampler2D u_image;

      void main() {
        gl_FragColor = texture2D(u_image, v_texCoord);
      }
    `);

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.warn("WebGL program link failed", gl.getProgramInfoLog(program));
    } else {
      gl.useProgram(program);

      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1,  1, 0, 0,
         1,  1, 1, 0,
        -1, -1, 0, 1,
         1, -1, 1, 1,
      ]), gl.STATIC_DRAW);

      const stride = 4 * Float32Array.BYTES_PER_ELEMENT;
      const positionLocation = gl.getAttribLocation(program, "a_position");
      gl.enableVertexAttribArray(positionLocation);
      gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, stride, 0);

      const texCoordLocation = gl.getAttribLocation(program, "a_texCoord");
      gl.enableVertexAttribArray(texCoordLocation);
      gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, stride, 2 * Float32Array.BYTES_PER_ELEMENT);

      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.uniform1i(gl.getUniformLocation(program, "u_image"), 0);

      return {
        drawFrame(frame, width, height, channels) {
          gl.viewport(0, 0, targetCanvas.width, targetCanvas.height);
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, texture);
          gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
          gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            channels === 4 ? gl.RGBA : gl.RGB,
            width,
            height,
            0,
            channels === 4 ? gl.RGBA : gl.RGB,
            gl.UNSIGNED_BYTE,
            frame,
          );
          gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        },
      };
    }
  }

  const ctx = targetCanvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  let imageData = null;

  return {
    drawFrame(frame, width, height, channels) {
      if (!imageData || imageData.width !== width || imageData.height !== height) {
        imageData = ctx.createImageData(width, height);
      }

      const rgba = imageData.data;
      if (channels === 4) {
        rgba.set(frame);
      } else if (channels === 3) {
        for (let source = 0, target = 0; source < frame.length; source += 3, target += 4) {
          rgba[target] = frame[source];
          rgba[target + 1] = frame[source + 1];
          rgba[target + 2] = frame[source + 2];
          rgba[target + 3] = 255;
        }
      } else {
        for (let source = 0, target = 0; source < frame.length; source += channels, target += 4) {
          const value = frame[source];
          rgba[target] = value;
          rgba[target + 1] = value;
          rgba[target + 2] = value;
          rgba[target + 3] = 255;
        }
      }

      ctx.putImageData(imageData, 0, 0);
    },
  };
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Could not compile WebGL shader: ${message}`);
  }

  return shader;
}

function drawRawFrame(frame, width, height, channels) {
  renderer.drawFrame(frame, width, height, channels);
}

function updateDebug(data, force = false) {
  const now = performance.now();
  if (!force && now - lastDebugUpdateMs < DEBUG_UPDATE_MS) return;
  lastDebugUpdateMs = now;

  const info = data.info || {};
  debugText.textContent = [
    `Step: ${data.step}`,
    `Action: ${data.action}`,
    `Env: ${info.env_id || "-"}`,
    `X: ${info.x_pos ?? "-"}`,
    `Y: ${info.y_pos ?? "-"}`,
    `Time: ${info.time ?? "-"}`,
    `Score: ${info.score ?? "-"}`,
    `Done: ${data.done}`,
  ].join("\n");
}

function handleLevelSelectionChanged() {
  const world = Number(worldSelect.value);
  const stage = Number(stageSelect.value);
  const version = Number(versionSelect.value);
  currentLevelKey = `${world}-${stage}-v${version}`;
  currentRunMode = modeSelect.value;
  applyRunMode();
  loadBestTime();
  statusText.textContent = `Ready. World ${world}-${stage} v${version}.`;
}

function getRunMode() {
  return RUN_MODES[currentRunMode] || RUN_MODES.stable;
}

function applyRunMode() {
  const mode = getRunMode();
  document.body.classList.toggle("light-ui", mode.lightUi);
}

function togglePause() {
  if (!gameStarted) {
    statusText.textContent = "No active run.";
    return;
  }

  running = !running;

  if (running) {
    resumeTimer();
  } else {
    pauseTimer();
  }

  pauseBtn.textContent = running ? "Pause" : "Resume";
  statusText.textContent = running ? "Running." : "Paused.";
}

startBtn.addEventListener("click", startGame);
resetBtn.addEventListener("click", resetGame);
pauseBtn.addEventListener("click", togglePause);

worldSelect.addEventListener("change", handleLevelSelectionChanged);
stageSelect.addEventListener("change", handleLevelSelectionChanged);
versionSelect.addEventListener("change", handleLevelSelectionChanged);
ghostSelect.addEventListener("change", handleLevelSelectionChanged);
modeSelect.addEventListener("change", handleLevelSelectionChanged);

applyRunMode();
loadGhostOptions();
loadBestTime();
updateTimerDisplay();
