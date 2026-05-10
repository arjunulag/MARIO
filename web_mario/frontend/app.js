const BACKEND_URL = "http://127.0.0.1:8000";
const FPS = 60; // HTTP frame streaming is expensive; 24 is usually smoother than forcing 30/60.

const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
ctx.imageSmoothingEnabled = false;

const startBtn = document.getElementById("startBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resetBtn = document.getElementById("resetBtn");
const worldSelect = document.getElementById("worldSelect");
const stageSelect = document.getElementById("stageSelect");
const versionSelect = document.getElementById("versionSelect");
const statusText = document.getElementById("status");
const debugText = document.getElementById("debug");
const runTimerText = document.getElementById("runTimer");
const bestTimerText = document.getElementById("bestTimer");

let running = false;
let loopId = null;
let inFlight = false;

let timerStartMs = 0;
let elapsedBeforePauseMs = 0;
let timerRunning = false;
let timerAnimationId = null;
let currentLevelKey = "1-1-v0";

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

window.addEventListener("keydown", (event) => {
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

async function postJson(path, body = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);

  try {
    const response = await fetch(`${BACKEND_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    const text = await response.text();

    if (!response.ok) {
      throw new Error(`${path} failed with ${response.status}: ${text}`);
    }

    return JSON.parse(text);
  } finally {
    clearTimeout(timeoutId);
  }
}

async function startGame() {
  try {
    const world = Number(worldSelect.value);
    const stage = Number(stageSelect.value);
    const version = Number(versionSelect.value);

    running = false;
    stopLoop();
    inFlight = false;
    clearKeys();

    currentLevelKey = `${world}-${stage}-v${version}`;
    loadBestTime();
    resetTimer();

    statusText.textContent = `Starting World ${world}-${stage} v${version}...`;
    const data = await postJson("/start", { world, stage, version });
    drawFrame(data.frame);
    updateDebug(data);

    running = true;
    startTimer();
    startLoop();
    statusText.textContent = `Running ${data.info?.env_id || `World ${world}-${stage} v${version}`}.`;
  } catch (error) {
    statusText.textContent = `Could not start game: ${error.message}`;
    console.error(error);
  }
}

async function resetGame() {
  try {
    running = false;
    stopLoop();
    inFlight = false;
    clearKeys();
    resetTimer();

    const data = await postJson("/reset");
    drawFrame(data.frame);
    updateDebug(data);

    running = true;
    startTimer();
    startLoop();
    statusText.textContent = `Reset ${data.info?.env_id || "current level"}.`;
  } catch (error) {
    statusText.textContent = `Could not reset: ${error.message}`;
  }
}

function startLoop() {
  if (loopId !== null) return;

  const frameDelayMs = 1000 / FPS;
  let lastFrameTime = 0;

  const loop = async (now) => {
    if (running && !inFlight && now - lastFrameTime >= frameDelayMs) {
      lastFrameTime = now;
      inFlight = true;

      try {
        const data = await postJson("/step", {
          input: { ...keys },
        });

        drawFrame(data.frame);
        updateDebug(data);

        if (data.done) {
          const finalTime = stopTimer();

          // Only count as a valid speedrun if the player reached the flag
          if (data.info && data.info.flag_get) {
            saveBestTime(finalTime);
            statusText.textContent = `Finished! ${formatTime(finalTime)}.`;
          } else {
            statusText.textContent = `Run ended (death). Time not saved.`;
          }

          running = false;
        }
      } catch (error) {
        statusText.textContent = `Step failed: ${error.message}`;
        running = false;
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
  updateTimerDisplay();
}

function pauseTimer() {
  if (!timerRunning) return;
  elapsedBeforePauseMs += performance.now() - timerStartMs;
  timerRunning = false;
  updateTimerDisplay();
}

function resumeTimer() {
  if (timerRunning) return;
  timerStartMs = performance.now();
  timerRunning = true;
  startTimerRenderLoop();
  updateTimerDisplay();
}

function stopTimer() {
  if (timerRunning) {
    elapsedBeforePauseMs += performance.now() - timerStartMs;
    timerRunning = false;
  }
  stopTimerRenderLoop();
  updateTimerDisplay();
  return elapsedBeforePauseMs;
}

function resetTimer() {
  timerStartMs = 0;
  elapsedBeforePauseMs = 0;
  timerRunning = false;
  stopTimerRenderLoop();
  updateTimerDisplay();
}

function getElapsedMs() {
  if (!timerRunning) return elapsedBeforePauseMs;
  return elapsedBeforePauseMs + (performance.now() - timerStartMs);
}

function updateTimerDisplay() {
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
  return `mario-best-${currentLevelKey}`;
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

function drawFrame(dataUri) {
  if (!dataUri) return;

  const image = new Image();
  image.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  };
  image.src = dataUri;
}

function updateDebug(data) {
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
  loadBestTime();
  statusText.textContent = "Level changed. Click Start Level to switch.";
}

startBtn.addEventListener("click", startGame);
resetBtn.addEventListener("click", resetGame);

pauseBtn.addEventListener("click", () => {
  running = !running;

  if (running) {
    resumeTimer();
  } else {
    pauseTimer();
  }

  pauseBtn.textContent = running ? "Pause" : "Resume";
  statusText.textContent = running ? "Running." : "Paused.";
});

worldSelect.addEventListener("change", handleLevelSelectionChanged);
stageSelect.addEventListener("change", handleLevelSelectionChanged);
versionSelect.addEventListener("change", handleLevelSelectionChanged);

loadBestTime();
updateTimerDisplay();