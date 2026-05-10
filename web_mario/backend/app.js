const BACKEND_URL = "http://127.0.0.1:8000";
const FPS = 60;

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

let running = false;
let loopId = null;
let inFlight = false;

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
  const response = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`${path} failed with ${response.status}`);
  }

  return response.json();
}

async function startGame() {
  try {
    const world = Number(worldSelect.value);
    const stage = Number(stageSelect.value);
    const version = Number(versionSelect.value);

    // Important: stop the current stepping loop before switching envs.
    // Otherwise an old /step request can keep advancing right after /start.
    running = false;
    stopLoop();
    inFlight = false;
    clearKeys();

    statusText.textContent = `Starting World ${world}-${stage} v${version}...`;
    const data = await postJson("/start", { world, stage, version });
    drawFrame(data.frame);
    updateDebug(data);

    running = true;
    startLoop();
    statusText.textContent = `Running ${data.info?.env_id || `World ${world}-${stage} v${version}`}.`;
  } catch (error) {
    statusText.textContent = `Could not start game: ${error.message}`;
  }
}

async function resetGame() {
  try {
    running = false;
    stopLoop();
    inFlight = false;
    clearKeys();

    const data = await postJson("/reset");
    drawFrame(data.frame);
    updateDebug(data);

    running = true;
    startLoop();
    statusText.textContent = `Reset ${data.info?.env_id || "current level"}.`;
  } catch (error) {
    statusText.textContent = `Could not reset: ${error.message}`;
  }
}

function startLoop() {
  if (loopId !== null) return;

  loopId = setInterval(async () => {
    if (!running || inFlight) return;
    inFlight = true;

    try {
      const data = await postJson("/step", {
        input: { ...keys },
      });

      drawFrame(data.frame);
      updateDebug(data);

      if (data.done) {
        statusText.textContent = "Run ended. Press Reset or Start.";
        running = false;
      }
    } catch (error) {
      statusText.textContent = `Step failed: ${error.message}`;
      running = false;
    } finally {
      inFlight = false;
    }
  }, 1000 / FPS);
}

function stopLoop() {
  if (loopId !== null) {
    clearInterval(loopId);
    loopId = null;
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

startBtn.addEventListener("click", startGame);

worldSelect.addEventListener("change", () => {
  statusText.textContent = "Level changed. Click Start Level to switch.";
});

stageSelect.addEventListener("change", () => {
  statusText.textContent = "Level changed. Click Start Level to switch.";
});

versionSelect.addEventListener("change", () => {
  statusText.textContent = "Version changed. Click Start Level to switch.";
});

pauseBtn.addEventListener("click", () => {
  running = !running;
  pauseBtn.textContent = running ? "Pause" : "Resume";
  statusText.textContent = running ? "Running." : "Paused.";
});

resetBtn.addEventListener("click", resetGame);