/* ============================================================
   main.js — Wire up the play page
   ------------------------------------------------------------
   - Reads keyboard input
   - Runs the game loop at a fixed 60 Hz tick (decoupled from
     the browser's variable rAF rate)
   - Updates the HUD + sidebar metadata
   - Handles win / lose / restart / pause / ghost-toggle
   ============================================================ */

(function () {
  'use strict';

  const { TILE, VIEW_W, VIEW_H, buildWorld, resetWorld, step, drawWorld } = window.Game;

  const FRAME_MS = 1000 / 60;

  // --- DOM refs ----------------------------------------------
  const canvas = document.getElementById('game');
  const ctx    = canvas.getContext('2d');

  const elTime  = document.getElementById('hud-time');
  const elCoins = document.getElementById('hud-coins');
  const elScore = document.getElementById('hud-score');
  const elDiff  = document.getElementById('hud-diff');

  const elOverlay      = document.getElementById('overlay');
  const elOverTitle    = document.getElementById('overlay-title');
  const elOverLine     = document.getElementById('overlay-line');
  const elOverTime     = document.getElementById('overlay-time');
  const elOverCoins    = document.getElementById('overlay-coins');
  const elOverResult   = document.getElementById('overlay-result');
  const elOverRestart  = document.getElementById('overlay-restart');

  const elGhostMeta   = document.getElementById('ghost-meta');
  const elGhostStatus = document.getElementById('ghost-status');
  const elGhostToggle = document.getElementById('ghost-toggle');

  const btnPause   = document.getElementById('pause-btn');
  const lblPause   = document.getElementById('pause-label');
  const btnRestart = document.getElementById('restart-btn');

  // --- Keyboard ----------------------------------------------
  const keys = new Set();
  const KEY_MAP = {
    ArrowLeft: 'left',  KeyA: 'left',
    ArrowRight: 'right', KeyD: 'right',
    ArrowUp: 'jump',    KeyW: 'jump',  Space: 'jump',  KeyZ: 'jump',
    ShiftLeft: 'sprint', ShiftRight: 'sprint',
  };

  function onKeyDown(e) {
    const action = KEY_MAP[e.code];
    if (action) {
      keys.add(action);
      if (action === 'jump' || action === 'sprint' || action === 'left' || action === 'right') {
        e.preventDefault();
      }
    }
    if (e.code === 'KeyR') restart();
    else if (e.code === 'KeyP') togglePause();
    else if (e.code === 'KeyG') {
      elGhostToggle.checked = !elGhostToggle.checked;
      applyGhostToggle();
    }
  }
  function onKeyUp(e) {
    const action = KEY_MAP[e.code];
    if (action) keys.delete(action);
  }
  function getPlayerInputs() { return keys; }

  // --- Game state --------------------------------------------
  let world = buildWorld();
  let ghostPolicy = window.Ghost.scriptedPolicy;
  let ghostMeta = window.Ghost.SAMPLE_META;

  // Race timing
  let runStarted = false;
  let runStartMs = 0;
  let runEndMs = 0;

  // Pause
  let paused = false;

  // Ghost toggle
  function applyGhostToggle() {
    world.ghostEnabled = elGhostToggle.checked;
    world.ghostVisible = elGhostToggle.checked;
    elGhostStatus.textContent = elGhostToggle.checked ? 'Active' : 'Paused';
    elGhostStatus.style.background = elGhostToggle.checked
      ? 'rgba(6, 214, 160, 0.1)' : 'rgba(255, 209, 102, 0.1)';
    elGhostStatus.style.color = elGhostToggle.checked
      ? 'var(--green)' : 'var(--gold)';
  }

  // Pause toggle
  function togglePause() {
    if (world.state !== 'playing') return;
    paused = !paused;
    lblPause.textContent = paused ? 'Resume' : 'Pause';
  }

  // Restart
  function restart() {
    resetWorld(world);
    runStarted = false;
    runStartMs = 0;
    runEndMs = 0;
    paused = false;
    lblPause.textContent = 'Pause';
    elOverlay.classList.remove('show');
    applyGhostToggle();
  }

  // --- Sidebar render ----------------------------------------
  function renderGhostMeta() {
    elGhostMeta.innerHTML =
      '<strong>' + escapeHtml(ghostMeta.name) + '</strong>' +
      escapeHtml(ghostMeta.description);
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // --- HUD ---------------------------------------------------
  function updateHud(now) {
    // Time
    let elapsed;
    if (!runStarted) elapsed = 0;
    else if (world.state === 'playing') elapsed = (now - runStartMs) / 1000;
    else elapsed = (runEndMs - runStartMs) / 1000;
    elTime.textContent = elapsed.toFixed(2);

    elCoins.textContent = world.coinCount;
    elScore.textContent = world.score;

    // Differential = player.x - ghost.x in tiles
    if (world.ghostEnabled) {
      const dxTiles = (world.player.x - world.ghost.x) / TILE;
      const sign = dxTiles >= 0 ? '+' : '−';
      elDiff.textContent = sign + Math.abs(dxTiles).toFixed(1);
      elDiff.className = 'val ' + (dxTiles >= 0 ? 'green' : 'red');
    } else {
      elDiff.textContent = '—';
      elDiff.className = 'val';
    }
  }

  // --- End-of-run overlay ------------------------------------
  function showOverlay() {
    const elapsed = (runEndMs - runStartMs) / 1000;
    elOverTime.textContent  = elapsed.toFixed(2) + 's';
    elOverCoins.textContent = world.coinCount;

    if (world.state === 'won') {
      if (world.winner === 'player') {
        elOverTitle.textContent = 'You beat the ghost!';
        elOverLine.textContent  = 'Clean run. Try a faster line next time.';
        elOverResult.textContent = 'Win';
        elOverResult.style.color = 'var(--green)';
      } else if (world.winner === 'tie') {
        elOverTitle.textContent = 'Photo finish';
        elOverLine.textContent  = 'You hit the flag on the same frame as the ghost.';
        elOverResult.textContent = 'Tie';
        elOverResult.style.color = 'var(--gold)';
      } else {
        elOverTitle.textContent = 'You reached the flag';
        elOverLine.textContent  = 'Nice run.';
        elOverResult.textContent = 'Cleared';
        elOverResult.style.color = 'var(--text)';
      }
    } else {
      elOverTitle.textContent = world.winner === 'ghost'
        ? 'The ghost beat you'
        : 'Run ended';
      elOverLine.textContent = world.player.alive
        ? 'Better luck on the next run.'
        : 'You took a hit. Press R or click below to retry.';
      elOverResult.textContent = world.winner === 'ghost' ? 'Loss' : '—';
      elOverResult.style.color = world.winner === 'ghost' ? 'var(--red-soft)' : 'var(--text-muted)';
    }

    elOverlay.classList.add('show');
  }

  // --- Fixed-timestep game loop ------------------------------
  let lastMs = performance.now();
  let acc = 0;

  function loop(now) {
    requestAnimationFrame(loop);
    const dt = now - lastMs;
    lastMs = now;
    if (paused) {
      // Still draw so the canvas isn't blank
      drawWorld(ctx, world);
      drawPauseHint(ctx);
      return;
    }

    acc += dt;
    if (acc > 250) acc = 250;       // avoid catch-up storm after tab switch

    // Tick at 60 Hz
    while (acc >= FRAME_MS) {
      tick(now);
      acc -= FRAME_MS;
    }

    drawWorld(ctx, world);
    updateHud(now);
  }

  function tick(now) {
    if (world.state !== 'playing') return;

    const playerInputs = getPlayerInputs();

    // Start the timer on first input
    if (!runStarted && playerInputs.size > 0) {
      runStarted = true;
      runStartMs = now;
    }

    let ghostInputs = new Set();
    if (world.ghostEnabled && runStarted) {
      try { ghostInputs = ghostPolicy(world); }
      catch (err) {
        console.error('[ghost policy error]', err);
        ghostInputs = new Set();
      }
    }

    step(world, playerInputs, ghostInputs);

    if (world.state !== 'playing' && runEndMs === 0) {
      runEndMs = now;
      // Defer overlay one tick so the final frame paints first
      setTimeout(showOverlay, 60);
    }
  }

  function drawPauseHint(ctx) {
    ctx.save();
    ctx.fillStyle = 'rgba(8, 12, 22, 0.55)';
    ctx.fillRect(0, 0, VIEW_W, VIEW_H);
    ctx.fillStyle = 'white';
    ctx.font = 'bold 28px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Paused', VIEW_W / 2, VIEW_H / 2 - 12);
    ctx.font = '14px Inter, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.fillText('Press P to resume', VIEW_W / 2, VIEW_H / 2 + 18);
    ctx.restore();
  }

  // --- Wire up event listeners -------------------------------
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);

  btnRestart.addEventListener('click', restart);
  btnPause.addEventListener('click', togglePause);
  elOverRestart.addEventListener('click', restart);
  elGhostToggle.addEventListener('change', applyGhostToggle);

  // Focus canvas so it gets keyboard events on first click too
  canvas.tabIndex = 0;
  canvas.addEventListener('click', () => canvas.focus());

  // --- Boot --------------------------------------------------
  applyGhostToggle();
  renderGhostMeta();
  requestAnimationFrame(loop);

  // ----------------------------------------------------------
  // To swap in a real RL trajectory, uncomment + edit:
  //
  //   window.Ghost.loadFromJson('data/ghost_trained.json')
  //     .then(({policy, meta}) => {
  //       ghostPolicy = policy;
  //       ghostMeta = meta;
  //       renderGhostMeta();
  //       restart();
  //     });
  //
  // The format is documented in js/ghost.js.
  // ----------------------------------------------------------

})();
