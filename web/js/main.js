/* ============================================================
   main.js — Wire up the play page
   ============================================================ */

(function () {
  'use strict';

  const { TILE, VIEW_W, VIEW_H, buildWorld, step, drawWorld } = window.Game;

  const FRAME_MS = 1000 / 60;
  const BEST_TIME_KEY = 'mario-play-best-1-1';

  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');

  const elTime = document.getElementById('stat-time');
  const elBest = document.getElementById('stat-best');
  const elCoins = document.getElementById('stat-coins');
  const elGap = document.getElementById('stat-gap');

  const elOverlay = document.getElementById('overlay');
  const elOverTitle = document.getElementById('overlay-title');
  const elOverLine = document.getElementById('overlay-line');
  const elOverTime = document.getElementById('overlay-time');
  const elOverBest = document.getElementById('overlay-best');
  const elOverCoins = document.getElementById('overlay-coins');
  const elOverResult = document.getElementById('overlay-result');
  const elOverRecord = document.getElementById('overlay-record');
  const elOverRestart = document.getElementById('overlay-restart');

  const elGhostMeta = document.getElementById('ghost-meta');
  const elGhostStatus = document.getElementById('ghost-status');
  const elGhostToggle = document.getElementById('ghost-toggle');

  const btnPause = document.getElementById('pause-btn');
  const lblPause = document.getElementById('pause-label');
  const btnRestart = document.getElementById('restart-btn');

  const keys = new Set();
  const KEY_MAP = {
    ArrowLeft: 'left', KeyA: 'left',
    ArrowRight: 'right', KeyD: 'right',
    ArrowUp: 'jump', KeyW: 'jump', Space: 'jump', KeyZ: 'jump',
    ShiftLeft: 'sprint', ShiftRight: 'sprint',
  };

  let world = buildWorld();
  let ghostPolicy = window.Ghost.scriptedPolicy;
  let ghostMeta = window.Ghost.SAMPLE_META;

  let runStarted = false;
  let runStartMs = 0;
  let runEndMs = 0;
  let bestTimeSec = loadBestTimeSec();
  let paused = false;
  let overlayTimerId = null;

  let lastMs = performance.now();
  let acc = 0;

  function formatTimeSec(seconds) {
    return Number(seconds).toFixed(2);
  }

  function formatBestDisplay(seconds) {
    return seconds === null ? '—' : formatTimeSec(seconds);
  }

  function loadBestTimeSec() {
    try {
      const raw = localStorage.getItem(BEST_TIME_KEY);
      if (raw === null) return null;
      const value = Number(raw);
      if (!Number.isFinite(value) || value <= 0) return null;
      return value;
    } catch (err) {
      console.warn('[best time] could not read saved time', err);
      return null;
    }
  }

  function persistBestTimeSec(seconds) {
    try {
      localStorage.setItem(BEST_TIME_KEY, String(seconds));
      return true;
    } catch (err) {
      console.warn('[best time] could not save time', err);
      return false;
    }
  }

  function tryRecordBestTime(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) return false;
    if (bestTimeSec !== null && seconds >= bestTimeSec) return false;
    bestTimeSec = seconds;
    persistBestTimeSec(seconds);
    return true;
  }

  function renderBestTime() {
    elBest.textContent = formatBestDisplay(bestTimeSec);
  }

  function applyGhostToggle() {
    const on = elGhostToggle.checked;
    world.ghostEnabled = on;
    world.ghostVisible = on;
    elGhostStatus.textContent = on ? 'Active' : 'Off';
    elGhostStatus.classList.toggle('is-off', !on);
  }

  function togglePause() {
    if (world.state !== 'playing') return;
    paused = !paused;
    lblPause.textContent = paused ? 'Resume' : 'Pause';
  }

  function clearOverlayTimer() {
    if (overlayTimerId !== null) {
      clearTimeout(overlayTimerId);
      overlayTimerId = null;
    }
  }

  function hideOverlay() {
    clearOverlayTimer();
    elOverlay.hidden = true;
  }

  function restart() {
    clearOverlayTimer();
    keys.clear();

    const ghostOn = elGhostToggle.checked;
    world = buildWorld();
    world.ghostEnabled = ghostOn;
    world.ghostVisible = ghostOn;

    runStarted = false;
    runStartMs = 0;
    runEndMs = 0;
    paused = false;
    acc = 0;
    lastMs = performance.now();

    lblPause.textContent = 'Pause';
    hideOverlay();
    applyGhostToggle();
    updateHud(lastMs);
    drawWorld(ctx, world);
  }

  function onKeyDown(e) {
    if (e.code === 'KeyR') {
      e.preventDefault();
      restart();
      return;
    }

    const action = KEY_MAP[e.code];
    if (action) {
      keys.add(action);
      if (action === 'jump' || action === 'sprint' || action === 'left' || action === 'right') {
        e.preventDefault();
      }
      return;
    }

    if (e.code === 'KeyP') {
      e.preventDefault();
      togglePause();
    } else if (e.code === 'KeyG') {
      e.preventDefault();
      elGhostToggle.checked = !elGhostToggle.checked;
      applyGhostToggle();
    }
  }

  function onKeyUp(e) {
    const action = KEY_MAP[e.code];
    if (action) keys.delete(action);
  }

  function getPlayerInputs() {
    return keys;
  }

  function renderGhostMeta() {
    elGhostMeta.innerHTML =
      '<strong>' + escapeHtml(ghostMeta.name) + '</strong>' +
      escapeHtml(ghostMeta.description);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function elapsedSec(now) {
    if (!runStarted) return 0;
    if (world.state === 'playing') return (now - runStartMs) / 1000;
    return (runEndMs - runStartMs) / 1000;
  }

  function updateHud(now) {
    const elapsed = elapsedSec(now);
    elTime.textContent = formatTimeSec(elapsed);
    elCoins.textContent = world.coinCount;

    if (world.ghostEnabled) {
      const dxTiles = (world.player.x - world.ghost.x) / TILE;
      const sign = dxTiles >= 0 ? '+' : '−';
      elGap.textContent = sign + Math.abs(dxTiles).toFixed(1);
      elGap.classList.remove('is-ahead', 'is-behind');
      elGap.classList.add(dxTiles >= 0 ? 'is-ahead' : 'is-behind');
    } else {
      elGap.textContent = '—';
      elGap.classList.remove('is-ahead', 'is-behind');
    }
  }

  function showOverlay() {
    overlayTimerId = null;
    const elapsed = (runEndMs - runStartMs) / 1000;
    const cleared = world.state === 'won' && runStarted && runEndMs > runStartMs;
    let isNewBest = false;

    if (cleared) {
      isNewBest = tryRecordBestTime(elapsed);
      renderBestTime();
    }

    elOverTime.textContent = formatTimeSec(elapsed) + 's';
    elOverBest.textContent = bestTimeSec === null ? '—' : formatTimeSec(bestTimeSec) + 's';
    elOverCoins.textContent = world.coinCount;
    elOverRecord.hidden = !isNewBest;

    if (world.state === 'won') {
      if (world.winner === 'player') {
        elOverTitle.textContent = 'You beat the ghost';
        elOverLine.textContent = 'Fastest line wins bragging rights.';
        elOverResult.textContent = 'Win';
        elOverResult.className = 'mono is-win';
      } else if (world.winner === 'tie') {
        elOverTitle.textContent = 'Photo finish';
        elOverLine.textContent = 'Same frame at the flagpole.';
        elOverResult.textContent = 'Tie';
        elOverResult.className = 'mono';
      } else {
        elOverTitle.textContent = 'Level clear';
        elOverLine.textContent = 'You reached the flag.';
        elOverResult.textContent = 'Clear';
        elOverResult.className = 'mono';
      }
    } else {
      elOverTitle.textContent = world.winner === 'ghost'
        ? 'Ghost wins'
        : 'Run over';
      elOverLine.textContent = world.player.alive
        ? 'Try again — press R to reset.'
        : 'You were hit. Press R to reset.';
      elOverResult.textContent = world.winner === 'ghost' ? 'Loss' : '—';
      elOverResult.className = 'mono is-loss';
    }

    elOverlay.hidden = false;
  }

  function loop(now) {
    requestAnimationFrame(loop);
    const dt = now - lastMs;
    lastMs = now;

    if (paused) {
      drawWorld(ctx, world);
      drawPauseHint(ctx);
      return;
    }

    acc += dt;
    if (acc > 250) acc = 250;

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

    if (!runStarted && playerInputs.size > 0) {
      runStarted = true;
      runStartMs = now;
    }

    let ghostInputs = new Set();
    if (world.ghostEnabled && runStarted) {
      try {
        ghostInputs = ghostPolicy(world);
      } catch (err) {
        console.error('[ghost policy error]', err);
        ghostInputs = new Set();
      }
    }

    step(world, playerInputs, ghostInputs);

    if (world.state !== 'playing' && runEndMs === 0) {
      runEndMs = now;
      clearOverlayTimer();
      overlayTimerId = setTimeout(showOverlay, 60);
    }
  }

  function drawPauseHint(ctx) {
    ctx.save();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.fillRect(0, 0, VIEW_W, VIEW_H);
    ctx.fillStyle = '#fff';
    ctx.font = '600 24px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Paused', VIEW_W / 2, VIEW_H / 2 - 10);
    ctx.font = '13px Inter, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    ctx.fillText('P to resume · R to restart', VIEW_W / 2, VIEW_H / 2 + 16);
    ctx.restore();
  }

  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);

  btnRestart.addEventListener('click', restart);
  btnPause.addEventListener('click', togglePause);
  elOverRestart.addEventListener('click', restart);
  elGhostToggle.addEventListener('change', applyGhostToggle);

  canvas.addEventListener('pointerdown', () => canvas.focus());

  applyGhostToggle();
  renderGhostMeta();
  renderBestTime();
  requestAnimationFrame(loop);

})();
