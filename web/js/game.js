/* ============================================================
   game.js — MARIO platformer engine
   ------------------------------------------------------------
   Tile-based side-scroller with deterministic physics that is
   shared between the keyboard-controlled player and the
   ghost. Exposes a single `Game` object on window.
   ============================================================ */

(function () {
  'use strict';

  const TILE = 32;
  const VIEW_W = 768;
  const VIEW_H = 448;

  // --- Level layout ---------------------------------------------
  // The level is built programmatically rather than from an ASCII
  // map so that pits, pipes, and goomba placements are guaranteed
  // to line up. Tile codes (these are the values stored after
  // build):
  //
  //   . = air            # = ground (solid)
  //   B = brick (solid)  ? = question block (solid)
  //   p = pipe top-L     q = pipe top-R
  //   P = pipe body-L    Q = pipe body-R
  //   C = coin           G = goomba spawn (extracted at load)
  //   | = flagpole       F = flag base (solid)
  //
  // Ground rows are 11..13. Player spawn sits on top of row 11.

  const LEVEL_W = 90;
  const LEVEL_H = 14;
  const GROUND_ROW = 11;

  // Ground segments (inclusive col ranges) and pits (gaps in between).
  const GROUND_SEGMENTS = [
    [0, 12],   // start
    [16, 32],  // pipe section
    [35, 52],  // mid plain
    [56, 87],  // run-up to flag
    [88, 89],  // flag base
  ];
  // Resulting pits: 13-15, 33-34, 53-55

  // Pipe positions (top-row index, left col, height in tiles)
  const PIPES = [
    { topRow: 9, col: 24, height: 2 },
  ];

  // Floating brick platforms: { row, c1, c2 }
  const BRICK_PLATFORMS = [
    { row: 9, c1: 1, c2: 4 },
    { row: 5, c1: 21, c2: 23 },
    { row: 5, c1: 25, c2: 27 },
    { row: 9, c1: 38, c2: 41 },
    { row: 9, c1: 65, c2: 68 },
    { row: 9, c1: 70, c2: 75 },
    { row: 9, c1: 80, c2: 83 },
  ];

  // Single ? blocks
  const QUESTION_BLOCKS = [
    { row: 5, col: 8 },
    { row: 5, col: 24 },
  ];

  // Coins
  const COIN_TILES = [
    { row: 8, col: 11 }, { row: 8, col: 12 },
    { row: 4, col: 22 }, { row: 4, col: 23 }, { row: 4, col: 24 },
    { row: 7, col: 30 }, { row: 7, col: 31 }, { row: 7, col: 32 },
    { row: 8, col: 41 }, { row: 8, col: 42 },
    { row: 7, col: 70 }, { row: 7, col: 71 }, { row: 7, col: 72 },
    { row: 8, col: 81 }, { row: 8, col: 82 },
  ];

  // Goomba spawns (must be just above ground or on a platform)
  const GOOMBA_SPAWNS = [
    { row: 10, col: 19 },
    { row: 10, col: 39 },
    { row: 10, col: 45 },
    { row: 10, col: 60 },
    { row: 10, col: 62 },
    { row: 10, col: 78 },
    { row: 10, col: 82 },
  ];

  // Flag column (pole)
  const FLAG_COL = 88;

  // --- Tile classification --------------------------------------
  function isSolid(ch) {
    return ch === '#' || ch === 'B' || ch === '?' ||
           ch === 'p' || ch === 'q' || ch === 'P' || ch === 'Q' ||
           ch === 'F';
  }

  // --- Level object ---------------------------------------------
  function buildLevel() {
    // Initialise all-air grid
    const tiles = Array.from({ length: LEVEL_H }, () =>
      Array(LEVEL_W).fill('.'));

    // Ground bands
    for (const [c1, c2] of GROUND_SEGMENTS) {
      for (let c = c1; c <= c2; c++) {
        for (let r = GROUND_ROW; r < LEVEL_H; r++) {
          tiles[r][c] = '#';
        }
      }
    }

    // Pipes
    for (const p of PIPES) {
      // Top row uses pq, body rows use PQ
      tiles[p.topRow][p.col]     = 'p';
      tiles[p.topRow][p.col + 1] = 'q';
      for (let i = 1; i < p.height; i++) {
        tiles[p.topRow + i][p.col]     = 'P';
        tiles[p.topRow + i][p.col + 1] = 'Q';
      }
    }

    // Brick platforms
    for (const b of BRICK_PLATFORMS) {
      for (let c = b.c1; c <= b.c2; c++) tiles[b.row][c] = 'B';
    }

    // ? blocks
    for (const q of QUESTION_BLOCKS) tiles[q.row][q.col] = '?';

    // Flag pole (visual only, non-solid)
    for (let r = 4; r <= GROUND_ROW - 1; r++) tiles[r][FLAG_COL] = '|';

    return {
      tiles,
      width: LEVEL_W,
      height: LEVEL_H,
      pixelWidth: LEVEL_W * TILE,
      pixelHeight: LEVEL_H * TILE,
      goombaSpawns: GOOMBA_SPAWNS.slice(),
      coinTiles: COIN_TILES.slice(),
      flagX: FLAG_COL * TILE,
    };
  }

  // --- Solid-tile lookup for collision --------------------------
  function tileAt(level, col, row) {
    if (col < 0 || row < 0 || col >= level.width || row >= level.height) {
      return col < 0 ? '#' : '.';        // wall on left edge, air everywhere else off-grid
    }
    return level.tiles[row][col];
  }

  function isSolidAt(level, col, row) {
    return isSolid(tileAt(level, col, row));
  }

  // --- Entity factory -------------------------------------------
  function createPlayer(opts) {
    return {
      x: opts.x, y: opts.y,
      vx: 0, vy: 0,
      w: 22, h: 30,
      facing: 1,
      onGround: false,
      coyote: 0,
      jumpHeld: false,
      jumpHoldFrames: 0,
      alive: true,
      isGhost: !!opts.isGhost,
      runFrame: 0,        // animation timer
    };
  }

  function createGoomba(col, row) {
    return {
      x: col * TILE,
      y: row * TILE + (TILE - 24),
      vx: -0.6,
      vy: 0,
      w: 28, h: 24,
      alive: true,
      squashTimer: 0,
    };
  }

  // --- Physics --------------------------------------------------
  // Inputs is a Set-like object: inputs.has('left'|'right'|'jump'|'sprint')
  const PHYS = {
    accel: 0.45,
    friction: 0.32,
    walkMax: 3.0,
    runMax: 5.4,
    gravity: 0.55,
    maxFall: 13,
    jumpV: -9.5,
    jumpHoldExtra: -0.25,   // extra upward force per frame while held
    jumpHoldMaxFrames: 12,
    coyoteFrames: 5,
  };

  function stepPlayer(player, inputs, level) {
    if (!player.alive) {
      // dead players still fall
      player.vy = Math.min(player.vy + PHYS.gravity, PHYS.maxFall);
      player.y += player.vy;
      return;
    }

    const left = inputs.has('left');
    const right = inputs.has('right');
    const jump = inputs.has('jump');
    const sprint = inputs.has('sprint');

    const targetMax = sprint ? PHYS.runMax : PHYS.walkMax;

    // Horizontal accel
    if (left && !right) {
      player.vx -= PHYS.accel;
      player.facing = -1;
    } else if (right && !left) {
      player.vx += PHYS.accel;
      player.facing = 1;
    } else {
      // friction
      if (player.vx > 0)       player.vx = Math.max(0, player.vx - PHYS.friction);
      else if (player.vx < 0)  player.vx = Math.min(0, player.vx + PHYS.friction);
    }

    // Clamp horizontal speed
    if (player.vx > targetMax)  player.vx = targetMax;
    if (player.vx < -targetMax) player.vx = -targetMax;

    // Jump (with coyote time + variable height)
    const justPressedJump = jump && !player.jumpHeld;
    player.jumpHeld = jump;

    if (justPressedJump && (player.onGround || player.coyote > 0)) {
      player.vy = PHYS.jumpV;
      player.onGround = false;
      player.coyote = 0;
      player.jumpHoldFrames = 0;
    } else if (jump && player.vy < 0 && player.jumpHoldFrames < PHYS.jumpHoldMaxFrames) {
      player.vy += PHYS.jumpHoldExtra;
      player.jumpHoldFrames++;
    } else if (!jump) {
      player.jumpHoldFrames = PHYS.jumpHoldMaxFrames; // disable hold once released
    }

    // Gravity
    player.vy += PHYS.gravity;
    if (player.vy > PHYS.maxFall) player.vy = PHYS.maxFall;

    // --- Move + collide axis-by-axis ---
    moveX(player, level);
    moveY(player, level);

    // Animation timer
    if (Math.abs(player.vx) > 0.1 && player.onGround) {
      player.runFrame += Math.abs(player.vx) * 0.15;
    }

    // Coyote time
    if (player.onGround) player.coyote = PHYS.coyoteFrames;
    else if (player.coyote > 0) player.coyote--;
  }

  function moveX(p, level) {
    const newX = p.x + p.vx;
    if (p.vx === 0) { p.x = newX; return; }

    const dir = p.vx > 0 ? 1 : -1;
    const probeX = dir > 0 ? newX + p.w : newX;
    const colHit = Math.floor(probeX / TILE);

    const top = Math.floor(p.y / TILE);
    const bot = Math.floor((p.y + p.h - 1) / TILE);

    let blocked = false;
    for (let r = top; r <= bot; r++) {
      if (isSolidAt(level, colHit, r)) { blocked = true; break; }
    }

    if (blocked) {
      if (dir > 0) p.x = colHit * TILE - p.w - 0.001;
      else         p.x = (colHit + 1) * TILE + 0.001;
      p.vx = 0;
    } else {
      p.x = newX;
    }
  }

  function moveY(p, level) {
    const newY = p.y + p.vy;
    p.onGround = false;

    if (p.vy === 0) { p.y = newY; return; }

    const dir = p.vy > 0 ? 1 : -1;
    const probeY = dir > 0 ? newY + p.h : newY;
    const rowHit = Math.floor(probeY / TILE);

    const left = Math.floor(p.x / TILE);
    const right = Math.floor((p.x + p.w - 1) / TILE);

    let blocked = false;
    for (let c = left; c <= right; c++) {
      if (isSolidAt(level, c, rowHit)) { blocked = true; break; }
    }

    if (blocked) {
      if (dir > 0) {
        p.y = rowHit * TILE - p.h - 0.001;
        p.onGround = true;
      } else {
        p.y = (rowHit + 1) * TILE + 0.001;
      }
      p.vy = 0;
    } else {
      p.y = newY;
    }
  }

  // --- Goomba step ----------------------------------------------
  function stepGoomba(g, level) {
    if (!g.alive) {
      if (g.squashTimer > 0) g.squashTimer--;
      return;
    }
    g.vy += PHYS.gravity;
    if (g.vy > PHYS.maxFall) g.vy = PHYS.maxFall;

    // Horizontal move with edge / wall check
    const newX = g.x + g.vx;
    const probeCol = Math.floor((g.vx > 0 ? newX + g.w : newX) / TILE);
    const top = Math.floor(g.y / TILE);
    const bot = Math.floor((g.y + g.h - 1) / TILE);

    let wallBlock = false;
    for (let r = top; r <= bot; r++) {
      if (isSolidAt(level, probeCol, r)) { wallBlock = true; break; }
    }

    if (wallBlock) g.vx = -g.vx;
    else g.x = newX;

    // Vertical move
    moveY(g, level);

    // Off-bottom: kill it
    if (g.y > level.pixelHeight + 100) g.alive = false;
  }

  // --- Player vs goomba collision -------------------------------
  function aabb(a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x &&
           a.y < b.y + b.h && a.y + a.h > b.y;
  }

  function resolvePlayerVsGoombas(player, goombas) {
    if (!player.alive || player.isGhost) return { stomped: 0, died: false };

    let stomped = 0;
    let died = false;

    for (const g of goombas) {
      if (!g.alive) continue;
      if (!aabb(player, g)) continue;

      // If player is descending and bottom of player is above goomba's middle => stomp
      const playerBottom = player.y + player.h;
      const stomp = player.vy > 0.5 && playerBottom < g.y + g.h * 0.6;

      if (stomp) {
        g.alive = false;
        g.squashTimer = 18;
        player.vy = -7;  // bounce
        stomped++;
      } else {
        died = true;
      }
    }

    return { stomped, died };
  }

  // --- Coin collection ------------------------------------------
  function resolveCoins(player, coins) {
    if (player.isGhost) return 0;
    let collected = 0;
    for (const c of coins) {
      if (c.taken) continue;
      const cx = c.col * TILE + 6, cy = c.row * TILE + 6;
      const cw = TILE - 12, ch = TILE - 12;
      if (player.x < cx + cw && player.x + player.w > cx &&
          player.y < cy + ch && player.y + player.h > cy) {
        c.taken = true;
        collected++;
      }
    }
    return collected;
  }

  // --- Camera ---------------------------------------------------
  function updateCamera(camera, player, level) {
    const targetX = player.x - VIEW_W / 3;
    camera.x += (targetX - camera.x) * 0.18;
    if (camera.x < 0) camera.x = 0;
    const maxX = level.pixelWidth - VIEW_W;
    if (camera.x > maxX) camera.x = maxX;
  }

  // ============================================================
  //   RENDERING
  // ============================================================

  function drawWorld(ctx, world) {
    drawBackground(ctx, world);
    drawTiles(ctx, world);
    drawCoins(ctx, world);
    drawFlag(ctx, world);
    for (const g of world.goombas) drawGoomba(ctx, g, world.camera);

    // Ghost first (behind player), then player
    if (world.ghost && world.ghostVisible) drawPlayer(ctx, world.ghost, world.camera, true);
    drawPlayer(ctx, world.player, world.camera, false);
  }

  function drawBackground(ctx, world) {
    // Sky is the canvas background (CSS), draw clouds + hills
    const cam = world.camera.x;

    // Distant hills
    ctx.fillStyle = '#1f8f3a';
    for (let i = 0; i < 8; i++) {
      const hx = i * 420 - (cam * 0.3) % 420;
      drawHill(ctx, hx, VIEW_H - 96, 220, 80);
    }

    // Clouds
    ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
    const clouds = [
      [80, 60], [320, 90], [560, 50], [820, 110], [1100, 70],
      [1380, 95], [1700, 60], [2000, 100], [2280, 70], [2580, 85],
    ];
    for (const [cx, cy] of clouds) {
      const sx = cx - cam * 0.5;
      if (sx < -120 || sx > VIEW_W + 120) continue;
      drawCloud(ctx, sx, cy);
    }
  }

  function drawHill(ctx, x, y, w, h) {
    ctx.beginPath();
    ctx.moveTo(x, y + h);
    ctx.quadraticCurveTo(x + w / 2, y - h * 0.4, x + w, y + h);
    ctx.closePath();
    ctx.fill();
  }

  function drawCloud(ctx, x, y) {
    ctx.beginPath();
    ctx.arc(x, y, 18, 0, Math.PI * 2);
    ctx.arc(x + 22, y - 8, 22, 0, Math.PI * 2);
    ctx.arc(x + 48, y, 20, 0, Math.PI * 2);
    ctx.arc(x + 28, y + 12, 18, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawTiles(ctx, world) {
    const { level, camera } = world;
    const startCol = Math.max(0, Math.floor(camera.x / TILE));
    const endCol = Math.min(level.width - 1, Math.ceil((camera.x + VIEW_W) / TILE));

    for (let r = 0; r < level.height; r++) {
      for (let c = startCol; c <= endCol; c++) {
        const ch = level.tiles[r][c];
        if (ch === '.') continue;
        const x = Math.round(c * TILE - camera.x);
        const y = r * TILE;
        drawTile(ctx, ch, x, y);
      }
    }
  }

  function drawTile(ctx, ch, x, y) {
    switch (ch) {
      case '#': {
        // Ground block
        ctx.fillStyle = '#c84e1a';
        ctx.fillRect(x, y, TILE, TILE);
        ctx.fillStyle = '#8b3210';
        ctx.fillRect(x, y + TILE - 4, TILE, 4);
        ctx.fillStyle = 'rgba(0,0,0,0.18)';
        ctx.fillRect(x, y, 2, TILE);
        ctx.fillRect(x + TILE - 2, y, 2, TILE);
        // Brick pattern
        ctx.fillStyle = 'rgba(0,0,0,0.22)';
        ctx.fillRect(x, y + TILE / 2 - 1, TILE, 2);
        ctx.fillRect(x + TILE / 2 - 1, y, 2, TILE / 2);
        ctx.fillRect(x + TILE / 4 - 1, y + TILE / 2, 2, TILE / 2);
        ctx.fillRect(x + (TILE * 3) / 4 - 1, y + TILE / 2, 2, TILE / 2);
        break;
      }
      case 'B': {
        // Brick block
        ctx.fillStyle = '#d96930';
        ctx.fillRect(x, y, TILE, TILE);
        ctx.fillStyle = '#7a2f10';
        ctx.fillRect(x, y, TILE, 3);
        ctx.fillRect(x, y + TILE - 3, TILE, 3);
        ctx.fillRect(x, y, 3, TILE);
        ctx.fillRect(x + TILE - 3, y, 3, TILE);
        ctx.fillStyle = 'rgba(0,0,0,0.25)';
        ctx.fillRect(x, y + TILE / 2, TILE, 2);
        break;
      }
      case '?': {
        // Question block
        ctx.fillStyle = '#e09e1a';
        ctx.fillRect(x, y, TILE, TILE);
        ctx.fillStyle = '#7a4a00';
        ctx.fillRect(x, y, TILE, 3);
        ctx.fillRect(x, y + TILE - 3, TILE, 3);
        ctx.fillRect(x, y, 3, TILE);
        ctx.fillRect(x + TILE - 3, y, 3, TILE);
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 22px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('?', x + TILE / 2, y + TILE / 2 + 1);
        break;
      }
      case 'p':
      case 'q': {
        // Pipe top
        ctx.fillStyle = '#1f8f3a';
        ctx.fillRect(x, y, TILE, TILE);
        ctx.fillStyle = '#0e5e23';
        if (ch === 'p') {
          ctx.fillRect(x, y, 4, TILE);
          ctx.fillRect(x, y, TILE, 4);
        } else {
          ctx.fillRect(x + TILE - 4, y, 4, TILE);
          ctx.fillRect(x, y, TILE, 4);
        }
        // Lip overhang
        const lipX = ch === 'p' ? x - 4 : x;
        ctx.fillStyle = '#26b04a';
        ctx.fillRect(lipX, y, TILE + 4, 6);
        ctx.fillStyle = '#0e5e23';
        ctx.fillRect(lipX, y, TILE + 4, 2);
        break;
      }
      case 'P':
      case 'Q': {
        ctx.fillStyle = '#1f8f3a';
        ctx.fillRect(x, y, TILE, TILE);
        ctx.fillStyle = '#0e5e23';
        if (ch === 'P') ctx.fillRect(x, y, 4, TILE);
        else            ctx.fillRect(x + TILE - 4, y, 4, TILE);
        break;
      }
      case 'F': {
        // Flag base
        ctx.fillStyle = '#2c3e50';
        ctx.fillRect(x, y, TILE, TILE);
        ctx.fillStyle = '#1a252f';
        ctx.fillRect(x, y + TILE - 4, TILE, 4);
        break;
      }
    }
  }

  function drawCoins(ctx, world) {
    const cam = world.camera.x;
    for (const c of world.coins) {
      if (c.taken) continue;
      const x = c.col * TILE + TILE / 2 - cam;
      const y = c.row * TILE + TILE / 2;
      if (x < -20 || x > VIEW_W + 20) continue;

      // Spin animation: oscillate width
      const phase = (world.frame * 0.08 + c.col * 0.4) % (Math.PI * 2);
      const w = Math.abs(Math.cos(phase)) * 9 + 2;

      ctx.fillStyle = '#ffd166';
      ctx.beginPath();
      ctx.ellipse(x, y, w, 11, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#7a4a00';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  function drawFlag(ctx, world) {
    const cam = world.camera.x;
    const flagCol = Math.floor(world.level.flagX / TILE);
    const px = flagCol * TILE + 14 - cam;
    const topY = 4 * TILE;
    const botY = 11 * TILE;
    if (px < -40 || px > VIEW_W + 40) return;

    // Pole
    ctx.fillStyle = '#cccccc';
    ctx.fillRect(px, topY, 4, botY - topY);
    ctx.fillStyle = '#888';
    ctx.fillRect(px - 6, topY - 8, 16, 8);

    // Flag fabric, animated wave
    const flagY = topY + 14 + Math.sin(world.frame * 0.06) * 2;
    ctx.fillStyle = '#06d6a0';
    ctx.beginPath();
    ctx.moveTo(px + 4, flagY);
    ctx.lineTo(px + 32, flagY + 8);
    ctx.lineTo(px + 4, flagY + 18);
    ctx.closePath();
    ctx.fill();
  }

  function drawGoomba(ctx, g, camera) {
    const x = g.x - camera.x;
    const y = g.y;
    if (x < -50 || x > VIEW_W + 20) return;

    if (!g.alive) {
      // Squashed sprite
      if (g.squashTimer > 0) {
        ctx.fillStyle = '#8b4513';
        ctx.fillRect(x + 2, y + g.h - 8, g.w - 4, 8);
      }
      return;
    }

    // Body
    ctx.fillStyle = '#8b4513';
    ctx.beginPath();
    ctx.ellipse(x + g.w / 2, y + g.h / 2 + 2, g.w / 2, g.h / 2, 0, 0, Math.PI * 2);
    ctx.fill();

    // Feet
    ctx.fillStyle = '#5a2c0d';
    ctx.fillRect(x + 2, y + g.h - 4, 8, 4);
    ctx.fillRect(x + g.w - 10, y + g.h - 4, 8, 4);

    // Eyes
    ctx.fillStyle = 'white';
    ctx.fillRect(x + 6, y + 7, 6, 7);
    ctx.fillRect(x + g.w - 12, y + 7, 6, 7);
    ctx.fillStyle = 'black';
    ctx.fillRect(x + 8, y + 9, 3, 4);
    ctx.fillRect(x + g.w - 10, y + 9, 3, 4);
  }

  function drawPlayer(ctx, p, camera, isGhost) {
    const x = Math.round(p.x - camera.x);
    const y = Math.round(p.y);

    ctx.save();
    if (isGhost) ctx.globalAlpha = 0.55;

    // Mario-style figure rendered as a small layered shape.
    // Layout: 22x30 bounding box.
    //   Hat   : top 8px, red
    //   Face  : middle 10px, peach
    //   Body  : bottom 12px, blue overalls + red shirt
    const hatColor   = isGhost ? '#06d6a0' : '#e63946';
    const faceColor  = isGhost ? '#bff5e3' : '#ffcfa3';
    const shirtColor = isGhost ? '#06d6a0' : '#e63946';
    const overColor  = isGhost ? '#0a8460' : '#264e9c';
    const shoeColor  = isGhost ? '#054d3a' : '#3a1d05';
    const skinShade  = isGhost ? '#7fd9bd' : '#c8895c';

    // Hat
    ctx.fillStyle = hatColor;
    ctx.fillRect(x + 2, y, 18, 7);
    ctx.fillRect(x, y + 5, 22, 4);

    // Hat brim shadow
    ctx.fillStyle = 'rgba(0,0,0,0.18)';
    ctx.fillRect(x, y + 8, 22, 1);

    // Face
    ctx.fillStyle = faceColor;
    ctx.fillRect(x + 3, y + 9, 16, 9);

    // Eye
    ctx.fillStyle = '#1a1a1a';
    if (p.facing >= 0) {
      ctx.fillRect(x + 13, y + 12, 2, 3);
    } else {
      ctx.fillRect(x + 7, y + 12, 2, 3);
    }

    // Mustache
    ctx.fillStyle = skinShade;
    ctx.fillRect(x + 4, y + 16, 14, 2);

    // Shirt sleeves
    ctx.fillStyle = shirtColor;
    ctx.fillRect(x, y + 18, 22, 5);

    // Overalls body
    ctx.fillStyle = overColor;
    ctx.fillRect(x + 3, y + 21, 16, 7);

    // Overall straps
    ctx.fillStyle = overColor;
    ctx.fillRect(x + 6, y + 18, 3, 4);
    ctx.fillRect(x + 13, y + 18, 3, 4);

    // Shoes
    ctx.fillStyle = shoeColor;
    const stride = p.onGround && Math.abs(p.vx) > 0.1
      ? Math.sin(p.runFrame) * 2
      : 0;
    ctx.fillRect(x + 2,  y + 27 + (stride > 0 ? -1 : 0), 8, 3);
    ctx.fillRect(x + 12, y + 27 + (stride < 0 ? -1 : 0), 8, 3);

    ctx.restore();
  }

  // ============================================================
  //   World construction + reset
  // ============================================================
  function buildWorld() {
    const level = buildLevel();

    // Player spawn at first ground column, on top of ground.
    const groundRow = 11;
    const spawn = { x: 2 * TILE, y: groundRow * TILE - 30 - 2 };

    const player = createPlayer({ x: spawn.x, y: spawn.y, isGhost: false });
    const ghost  = createPlayer({ x: spawn.x, y: spawn.y, isGhost: true });

    const goombas = level.goombaSpawns.map(s => createGoomba(s.col, s.row));
    const coins   = level.coinTiles.map(c => ({ ...c, taken: false }));

    return {
      level,
      player, ghost,
      ghostVisible: true,
      ghostEnabled: true,
      goombas, coins,
      camera: { x: 0, y: 0 },
      frame: 0,
      spawn,
      coinCount: 0,
      score: 0,
      state: 'playing',           // 'playing' | 'won' | 'lost'
      winner: null,               // 'player' | 'ghost' | 'tie' | null
      playerFinishedAt: null,     // frame index when player crossed flag
      ghostFinishedAt: null,      // frame index when ghost crossed flag
    };
  }

  function resetWorld(world) {
    const fresh = buildWorld();
    world.level = fresh.level;
    world.player = fresh.player;
    world.ghost = fresh.ghost;
    world.goombas = fresh.goombas;
    world.coins = fresh.coins;
    world.camera = fresh.camera;
    world.frame = 0;
    world.coinCount = 0;
    world.score = 0;
    world.state = 'playing';
    world.winner = null;
    world.playerFinishedAt = null;
    world.ghostFinishedAt = null;
  }

  // ============================================================
  //   Per-frame step (does NOT handle input — caller passes sets)
  // ============================================================
  function step(world, playerInputs, ghostInputs) {
    if (world.state !== 'playing') return null;

    world.frame++;

    // Player + ghost physics
    stepPlayer(world.player, playerInputs, world.level);
    if (world.ghostEnabled) stepPlayer(world.ghost, ghostInputs, world.level);

    // Enemies
    for (const g of world.goombas) stepGoomba(g, world.level);

    // Player vs goomba
    const collide = resolvePlayerVsGoombas(world.player, world.goombas);
    if (collide.died) world.player.alive = false;
    if (collide.stomped) world.score += collide.stomped * 100;

    // Coins
    const got = resolveCoins(world.player, world.coins);
    if (got) {
      world.coinCount += got;
      world.score += got * 200;
    }

    // Death by pit
    if (world.player.y > world.level.pixelHeight + 80) {
      world.player.alive = false;
    }
    // Ghost falling off world: stop simulating it
    if (world.ghost.y > world.level.pixelHeight + 80) {
      world.ghost.alive = false;
    }

    // Camera
    updateCamera(world.camera, world.player, world.level);

    // Track flag crossings (don't overwrite earlier crossings)
    const playerAtFlag = world.player.x + world.player.w > world.level.flagX;
    const ghostAtFlag  = world.ghostEnabled &&
                          world.ghost.x + world.ghost.w > world.level.flagX;

    if (playerAtFlag && world.playerFinishedAt === null && world.player.alive) {
      world.playerFinishedAt = world.frame;
    }
    if (ghostAtFlag && world.ghostFinishedAt === null) {
      world.ghostFinishedAt = world.frame;
    }

    // End condition
    if (!world.player.alive) {
      world.state = 'lost';
      world.winner = world.ghostFinishedAt !== null ? 'ghost' : null;
    } else if (world.playerFinishedAt !== null) {
      world.state = 'won';
      if (world.ghostFinishedAt === null) {
        world.winner = 'player';
      } else if (world.playerFinishedAt < world.ghostFinishedAt) {
        world.winner = 'player';
      } else if (world.playerFinishedAt > world.ghostFinishedAt) {
        world.winner = 'ghost';
      } else {
        world.winner = 'tie';
      }
    }

    return null;
  }

  // ============================================================
  //   Public API
  // ============================================================
  window.Game = {
    TILE, VIEW_W, VIEW_H,
    buildWorld, resetWorld, step,
    drawWorld,
  };

})();
