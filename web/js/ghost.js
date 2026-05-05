/* ============================================================
   ghost.js — Shadow ghost controller
   ------------------------------------------------------------
   The ghost is just another player entity that runs the same
   physics. Only its inputs change. This file provides two
   "policies" — functions that take the world and return the
   set of buttons held this frame for the ghost:

     scriptedPolicy(world)            — default sample ghost.
                                        Hand-tuned reactive
                                        agent. Always sprints
                                        right, jumps over pits
                                        and goombas.
     trajectoryPolicy(trajectory)     — returns a function that
                                        plays back a per-frame
                                        list of inputs (for use
                                        with a saved RL agent
                                        trajectory).

   The trajectory format is plain JSON:

     {
       "name": "...",
       "description": "...",
       "fps": 60,
       "inputs": [
         { "frame": 0,  "duration": 9999, "keys": ["right","sprint"] },
         { "frame": 64, "duration": 14,   "keys": ["jump"] },
         ...
       ]
     }

   To swap the sample ghost for a real RL trajectory:
     1. Train the DQN agent until it reliably finishes the
        level.
     2. During its best episode, log every frame's chosen
        action in the same format above.
     3. Dump that as data/ghost_trained.json (or any name).
     4. In main.js, change the ghost source to
        Ghost.loadFromJson('data/ghost_trained.json'); — the
        rendering and HUD continue to work unchanged.
   ============================================================ */

(function () {
  'use strict';

  // ---- Sample ghost metadata (shown in the sidebar) ----------
  const SAMPLE_META = {
    name: 'Sample Ghost',
    description:
      'Hand-tuned scripted run. Replace with the RL agent’s ' +
      'trajectory once training is finished.',
    fps: 60,
    source: 'scripted',
  };

  // ============================================================
  //   Scripted policy
  //   Reactive ghost: sprints right, jumps over upcoming pits
  //   and goombas. Coupled to the level layout in game.js but
  //   that's expected for a scripted demo.
  // ============================================================

  // x-coordinates (in pixels) where the ghost should jump.
  // Each window is wide enough that sprint variability between
  // runs still triggers a jump at the right spot.
  const JUMP_WINDOWS = [
    // [enter, exit, holdFrames]
    [385, 430, 14],   // first 3-tile pit (cols 13-15)
    [720, 760, 14],   // pipe at cols 24-25
    [1010, 1055, 12], // 2-tile pit (cols 33-34)
    [1660, 1710, 14], // 3-tile pit (cols 53-55)
  ];

  function scriptedPolicy(world) {
    const inputs = new Set(['right', 'sprint']);
    const g = world.ghost;

    if (!g.alive) return inputs;

    const px = g.x;

    // Positional jumps for pits and pipe
    for (const [lo, hi] of JUMP_WINDOWS) {
      if (px >= lo && px <= hi && g.onGround) {
        inputs.add('jump');
        g._scriptedJumpFrames = 0;
        break;
      }
    }
    // Sustain jump while rising (variable jump height)
    if (g.vy < -3 && (g._scriptedJumpFrames || 0) < 14) {
      inputs.add('jump');
      g._scriptedJumpFrames = (g._scriptedJumpFrames || 0) + 1;
    }

    // Reactive: jump if a live goomba is just ahead and we're on ground
    if (g.onGround) {
      for (const goomba of world.goombas) {
        if (!goomba.alive) continue;
        const dx = goomba.x - px;
        if (dx > 8 && dx < 80 && Math.abs(goomba.y - g.y) < 40) {
          inputs.add('jump');
          g._scriptedJumpFrames = 0;
          break;
        }
      }
    }

    return inputs;
  }

  // ============================================================
  //   Trajectory policy
  //   Plays back a list of {frame, duration, keys} entries.
  //   Used for swapping in a real RL agent.
  // ============================================================

  function buildFrameTable(entries, maxFrame = 7200) {
    // Convert sparse entries into a per-frame array of Set<string>.
    // maxFrame: 7200 = 2 minutes at 60fps.
    const table = Array.from({ length: maxFrame }, () => new Set());
    for (const e of entries) {
      const start = Math.max(0, e.frame | 0);
      const end = Math.min(maxFrame, start + (e.duration | 0));
      for (let f = start; f < end; f++) {
        for (const k of e.keys) table[f].add(k);
      }
    }
    return table;
  }

  function trajectoryPolicy(trajectory) {
    const table = buildFrameTable(trajectory.inputs || []);
    return function (world) {
      const f = world.frame;
      if (f < 0 || f >= table.length) return new Set();
      return table[f];
    };
  }

  // ============================================================
  //   JSON loading helper
  //   Returns a promise that resolves to a {policy, meta} pair.
  //   Falls back to the scripted policy if the fetch fails.
  // ============================================================

  function loadFromJson(url) {
    return fetch(url, { cache: 'no-store' })
      .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(data => ({
        policy: trajectoryPolicy(data),
        meta: {
          name: data.name || 'Trajectory',
          description: data.description || '',
          fps: data.fps || 60,
          source: 'json:' + url,
        },
      }))
      .catch(err => {
        console.warn('[ghost] Failed to load', url, '-> using scripted policy.', err);
        return {
          policy: scriptedPolicy,
          meta: { ...SAMPLE_META, description: SAMPLE_META.description +
            ' (JSON load failed; using fallback.)' },
        };
      });
  }

  // ============================================================
  //   Public API
  // ============================================================
  window.Ghost = {
    SAMPLE_META,
    scriptedPolicy,
    trajectoryPolicy,
    loadFromJson,
  };

})();
