def shape_mario_reward(
    raw_reward,
    x_pos,
    previous_x_pos,
    done,
    progress_reward_scale=0.05,
    idle_penalty=-0.01,
):
    """Return shaped reward, forward progress, and updated x position."""
    progress = 0.0
    if x_pos is not None and previous_x_pos is not None:
        progress = max(0.0, float(x_pos - previous_x_pos))

    next_previous_x_pos = previous_x_pos
    if x_pos is not None:
        next_previous_x_pos = float(x_pos)

    shaped_reward = float(raw_reward) + progress_reward_scale * progress
    if progress <= 0.0 and not done:
        shaped_reward += idle_penalty

    return shaped_reward, progress, next_previous_x_pos
