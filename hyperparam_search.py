import math
import random

def log_uniform(low, high):
    # sample uniformly in log space
    lo, hi = math.log(low), math.log(high)
    return math.exp(random.uniform(lo, hi))

space_stage1 = {
    "lr": lambda: log_uniform(1e-4, 1e-1),
    "weight_decay": lambda: log_uniform(1e-6, 1e-2),
    "hidden_size": [64, 128, 256],
}

def zoom_space(best_cfgs, lr_factor=3.0, wd_factor=3.0):
    """
    Create a new sampler that draws near the top configs.
    For discrete params, keep the best values (or small set).
    """
    best_hidden = sorted({c["hidden_size"] for c in best_cfgs})
    lrs = [c["lr"] for c in best_cfgs]
    wds = [c["weight_decay"] for c in best_cfgs]

    def lr_sampler():
        center = random.choice(lrs)
        return log_uniform(center / lr_factor, center * lr_factor)

    def wd_sampler():
        center = random.choice(wds)
        return log_uniform(center / wd_factor, center * wd_factor)

    return {
        "lr": lr_sampler,
        "weight_decay": wd_sampler,
        "hidden_size": best_hidden,  # restrict to promising discrete choices
    }

def adaptive_two_stage(space1, train_eval, n1=30, n2=30, budget1=5, budget2=20, top_k=5, seed=0):
    random.seed(seed)

    results1 = []
    for _ in range(n1):
        cfg = random_config(space1)
        loss = train_eval(cfg, budget1)
        results1.append((loss, cfg))

    results1.sort(key=lambda t: t[0])
    top_cfgs = [cfg for _, cfg in results1[:top_k]]

    space2 = zoom_space(top_cfgs)

    results2 = []
    for _ in range(n2):
        cfg = random_config(space2)
        loss = train_eval(cfg, budget2)
        results2.append((loss, cfg))

    results2.sort(key=lambda t: t[0])
    return results2[0][1], results2[0][0], (results1, results2)