"""Single-call reproducibility. Every script in scripts/ calls this first,
reading the seed from configs/experiment.yaml's `project.seed` — so a run is
reproducible by construction rather than by remembering to seed everything
by hand in each script.
"""
from __future__ import annotations

import random


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN costs some speed; the plan's timeline is fast-
        # tracked but reproducibility matters more for a paper's numbers than
        # a few percent of training throughput, so this defaults on.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
