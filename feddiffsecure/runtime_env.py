from __future__ import annotations

import os


# Prefer GNU threading to match the libgomp runtime commonly used by PyTorch
# on shared Linux servers, and import numpy before torch to avoid MKL init
# conflicts in CLI entrypoints.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: F401
