"""Set private compiler caches before importing the frozen extension runner."""

import os
from pathlib import Path


cache = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / "strange-gradient-extension-cache"
os.environ["NUMBA_CACHE_DIR"] = str(cache / f"process_{os.getpid()}")
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"

import extend_mot_strange_gradient_loading as campaign


if __name__ == "__main__":
    campaign.main()
