"""Paths, constants and hyperparameters shared across the pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROCESSED_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SYNTHETIC_MARKER = RAW_DIR / "SYNTHETIC_DATA_MARKER"
SYNTHETIC_WARNING = (
    "SYNTHETIC data: the real MovieLens 1M download failed or was blocked, "
    "so this run used the schema-identical synthetic fallback instead"
)

RANDOM_SEED = 42
TOP_N = 10

# Implicit-feedback threshold: ratings at or above this count as a positive interaction.
POSITIVE_THRESHOLD = 4

# A user needs at least this many positives to get held-out val/test slices;
# below that they stay train-only (not enough signal to evaluate on fairly).
MIN_POSITIVES_FOR_SPLIT = 15
N_TEST_PER_USER = 5
N_VAL_PER_USER = 5

# ALS hyperparameters: factors/alpha selected from the one-at-a-time grid in
# `python -m src.evaluate tune` (see reports/hyperparam_sensitivity.md) on the
# real MovieLens 1M validation slice; reg was already near-optimal at the
# starting value. Recall@10 improved monotonically as alpha decreased across
# the whole grid explored (1 to 80), unlike the mid-range optimum a purely
# synthetic benchmark would suggest; settled on alpha=1 given diminishing
# returns from searching further below it.
ALS_FACTORS = 64
ALS_REG = 0.1
ALS_ALPHA = 1.0
ALS_ITERATIONS = 15

ITEM_KNN_TOP_K = 100

BOOTSTRAP_RESAMPLES = 1000
