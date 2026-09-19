"""
geometry_utils.py
==================
Shared world-frame axis convention for this project's pipeline scripts.

Stray Scanner scans in this project use world frame Y = up (array column
index 1). X (0) and Z (2) span the horizontal ground plane. Verified via
PCA on clean meshes -- see 04_trim_mesh.py's estimate_gravity() (do not
modify that file), whose docstring states:
    "Stray Scanner su dung he toa do world Y=up (xac nhan qua PCA cac
     mesh sach)" -> gravity_down = [0, -1, 0].
This module exists so "which array column is vertical" is defined in one
place instead of being re-guessed (and sometimes mis-guessed as column 2)
in every script that needs it.

Used by: 05_trunk_isolation.py, 06_slope_analysis.py, diagnostic_trunk_axis.py.
Deliberately NOT used by 04_trim_mesh.py, 04b_finalize_mesh.py,
05_detect_lichen.py, 05_detect_lichen_v2.py -- those already have their own
correct, independent Y-up logic; migrating them is a pure-DRY refactor of
already-working code with zero bug-fixing benefit (out of scope here).
"""

import numpy as np

UP_AXIS = 1                                  # world column index that is vertical (Y)
HORIZONTAL_AXES = (0, 2)                     # world column indices spanning the ground plane (X, Z)
GRAVITY_DOWN = np.array([0.0, -1.0, 0.0])    # matches 04_trim_mesh.py estimate_gravity()
GRAVITY_UP = np.array([0.0, 1.0, 0.0])


def vertical(points: np.ndarray) -> np.ndarray:
    """(N,3) world points -> (N,) vertical/height coordinate (column Y)."""
    return points[:, UP_AXIS]


def horizontal(points: np.ndarray) -> np.ndarray:
    """(N,3) world points -> (N,2) horizontal-plane coordinates, order (X, Z)."""
    return points[:, HORIZONTAL_AXES]


def horizontal_center(points: np.ndarray) -> tuple[float, float]:
    """Median (x, z) center of `points` on the horizontal plane."""
    xz = horizontal(points)
    return float(np.median(xz[:, 0])), float(np.median(xz[:, 1]))


def horizontal_distance(points: np.ndarray, center_xz) -> np.ndarray:
    """(N,) distance from each point to `center_xz` (an (x, z) pair), on the horizontal plane."""
    d = horizontal(points) - np.asarray(center_xz, dtype=float)
    return np.linalg.norm(d, axis=1)


def embed_horizontal(xz: np.ndarray) -> np.ndarray:
    """Inverse of horizontal() for direction/offset vectors: embed an (N,2)
    or (2,) [x, z] array into full 3D with the vertical component = 0."""
    xz = np.asarray(xz, dtype=float)
    single = xz.ndim == 1
    if single:
        xz = xz[None, :]
    out = np.zeros((xz.shape[0], 3))
    out[:, HORIZONTAL_AXES[0]] = xz[:, 0]
    out[:, HORIZONTAL_AXES[1]] = xz[:, 1]
    return out[0] if single else out


def project_to_horizontal(vec3: np.ndarray) -> np.ndarray:
    """Copy of a world-frame 3-vector with its vertical component zeroed."""
    out = np.array(vec3, dtype=float, copy=True)
    out[UP_AXIS] = 0.0
    return out


def ensure_points_up(vec3: np.ndarray) -> np.ndarray:
    """Flip vec3 (e.g. a PCA trunk-axis eigenvector) if needed so its
    vertical component is >= 0 ('points up')."""
    vec3 = np.asarray(vec3, dtype=float)
    return -vec3 if vec3[UP_AXIS] < 0 else vec3
