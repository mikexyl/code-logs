"""Shared matplotlib style constants and geometry/plotting helpers."""

import numpy as np


# IEEE single-column style (3.5 in wide, 300 dpi, serif fonts, pdf/ps fonttype 42).
# Scripts that need a different figure size can override with:
#   plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.5)})
IEEE_RC = {
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.figsize': (3.5, 3.5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.8,
    'patch.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}

ROBOT_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']


def apply_alignment(positions, rotation, translation, scale):
    """Apply a Sim3 alignment: aligned = scale * R @ positions.T + translation.

    Args:
        positions   : (N,3) array
        rotation    : (3,3) rotation matrix
        translation : (3,)  translation vector
        scale       : float

    Returns:
        (N,3) aligned positions
    """
    return scale * (rotation @ positions.T).T + translation


def apply_frame_transform(positions, T):
    """Apply a 4x4 SE3 transform to an (N,3) array of positions.

    Returns (N,3) transformed positions.
    """
    if np.allclose(T, np.eye(4)):
        return positions
    ones = np.ones((positions.shape[0], 1))
    return (T @ np.hstack([positions, ones]).T).T[:, :3]


def find_tum_position(ts_s, timestamps, positions, max_gap_s=2.5):
    """Return the XYZ position nearest to ts_s from a TUM trajectory.

    Returns None if the nearest match is further than max_gap_s seconds away.
    """
    if len(timestamps) == 0:
        return None
    idx = int(np.argmin(np.abs(timestamps - ts_s)))
    if abs(timestamps[idx] - ts_s) > max_gap_s:
        return None
    return positions[idx]


def save_fig(fig, base_path, suffixes=('.pdf', '.png'), **kwargs):
    """Save *fig* to PDF and PNG (or other *suffixes*) at *base_path*.

    The extension of *base_path* is replaced by each suffix in turn.

    Args:
        fig       : matplotlib Figure
        base_path : destination path (extension is ignored / replaced)
        suffixes  : iterable of suffix strings
        **kwargs  : forwarded to fig.savefig(); bbox_inches and dpi have defaults
    """
    from pathlib import Path
    base_path = Path(base_path)
    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('dpi', 300)
    for suffix in suffixes:
        out = base_path.with_suffix(suffix)
        fig.savefig(out, **kwargs)
        print(f'Saved to {out}')


def mark_endpoint(ax, t_arr, v_arr, color, fmt='{:.0f}'):
    """Dot + value annotation at the last point of a curve."""
    ax.plot(t_arr[-1], v_arr[-1], 'o', color=color, markersize=3, zorder=5,
            clip_on=False)
    ax.annotate(fmt.format(v_arr[-1]),
                xy=(t_arr[-1], v_arr[-1]),
                xytext=(-4, 3), textcoords='offset points',
                fontsize=6, color=color, ha='right', va='bottom')
