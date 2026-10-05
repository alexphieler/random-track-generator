from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from random_track_generator import generate_track

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *, desc="", unit="", **_):
        """Provide basic progress feedback when tqdm is not installed."""
        total = len(iterable)
        for current, item in enumerate(iterable, start=1):
            print(f"\r{desc}: {current}/{total} {unit}s", end="", flush=True)
            yield item
        print()

# Use preset parameters
# track = generate_track("small")
# track = generate_track("medium")
# track = generate_track("large", seed=42)

# # Or set parameters manually
# track = generate_track(
#     n_points=60,       # Voronoi points
#     n_regions=20,      # Regions to select
#     min_bound=0.,      # Minimum x/y bound [m]
#     max_bound=150.,    # Maximum x/y bound [m]
#     mode="extend",     # Generation mode
#     seed=42            # Optional: for reproducibility
# )

# cones_left, cones_right = track.as_tuple()


# Generate a collection of reproducible tracks and save each as a PNG.
PLOT_COUNT = 1700
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
PACSIM_TRACKS_DIR = Path("pacsim_tracks")
PACSIM_TRACKS_DIR.mkdir(exist_ok=True)
TRACK_PROFILES = (
    {"name": "complex", "preset": "large", "n_regions": 20, "mode": "random"},
    {"name": "irregular", "preset": "medium", "n_regions": 12, "mode": "random"},
    {"name": "rounded", "preset": "medium", "n_regions": 4, "mode": "expand"},
    {"name": "oval", "preset": "large", "n_regions": 4, "mode": "extend"},
    {"name": "hairpin", "preset": "large", "n_regions": 3, "mode": "hairpin"},
    {"name": "slalom", "preset": "large", "n_regions": 3, "mode": "slalom"},
    {"name": "technical", "preset": "medium", "n_regions": 8, "mode": "technical"},
)
TRACK_WIDTH_MEAN = 3.5
TRACK_WIDTH_RANGE = (3.0, 5.5)
TRACK_WIDTH_SHAPE = 4
MIN_OUTER_TURNING_DIAMETER_RANGE = (8.7, 9.0)
CONE_SPACING_MEAN_RANGE = (3.5, 4.5)
CONE_SPACING_STDDEV_RANGE = (0.05, 0.20)
CONE_SPACING_MIN_RANGE = (1.5, 2.5)
CONE_SPACING_MAX_RANGE = (4.5, 5.2)
CONE_SPACING_OFFSET_RANGE = (0.0, 0.5)
INNER_CONE_CURVATURE_FACTOR_MEAN = 0.7
INNER_CONE_CURVATURE_FACTOR_STDDEV = 0.2
HAIRPIN_MIN_INNER_CONE_GAP = 3.0


def sample_track_width(rng, mode, minimum, maximum):
    """Sample widths near mode while making both extremes equally likely."""
    direction = rng.choice((-1, 1))
    limit = minimum if direction < 0 else maximum
    distance = rng.beta(1, TRACK_WIDTH_SHAPE) * abs(limit - mode)
    return mode + direction * distance


def sample_truncated_normal(rng, mean, stddev, minimum, maximum):
    """Sample a normal distribution constrained to an inclusive range."""
    while True:
        value = rng.normal(mean, stddev)
        if minimum <= value <= maximum:
            return value


for index in tqdm(range(PLOT_COUNT), desc="Generating track plots", unit="track"):
    profile = TRACK_PROFILES[index % len(TRACK_PROFILES)]
    rng = np.random.default_rng(index)
    track_width = sample_track_width(rng, TRACK_WIDTH_MEAN, *TRACK_WIDTH_RANGE)
    cone_spacing_mean = rng.uniform(*CONE_SPACING_MEAN_RANGE)
    cone_spacing_stddev = rng.uniform(*CONE_SPACING_STDDEV_RANGE)
    cone_spacing_min = rng.uniform(*CONE_SPACING_MIN_RANGE)
    cone_spacing_max = rng.uniform(*CONE_SPACING_MAX_RANGE)
    cone_spacing_offset = rng.uniform(*CONE_SPACING_OFFSET_RANGE)
    inner_cone_curvature_factor = sample_truncated_normal(
        rng,
        INNER_CONE_CURVATURE_FACTOR_MEAN,
        INNER_CONE_CURVATURE_FACTOR_STDDEV,
        0.0,
        1.0,
    )
    min_outer_turning_diameter = rng.uniform(*MIN_OUTER_TURNING_DIAMETER_RANGE)
    track = generate_track(
        profile["preset"],
        seed=index,
        n_regions=profile["n_regions"],
        mode=profile["mode"],
        track_width=track_width,
        cone_spacing_mean=cone_spacing_mean,
        cone_spacing_stddev=cone_spacing_stddev,
        cone_spacing_min=cone_spacing_min,
        cone_spacing_max=cone_spacing_max,
        cone_spacing_offset=cone_spacing_offset,
        inner_cone_curvature_factor=inner_cone_curvature_factor,
        min_outer_turning_diameter=min_outer_turning_diameter,
        hairpin_min_inner_cone_gap=HAIRPIN_MIN_INNER_CONE_GAP,
    )
    cones_left, cones_right = track.as_tuple()
    track.save(
        PACSIM_TRACKS_DIR,
        sim_type="pacsim",
        filename=f"track_{index:03d}.yaml",
    )
    time_keeping = track.pacsim_time_keeping()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(cones_left[:, 0], cones_left[:, 1], color="tab:blue", label="left cones")
    ax.scatter(cones_right[:, 0], cones_right[:, 1], color="#f1c40f", label="right cones")
    if time_keeping:
        time_keeping_positions = np.asarray(
            [point["position"][:2] for point in time_keeping]
        )
        ax.scatter(
            time_keeping_positions[:, 0],
            time_keeping_positions[:, 1],
            color="tab:orange",
            marker="x",
            s=100,
            linewidths=2,
            label="time keeping",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Generated {profile['name']} {profile['preset']} track (seed={index})\n"
        f"width={track_width:.2f} m · cones={cone_spacing_mean:.2f}±{cone_spacing_stddev:.2f} m · "
        f"offset={cone_spacing_offset:.2f} · curvature={inner_cone_curvature_factor:.2f} · "
        f"turn diameter={min_outer_turning_diameter:.2f} m"
        + (f" · hairpin gap≥{HAIRPIN_MIN_INNER_CONE_GAP:.1f} m" if profile["mode"] == "hairpin" else ""),
        fontsize=10,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"track_{index:03d}.png", dpi=150)
    plt.close(fig)
