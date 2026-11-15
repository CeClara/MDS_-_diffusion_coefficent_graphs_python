"""Figure 4: measured MSD curves, coloured by fitted diffusion category.
Place this Figure 4 script in MDS_-_diffusion_coefficent_graphs_python,
beside your existing time_vs_msd.py and XY_coord.csv.
Does not modify time_vs_msd.py.
Requires: pip install numpy pandas matplotlib

Note: XY_coord.csv has no frame column; positions are assumed chronological.
ID 2+23 is a provisional join requiring validation against frame-numbered data.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The folder containing time_vs_msd.py and XY_coord.csv
FOLDER = Path(__file__).resolve().parent
CSV_FILE = FOLDER / 'XY_coord.csv'
FRAME_INTERVAL = 0.20468      # seconds per video frame
EXPORT_UNITS_PER_PIXEL = 2336.8
PIXEL_SIZE_UM = None          # optional verified µm/pixel calibration; normalised curves unaffected
MAX_LAG_FRACTION = 0.25

# Selected experimental trajectories from Figure 3
VESICLES = ['9', '11', '13', '2+23', '33']

if not CSV_FILE.exists():
    raise FileNotFoundError(f'Put XY_coord.csv in the same folder as this script: {FOLDER}')

# Headerless export: track ID, X, Y, speed (same format as Figure 3).
data = pd.read_csv(CSV_FILE, header=None, names=['id', 'x', 'y', 'speed'])
for column in ['id', 'x', 'y']:
    data[column] = pd.to_numeric(data[column], errors='coerce')
data = data.dropna(subset=['id', 'x', 'y']).copy()
data['id'] = data['id'].astype(int)
data[['x', 'y']] = data[['x', 'y']] / EXPORT_UNITS_PER_PIXEL
if PIXEL_SIZE_UM is not None:
    data[['x', 'y']] *= PIXEL_SIZE_UM


def positions(name):
    # Return frame-numbered segments; do not bridge missing frames in ID 2+23.
    if name == '2+23':
        a = data.loc[data['id'] == 2, ['x', 'y']].to_numpy()
        b = data.loc[data['id'] == 23, ['x', 'y']].to_numpy()
        if len(a) < 12 or len(b) != 37:
            raise ValueError('Unexpected lengths for ID 2+23: inspect original tracks.')
        print('WARNING: ID 2+23 is a provisional join; verify frame identities.')
        return [(np.arange(1, 13), a[:12]), (np.arange(14, 51), b)]
    pts = data.loc[data['id'] == int(name), ['x', 'y']].to_numpy()
    if len(pts) < 8:
        raise ValueError(f'Insufficient positions for ID {name}')
    return [(np.arange(1, len(pts) + 1), pts)]


def calculate_msd(segments):
    longest = max(len(points) for frames, points in segments)
    max_lag = max(2, int(longest * MAX_LAG_FRACTION))
    times, msd = [], []
    for lag in range(1, max_lag + 1):
        squared = []
        for frames, points in segments:
            if len(points) <= lag:
                continue
            valid = (frames[lag:] - frames[:-lag]) == lag
            delta = points[lag:] - points[:-lag]
            squared.extend(np.sum(delta[valid] ** 2, axis=1))
        if squared:
            times.append(lag * FRAME_INTERVAL)
            msd.append(np.mean(squared))
    return np.asarray(times), np.asarray(msd)


# Category is determined by the exponent fitted to each measured curve.
# 0.9-1.1 is a descriptive, not a statistical, near-normal band.
colours = {
    'Subdiffusion': ['#0072B2'],        # Okabe-Ito blue
    'Approximately normal': ['#E69F00'], # Okabe-Ito orange
    'Superdiffusion': ['#D55E00']        # Okabe-Ito vermilion
}
counts = {name: 0 for name in colours}
groups = {name: [] for name in colours}

fig, ax = plt.subplots(figsize=(8, 5.2))

for name in VESICLES:
    time, msd = calculate_msd(positions(name))
    good = (time > 0) & (msd > 0)
    if good.sum() < 3:
        print(f'Skipping ID {name}: insufficient valid MSD measurements')
        continue
    alpha = np.polyfit(np.log(time[good]), np.log(msd[good]), 1)[0]
    if alpha < 0.9:
        category = 'Subdiffusion'
    elif alpha <= 1.1:
        category = 'Approximately normal'
    else:
        category = 'Superdiffusion'

    # Normalise only to compare curve shapes on one graph.
    # Values remain calculated from experimental MSD, not from idealised powers.
    relative_msd = msd / msd[0]
    palette = colours[category]
    colour = palette[counts[category] % len(palette)]
    counts[category] += 1
    groups[category].append((time, relative_msd, alpha))
    ax.plot(time, relative_msd, 'o-', markersize=3, linewidth=1.8,
            color=colour, label=f'ID {name} — {category} (α={alpha:.2f})')
    print(f'ID {name}: α={alpha:.3f}, {category}')

# Dashed black curves: category means of the experimental relative-MSD curves.
# Only show a group mean when 2+ trajectories exist in that group.
for category, curves in groups.items():
    if len(curves) < 2:
        continue
    common_time = min(curves, key=lambda item: len(item[0]))[0]
    mean_msd = np.mean([np.interp(common_time, t, y) for t, y, a in curves], axis=0)
    ax.plot(common_time, mean_msd, 'k--', linewidth=1.8,
            label=f'{category} mean (n={len(curves)})')

# Solid black reference: ideal α=1, anchored at relative MSD=1 at first lag.
reference_time = np.linspace(FRAME_INTERVAL, max(ax.get_xlim()[1], 2), 100)
ax.plot(reference_time, reference_time / FRAME_INTERVAL,
        color='black', linewidth=1.2, label='Normal reference (α=1)')

ax.set_title('Clustering analysis', loc='left', fontsize=12)
ax.set_xlabel('Lag time, Δt (s)', fontsize=11)
ax.set_ylabel('MSD / MSD at first lag (relative)', fontsize=11)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.tick_params(axis='both', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig(FOLDER / 'Figure_4_experimental_clustering.pdf', bbox_inches='tight')
plt.savefig(FOLDER / 'Figure_4_experimental_clustering.png', dpi=300,
            bbox_inches='tight')
plt.show()
