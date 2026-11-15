"""Figure 3 standalone, publication-quality vector PDF MSD figures.

Place XY_coord.csv in the same folder as this Python script. Uses the same
calculation and provisional segment assignment as figure3_msd.py.
No SEM/error bars. PDFs are vector artwork, so lines/text scale cleanly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_FOLDER = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_FOLDER / 'XY_coord.csv'
FRAME_INTERVAL = 0.20468
EXPORT_UNITS_PER_PIXEL = 2336.8  # provisional CSV-to-pixel factor; verify against FIJI/TrackMate
PIXEL_SIZE_UM = 0.093            # 100x objective, 1x1 binning (µm/pixel)
MAX_LAG_FRACTION = 0.25
OUTPUT_FOLDER = SCRIPT_FOLDER  # Save PDFs and PNGs beside this .py file

# Export crisp, editable vector text and paths into PDF
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.transparent': False,
                     'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
                     'figure.facecolor': 'white'})


def load_positions(csv_path=CSV_FILE):

    if PIXEL_SIZE_UM is None or PIXEL_SIZE_UM <= 0:
        raise ValueError('Set PIXEL_SIZE_UM to the verified µm/pixel calibration.')
    if not csv_path.exists():
        raise FileNotFoundError(
            f'Cannot find {csv_path.name}. Put XY_coord.csv in the same folder as this .py file.'
        )

    # The original XY_coord.csv has no header: track ID, X, Y, speed.
    df = pd.read_csv(csv_path, header=None,
                     names=['track_id', 'x_export', 'y_export', 'speed_export'])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['track_id', 'x_export', 'y_export']).copy()
    df['track_id'] = df['track_id'].astype(int)
    df['x'] = df['x_export'] / EXPORT_UNITS_PER_PIXEL * PIXEL_SIZE_UM
    df['y'] = df['y_export'] / EXPORT_UNITS_PER_PIXEL * PIXEL_SIZE_UM
    return df


def trajectory(data, track_id, first_n=None):
    """Each result is a list of (frame numbers, XY points) segments."""
    if track_id == '2+23':
        a = data.loc[data.track_id == 2, ['x', 'y']].to_numpy()
        b = data.loc[data.track_id == 23, ['x', 'y']].to_numpy()
        if len(a) < 12 or len(b) != 37:
            raise ValueError('Unexpected track lengths. Check the TrackMate frame-numbered export.')
        # Frame 13 missing; never use the incorrect later ID 2 detections
        return [(np.arange(1, 13), a[:12]), (np.arange(14, 51), b)]
    points = data.loc[data.track_id == int(track_id), ['x', 'y']].to_numpy()
    if first_n is not None:
        points = points[:first_n]
    if len(points) < 6:
        raise ValueError(f'Track ID {track_id} has too few positions: {len(points)}')
    return [(np.arange(1, len(points) + 1), points)]


def calculate_msd(segments, dt=FRAME_INTERVAL):
    """Time-averaged MSD; only within-segment pairs, never across missing frame."""
    longest = max(len(points) for _, points in segments)
    max_lag = max(2, int(longest * MAX_LAG_FRACTION))
    times, means = [], []
    for lag in range(1, max_lag + 1):
        squared = []
        for frames, points in segments:
            if len(points) <= lag:
                continue
            valid = (frames[lag:] - frames[:-lag]) == lag
            delta = points[lag:] - points[:-lag]
            squared.extend(np.sum(delta[valid] ** 2, axis=1))
        if squared:
            times.append(lag * dt)
            means.append(np.mean(squared))
    return np.array(times), np.array(means)


def plot_msd(ax, data, track_id, title, colour, first_n=None):
    segments = trajectory(data, track_id, first_n=first_n)
    time, msd = calculate_msd(segments)
    good = np.isfinite(msd) & (time > 0) & (msd > 0)
    if good.sum() < 2:
        raise ValueError(f'Not enough valid MSD values for ID {track_id}')
    alpha, ln_amp = np.polyfit(np.log(time[good]), np.log(msd[good]), 1)
    fitted = np.exp(ln_amp) * time ** alpha
    reference = fitted[0] * time / time[0]

    ax.plot(time, msd, 'o-', color=colour, lw=2.1, ms=5, label='Measured MSD')
    ax.plot(time, fitted, '--', color='black', lw=1.5,
            label=f'Scaling exponent = {alpha:.2f}')
    ax.plot(time, reference, ':', color='gray', lw=1.5,
            label='Brownian diffusion (scaling exponent = 1)')
    # Like the user's earlier GFP plots: bold panel letter and smaller track name
    ax.text(-0.10, 1.04, title[0], transform=ax.transAxes,
            fontsize=22, fontweight='bold')
    ax.set_title(title[4:], loc='left', fontsize=12, pad=12)
    ax.set_xlabel('Time, Δt (s)', fontsize=12)
    ax.set_ylabel('MSD (µm²)', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    print(f'{title}: alpha = {alpha:.3f}, {sum(len(p) for _, p in segments)} spots')


def save_plot(data, panel, track_id, description, colour, suffix=None, first_n=None):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    label = f'{panel}   ID {track_id} — {description}'
    plot_msd(ax, data, track_id, label, colour, first_n=first_n)
    fig.tight_layout()
    stem = f'Figure3_{panel}_ID_{str(track_id).replace("+", "_")}'
    if suffix:
        stem += f'_{suffix}'
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_FOLDER / f'{stem}.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')  # vector resolution-independent
    # Optional large raster backup for software that cannot import PDFs properly
    fig.savefig(OUTPUT_FOLDER / f'{stem}.png', format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    return pdf_path


if __name__ == '__main__':
    data = load_positions()
    save_plot(data, 'E', 33, 'Multiple diffusion catagories', "#053347", suffix='full')
    save_plot(data, 'A', 9 ,'Subdiffusion', '#A32997')
    save_plot(data, 'B', 11, 'Approximately normal', "#A32929")
    save_plot(data, 'C', 13, 'Superdiffusion', '#35C76B')
    save_plot(data, 'D', '2+23', 'Super diffusion', '#F07832')
    print('PDF and 600 dpi PNG outputs saved in', OUTPUT_FOLDER)
