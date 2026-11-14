# -*- coding: utf-8 -*-
"""
TrackMateCSV.py   ( was BeadDiffusion.py )
Analysis of Brownian motion from ImageJ TrackMate CSV data

Calculates:
    - Original bead trajectories
    - Global stage drift
    - Drift-corrected trajectories
    - Original and corrected MSD
    - Diffusion coefficients
"""
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS
# ============================================================

# Plotting
plotRawTracks = True
plotCorrectedTracks = True
plotMSD = True

# MSD analysis (expressed in frames)
maxLagFrames = 20        # Maximum time interval (frames)
fitMaxLagFrames = 5      # Linear fitting interval (frames)


# ============================================================
# READ TRACKMATE DATA
# ============================================================

def read_trackmate_csv(filepath):

    # Read only the four columns we need
    data = pd.read_csv(filepath, skiprows=4, header=None,
                       usecols=[2, 4, 5, 7])

    data.columns = ['ID', 'X', 'Y', 'Time']

    # Convert everything to numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Remove incomplete observations
    data = data.dropna(subset=['ID', 'X', 'Y', 'Time'])

    # Sort by bead ID and time
    data = data.sort_values(['ID', 'Time']).reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid observations found in CSV file")

    return data


# ============================================================
# DETERMINE FRAME INTERVAL
# ============================================================

def get_frame_interval(data):

    # Calculate successive time intervals for each bead
    dt = data.groupby('ID')['Time'].diff()

    # Keep positive intervals
    dt = dt[dt > 0]

    if len(dt) == 0:
        raise ValueError("Unable to determine frame interval")

    # Median observed interval
    frameInterval_s = dt.median()

    frameRate_Hz = 1 / frameInterval_s

    print("\nACQUISITION TIMING")
    print("------------------------------------")
    print(f"Frame interval: {frameInterval_s:.4f} s")
    print(f"Frame rate:     {frameRate_Hz:.4f} Hz")

    return frameInterval_s

# ============================================================
# PLOT TRAJECTORIES
# ============================================================

def plot_trajectories(data, x_col='X', y_col='Y',
                      title='Bead Tracks', xlim=None, ylim=None):

    fig, ax = plt.subplots(figsize=(8, 8))

    for bead_id, track in data.groupby('ID'):

        ax.plot(track[x_col], track[y_col],
                linewidth=1.5, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('X Position (um)')
    ax.set_ylabel('Y Position (um)')

    # Use supplied limits, or calculate from the data
    if xlim is None:
        xlim = (data[x_col].min(), data[x_col].max())

    if ylim is None:
        ylim = (data[y_col].min(), data[y_col].max())

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

# ============================================================
# GLOBAL DRIFT CORRECTION
# ============================================================

def remove_drift(data):

    # Calculate successive displacements
    dx = data.groupby('ID')['X'].diff()
    dy = data.groupby('ID')['Y'].diff()
    dt = data.groupby('ID')['Time'].diff()

    # Individual velocity components (um/s)
    vx = dx / dt
    vy = dy / dt

    # Keep valid observations
    valid = (dt > 0) & np.isfinite(vx) & np.isfinite(vy)

    if not valid.any():
        raise ValueError("No valid displacements available for drift calculation")

    # Global median velocity vector
    driftX = np.median(vx[valid])
    driftY = np.median(vy[valid])

    # Convert to nm/s
    driftX_nm = driftX * 1000
    driftY_nm = driftY * 1000

    driftSpeed_nm = np.hypot(driftX_nm, driftY_nm)
    driftAngle = np.degrees(np.arctan2(driftY_nm, driftX_nm))

    # Report global drift
    print("\nGLOBAL DRIFT VECTOR")
    print("------------------------------------")
    print(f"Drift X:       {driftX_nm:8.2f} nm/s")
    print(f"Drift Y:       {driftY_nm:8.2f} nm/s")
    print(f"Magnitude:     {driftSpeed_nm:8.2f} nm/s")
    print(f"Direction:     {driftAngle:8.2f} degrees")

    # Subtract the same constant velocity from every trajectory
    corrected = data.copy()

    elapsed = corrected['Time'] - corrected['Time'].min()

    corrected['X_corr'] = corrected['X'] - driftX * elapsed
    corrected['Y_corr'] = corrected['Y'] - driftY * elapsed

    return corrected


# ============================================================
# CALCULATE MEAN SQUARED DISPLACEMENT
# ============================================================

def calculate_msd(data, x_col, y_col, frameInterval_s):

    lagFrames = []
    lagTimes = []
    squaredDisplacements = []

    # Analyse each bead independently
    for bead_id, track in data.groupby('ID'):

        track = track.sort_values('Time')

        x = track[x_col].to_numpy()
        y = track[y_col].to_numpy()
        t = track['Time'].to_numpy()

        n = len(track)

        # Calculate all pairwise displacements
        for i in range(n - 1):

            # Only consider intervals within maxLagFrames
            end = np.searchsorted(
                t, t[i] + maxLagFrames * frameInterval_s,
                side='right'
            )

            if end <= i + 1:
                continue

            dx = x[i+1:end] - x[i]
            dy = y[i+1:end] - y[i]

            dt = t[i+1:end] - t[i]

            # Convert time differences to nearest frame interval
            frames = np.rint(dt / frameInterval_s).astype(int)

            # Keep valid frame intervals
            valid = (frames > 0) & (frames <= maxLagFrames)

            r2 = dx**2 + dy**2

            lagFrames.extend(frames[valid])
            lagTimes.extend(dt[valid])
            squaredDisplacements.extend(r2[valid])

    if len(lagFrames) == 0:
        raise ValueError(
            "No displacement pairs found. Check track lengths and maxLagFrames."
        )

    # Assemble results
    results = pd.DataFrame({
        'Frames': lagFrames,
        'dT': lagTimes,
        'MSD': squaredDisplacements
    })

    # Average squared displacement at each frame interval
    msd = results.groupby('Frames').agg(
        dT=('dT', 'mean'),
        MSD=('MSD', 'mean'),
        N=('MSD', 'count')
    ).reset_index()

    return msd


# ============================================================
# CALCULATE DIFFUSION COEFFICIENT
# ============================================================

def calculate_diffusion(msd):

    # Select initial linear region
    fit = msd[msd['Frames'] <= fitMaxLagFrames]

    if len(fit) < 2:
        raise ValueError("Not enough MSD points for linear fit")

    # Linear regression: MSD = intercept + slope * dT
    slope, intercept = np.polyfit(fit['dT'], fit['MSD'], 1)

    # For 2D Brownian motion: MSD = 4 * D * dT
    D = slope / 4

    return D, slope, intercept, len(fit)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def get_filepath():
    # Use a command-line argument if given: python BeadDiffusion.py mytracks.csv
    if len(sys.argv) > 1:
        return sys.argv[1]

    # Otherwise show a native file-open dialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select TrackMate spots CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()

    if not path:
        raise SystemExit("No file selected.")
    return path

filepath = get_filepath()
print(f"\nAnalysing: {filepath}")

tracks = read_trackmate_csv(filepath)

print("\nCSV SUCCESSFULLY READ")
print("------------------------------------")
print("Number of observations:", len(tracks))
print("Number of bead tracks:", tracks['ID'].nunique())

print("\nFirst 10 observations:")
print(tracks.head(10))


# ============================================================
# ACQUISITION TIMING
# ============================================================

frameInterval_s = get_frame_interval(tracks)


# ============================================================
# DRIFT CORRECTION
# ============================================================

correctedTracks = remove_drift(tracks)


# ============================================================
# PLOT ORIGINAL AND CORRECTED TRAJECTORIES
# ============================================================

if plotRawTracks:
    plot_trajectories(tracks, title='Original Tracks')

if plotCorrectedTracks:
    plot_trajectories(correctedTracks,
                      x_col='X_corr', y_col='Y_corr',
                      title='Drift Corrected Tracks')


# ============================================================
# MSD ANALYSIS
# ============================================================

msdRaw = calculate_msd(tracks, 'X', 'Y', frameInterval_s)

msdCorrected = calculate_msd(correctedTracks,
                             'X_corr', 'Y_corr', frameInterval_s)


# ============================================================
# DIFFUSION COEFFICIENTS
# ============================================================

DRaw, slopeRaw, interceptRaw, nRaw = calculate_diffusion(msdRaw)

DCorrected, slopeCorrected, interceptCorrected, nCorrected = calculate_diffusion(msdCorrected)

# Actual fitting interval
fitTime_s = fitMaxLagFrames * frameInterval_s

print("\nDIFFUSION ANALYSIS")
print("------------------------------------")
print(f"Fit interval:           {fitMaxLagFrames} frames ({fitTime_s:.3f} s)")
print(f"Original fit points:    {nRaw}")
print(f"Corrected fit points:   {nCorrected}")

print(f"\nOriginal D:             {DRaw:.6f} um^2/s")
print(f"Corrected D:            {DCorrected:.6f} um^2/s")


# ============================================================
# PRINT MSD RESULTS
# ============================================================

print("\nMSD RESULTS (ORIGINAL)")
print("------------------------------------")
print(msdRaw.to_string(index=False))

print("\nMSD RESULTS (DRIFT CORRECTED)")
print("------------------------------------")
print(msdCorrected.to_string(index=False))


# ============================================================
# PLOT MSD AGAINST TIME INTERVAL
# ============================================================

if plotMSD:

    fig, ax = plt.subplots(figsize=(9, 6))

    # Original MSD
    ax.plot(msdRaw['dT'], msdRaw['MSD'],
            'o-', markersize=4, label='Original')

    # Corrected MSD
    ax.plot(msdCorrected['dT'], msdCorrected['MSD'],
            'o-', markersize=4, label='Drift corrected')

    # Linear fits
    fitTime = np.linspace(0, fitTime_s, 100)

    ax.plot(fitTime, interceptRaw + slopeRaw * fitTime,
            '--', label=f'Original: D = {DRaw:.4f}')

    ax.plot(fitTime, interceptCorrected + slopeCorrected * fitTime,
            '--', label=f'Corrected: D = {DCorrected:.4f}')

    ax.set_xlabel('Time interval, dT (s)')
    ax.set_ylabel('MSD (um$^2$)')
    ax.set_title('Mean Squared Displacement')

    ax.legend()
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


    