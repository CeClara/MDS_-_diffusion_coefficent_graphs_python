
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

    return {
        'tracks': tracks,
        'corrected_tracks': correctedTracks,
        'msd_raw': msdRaw,
        'msd_corrected': msdCorrected,
        'D_raw': DRaw,
        'D_corrected': DCorrected,
    }


if __name__ == "__main__":
    # Change this filename to the CSV file you want to analyse.
    analyse("mytracks.csv")
