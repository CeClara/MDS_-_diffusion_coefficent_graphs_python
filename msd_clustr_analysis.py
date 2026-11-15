"""Figure 4: experimental MSD scaling. Put beside XY_coord.csv.

Classification is by log-log fitted exponent (alpha), not clustering algorithm.
Means are arithmetic MSD means, not averages of log(MSD).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FOLDER = Path(__file__).resolve().parent
CSV_FILE = FOLDER / 'XY_coord.csv'
FRAME_INTERVAL = 0.20468
EXPORT_UNITS_PER_PIXEL = 2336.8
UM_PER_PIXEL = 0.093
MIN_POSITIONS = 21
MAX_LAG_FRACTION = 0.25
MIN_MEAN_FRACTION = 0.75  # Stop before small remaining sample distorts mean

SELECTED = {
    'Subdiffusion': [5, 8, 9, 17],
    'Approximately normal': [3, 11, 14, 20],
    'Superdiffusion': [2, 13, 23],
}
COLOURS = {
    'Subdiffusion': ['#007F86', '#129C9C', '#33B1AF', '#66C6BF'],
    'Approximately normal': ['#AF6600', '#CE8200', '#E69F00', '#E8B04A'],
    'Superdiffusion': ['#7230A3', '#9141CA', '#AB5CE1', '#B881E5'],
}
DARK_MEANS = {'Subdiffusion': '#004E54', 'Superdiffusion': '#4A176C'}
MARKERS = {'Subdiffusion':'o', 'Approximately normal':'s', 'Superdiffusion':'^'}

# Read headerless XY_coord.csv. The order within each ID is assumed chronological.
data = pd.read_csv(CSV_FILE, header=None, names=['ID', 'x', 'y', 'speed'])
for col in ['ID', 'x', 'y']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna(subset=['ID','x','y']).copy()
data['ID'] = data['ID'].astype(int)

groups = {name: [] for name in COLOURS}
for track_id, track in data.groupby('ID', sort=True):
    xy = track[['x','y']].to_numpy() / EXPORT_UNITS_PER_PIXEL * UM_PER_PIXEL
    if len(xy) < MIN_POSITIONS:
        continue
    n_lags = int(len(xy) * MAX_LAG_FRACTION)
    time = np.arange(1,n_lags+1) * FRAME_INTERVAL
    msd = np.array([np.mean(np.sum((xy[k:]-xy[:-k])**2, axis=1))
                    for k in range(1,n_lags+1)])
    valid = (msd>0) & np.isfinite(msd)
    if valid.sum()<5:
        continue
    log_t, log_y = np.log(time[valid]), np.log(msd[valid])
    alpha, intercept = np.polyfit(log_t, log_y, 1)
    ss_total = np.sum((log_y-log_y.mean())**2)
    r2 = 1-np.sum((log_y-(alpha*log_t+intercept))**2)/ss_total if ss_total>0 else float('nan')
    category = ('Subdiffusion' if alpha<0.9 else
                'Approximately normal' if alpha<=1.1 else 'Superdiffusion')
    groups[category].append((track_id,alpha,r2,time,msd))

def mean_curve(category):
    """Arithmetic mean at each lag from all available members, with >=75% retained."""
    records = groups[category]
    required = max(2, int(np.ceil(MIN_MEAN_FRACTION*len(records))))
    x, y, counts = [], [], []
    for lag_idx in range(max(len(r[4]) for r in records)):
        values = [r[4][lag_idx] for r in records if len(r[4])>lag_idx]
        if len(values)<required:
            break
        x.append((lag_idx+1)*FRAME_INTERVAL)
        y.append(float(np.mean(values)))
        counts.append(len(values))
    return np.array(x), np.array(y), counts

print('Figure 4 — Cluster MSD scaling')
for category, records in groups.items():
    selected = [r for r in records if r[0] in SELECTED[category]]
    print(f'\n{category}: {len(records)} classified, {len(selected)} plotted')
    for track_id, alpha, r2, _, _ in selected:
        print(f'  ID {track_id}: alpha={alpha:.3f}, R²={r2:.3f}')
for category in ('Subdiffusion','Superdiffusion'):
    _,_,counts = mean_curve(category)
    print(f'{category} arithmetic mean: n ranges from {max(counts)} to {min(counts)} over plotted lags')

plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(9.2,5.5))
for category, records in groups.items():
    shown = [r for r in records if r[0] in SELECTED[category]]
    for i, (_,_,_,time,msd) in enumerate(shown):
        ax.plot(time,msd,linewidth=1.5,alpha=0.80,
                color=COLOURS[category][i%len(COLOURS[category])],
                marker=MARKERS[category],markersize=3.5,markevery=2,zorder=2)
for category in ('Subdiffusion','Superdiffusion'):
    t, avg, _ = mean_curve(category)
    ax.plot(t,avg,'--',lw=3.4,color=DARK_MEANS[category],zorder=4)

# Reference alpha=1: fixed intercept anchored to measured first-lag normal-group mean.
normal_anchor = np.mean([r[4][0] for r in groups['Approximately normal']])
ref_t = np.geomspace(FRAME_INTERVAL,2.5,200)
ax.plot(ref_t,normal_anchor*(ref_t/FRAME_INTERVAL),color='#191919',lw=3.0,zorder=3)
ax.set(xscale='log',yscale='log',xlim=(FRAME_INTERVAL*.95,2.6))
ax.set_title('Cluster MSD scaling',loc='left',fontsize=15,pad=12)
ax.set_xlabel('Lag time, Δt (s)',fontsize=12)
ax.set_ylabel('MSD (µm²)',fontsize=12)
ax.tick_params(axis='both',labelsize=10)
ax.spines[['top','right']].set_visible(False)

# Matching each category with its mean, instead of separating legend columns by style.
handles = [
    Line2D([0],[0],color=COLOURS['Subdiffusion'][0],lw=1.7,marker='o',markersize=5,label='Subdiffusion · example tracks'),
    Line2D([0],[0],color=DARK_MEANS['Subdiffusion'],lw=3.4,ls='--',label=f'Subdiffusion · mean (n={len(groups["Subdiffusion"])})'),
    Line2D([0],[0],color=COLOURS['Approximately normal'][1],lw=1.7,marker='s',markersize=5,label='Near-normal · example tracks'),
    Line2D([0],[0],color='#191919',lw=3.0,label='Brownian reference · α = 1'),
    Line2D([0],[0],color=COLOURS['Superdiffusion'][1],lw=1.7,marker='^',markersize=5,label='Superdiffusion · example tracks'),
    Line2D([0],[0],color=DARK_MEANS['Superdiffusion'],lw=3.4,ls='--',label=f'Superdiffusion · mean (n={len(groups["Superdiffusion"])})'),
]
# Matplotlib legend order fills columns first: entries are paired by rows through careful reorder.
ax.legend(handles=handles,ncol=2,frameon=False,loc='upper left',fontsize=8.5,
          columnspacing=2.1,handlelength=3.0)
fig.tight_layout()
for ext in ('svg','pdf','png'):
    fig.savefig(FOLDER / f'Figure_4_Cluster_MSD_scaling_final.{ext}',
                dpi=300,bbox_inches='tight')
plt.show()
