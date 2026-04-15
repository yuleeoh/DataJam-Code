from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('absenteeism_by_county_year.csv')
county_year = df.groupby(['county', 'year'])['rate'].mean().reset_index()
pivot = county_year.pivot(index='county', columns='year', values='rate').reset_index()
pivot.columns.name = None
pivot = pivot.rename(columns={
    '2018-19': 'pre_covid',
    '2021-22': 'early_post',
    '2023-24': 'mid_post',
    '2024-25': 'late_post'
}).dropna()

# --- K-Means Clustering ---
traj   = pivot[['pre_covid','early_post','mid_post','late_post']].values
scaled = StandardScaler().fit_transform(traj)
km     = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaled)
pivot['cluster'] = km.labels_

order = pivot.groupby('cluster')['late_post'].mean().sort_values().index
labels_map = {
    order[0]: 'Best Recovery',
    order[1]: 'Moderate Recovery',
    order[2]: 'Stalled Recovery',
    order[3]: 'Persistently High',
}
pivot['cluster_label'] = pivot['cluster'].map(labels_map)

# --- Styling ---
BG     = '#2b2d3e'
CARD   = '#3a3c50'
HEADER = '#b8c9d9'
WHITE  = '#f0f4f8'
MUTED  = '#8a9bb0'

CLUSTER_COLORS = {
    'Best Recovery':     '#6dbf8a',
    'Moderate Recovery': '#7eb8d4',
    'Stalled Recovery':  '#d4b96d',
    'Persistently High': '#d47e7e',
}

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    CARD,
    'axes.edgecolor':    MUTED,
    'axes.labelcolor':   WHITE,
    'text.color':        WHITE,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'grid.color':        '#4a4c60',
    'grid.linewidth':    0.5,
    'font.family':       'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.titlecolor':   WHITE,
})

time_x      = [0, 1, 2, 3]
time_labels = ['2018–19\n(Pre-COVID)', '2021–22\n(Early Post)',
               '2023–24\n(Mid Post)', '2024–25\n(Late Post)']

fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs  = gridspec.GridSpec(1, 2, figure=fig, hspace=0.45, wspace=0.35,
                        left=0.07, right=0.97, top=0.88, bottom=0.12)

fig.text(0.5, 0.975, 'EVALUATING THE IMPACT OF COVID-19 ON CALIFORNIA SCHOOL ATTENDANCE RATES',
         ha='center', va='top', fontsize=13, fontweight='bold', color=WHITE)
fig.text(0.5, 0.955, 'Coding Extension: K-Means Trajectory Clustering + Recovery Heatmap  ·  Team 3: The Outliers',
         ha='center', va='top', fontsize=9.5, color=MUTED)
fig.patches.append(plt.Rectangle((0.04, 0.935), 0.92, 0.002,
                                  transform=fig.transFigure, color=HEADER, zorder=10))

# --- Plot 1: K-Means Clustering ---
ax1 = fig.add_subplot(gs[0, 0])

for label, color in CLUSTER_COLORS.items():
    sub = pivot[pivot['cluster_label'] == label]
    if len(sub) == 0: continue
    means = [sub[c].mean() for c in ['pre_covid','early_post','mid_post','late_post']]
    stds  = [sub[c].std()  for c in ['pre_covid','early_post','mid_post','late_post']]
    ax1.plot(time_x, means, color=color, linewidth=2.8, marker='o', markersize=7,
             label=f'{label} (n={len(sub)})', zorder=5)
    lo = [m - s for m, s in zip(means, stds)]
    hi = [m + s for m, s in zip(means, stds)]
    ax1.fill_between(time_x, lo, hi, color=color, alpha=0.10)

ax1.set_xticks(time_x); ax1.set_xticklabels(time_labels, fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax1.set_ylabel('Chronic Absenteeism Rate', fontsize=10)
ax1.set_title('K-MEANS TRAJECTORY CLUSTERING: 4 Recovery Archetypes  (shaded = ±1 SD)',
              fontsize=11, fontweight='bold', pad=10)
ax1.legend(fontsize=9, framealpha=0.2); ax1.grid(True, axis='y', alpha=0.4)

# --- Plot 2: Recovery Heatmap ---
ax2 = fig.add_subplot(gs[0, 1])

norm_data = []
for _, row in pivot.iterrows():
    base  = row['pre_covid']
    shock = max(row['early_post'] - base, 0.001)
    norm  = [(row[c] - base) / shock for c in ['pre_covid','early_post','mid_post','late_post']]
    norm_data.append(norm)

norm_arr      = np.array(norm_data)
sort_idx      = np.argsort(norm_arr[:, 3])[::-1]
norm_sorted   = norm_arr[sort_idx]
county_sorted = pivot.iloc[sort_idx]['county'].values

cmap = LinearSegmentedColormap.from_list('recovery', ['#6dbf8a', '#d4b96d', '#d47e7e'], N=256)
im = ax2.imshow(norm_sorted.T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(time_labels, fontsize=7)
ax2.set_xticks(range(len(county_sorted)))
ax2.set_xticklabels(county_sorted, rotation=70, ha='right', fontsize=5)
ax2.set_title('RECOVERY HEATMAP\n(green=baseline, red=peak shock)\nsorted by 2024–25 residual',
              fontsize=10, fontweight='bold', pad=10)
cb = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cb.set_label('Normalized Level', fontsize=7, color=MUTED)
cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=MUTED)

plt.savefig('datajam_extension.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# --- Cluster Summary ---
print("=" * 55)
print("CLUSTER SUMMARY")
print("=" * 55)
print(f"\nCluster Summary (sorted by 2024-25 rate):")
for label in ['Best Recovery','Moderate Recovery','Stalled Recovery','Persistently High']:
    sub = pivot[pivot['cluster_label'] == label]
    if len(sub) == 0: continue
    print(f"  {label} (n={len(sub)}): pre={sub['pre_covid'].mean():.3f}, "
          f"late={sub['late_post'].mean():.3f}")
