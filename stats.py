# code to run the matched pairs t-test with summary statistics and box plot

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

data = {
    'ID': list(range(1, 59)),
    'Post': [0.166,0.241,0.21,0.198,0.287,0.159,0.183,0.239,0.12,0.203,0.107,0.235,0.229,0.24,
             0.171,0.134,0.346,0.207,0.185,0.197,0.102,0.291,0.28,0.185,0.146,0.229,0.16,0.183,
             0.181,0.123,0.109,0.268,0.21,0.21,0.179,0.224,0.167,0.219,0.224,0.16,0.133,0.15,
             0.134,0.188,0.16,0.159,0.245,0.225,0.174,0.165,0.139,0.141,0.171,0.15,0.199,0.159,
             0.182,0.159],
    'Pre':  [0.108,0.157,0.124,0.143,0.152,0.087,0.099,0.156,0.09,0.113,0.1,0.134,0.119,0.13,
             0.11,0.08,0.192,0.146,0.107,0.106,0.07,0.175,0.181,0.104,0.197,0.113,0.1,0.102,
             0.127,0.065,0.058,0.198,0.109,0.127,0.107,0.118,0.096,0.117,0.13,0.091,0.075,0.081,
             0.073,0.109,0.118,0.069,0.156,0.129,0.101,0.092,0.102,0.111,0.143,0.08,0.13,0.071,
             0.102,0.096]
}
df = pd.DataFrame(data)
df['d'] = (df['Post'] - df['Pre']).round(3)

rng_ids = [29,20,26,56,14,58,55,44,10,5,39,47,46,11,15,1,19,2,9,52,3,7,48,28,41,25,30,13,4,43]
sample = df[df['ID'].isin(rng_ids)].copy().reset_index(drop=True)

d = sample['d'].values
n = len(d)
d_bar = np.mean(d)
s_d = np.std(d, ddof=1)
se = s_d / np.sqrt(n)
t_stat = d_bar / se
df_t = n - 1
p_value = 2 * stats.t.sf(abs(t_stat), df=df_t)
q1 = np.percentile(d, 25)
med = np.median(d)
q3 = np.percentile(d, 75)

BG     = '#2c2d3e'
PANEL  = '#232432'
BLUE   = '#68a0c2'
LIGHT  = '#c8d9e6'
MUTED  = '#8a9bb5'
MEAN_C = '#e07b6a'
GRID_C = '#3a3b52'

plt.rcParams.update({
    'text.color': LIGHT,
    'axes.labelcolor': LIGHT,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'axes.edgecolor': GRID_C,
})

fig = plt.figure(figsize=(13, 15), facecolor=BG)
gs = GridSpec(3, 1, figure=fig, height_ratios=[1.2, 1, 1.1], hspace=0.52)

fig.text(0.5, 0.975, 'EVALUATING THE IMPACT OF COVID-19 ON CALIFORNIA SCHOOLS',
         ha='center', va='top', fontsize=13, fontweight='bold', color=LIGHT)
fig.text(0.5, 0.950, 'Sample Distributions  —  Matched Pairs t-Test   (n = 30 Counties)',
         ha='center', va='top', fontsize=10, color=MUTED)

ax_tbl = fig.add_subplot(gs[0])
ax_tbl.set_facecolor(PANEL)
ax_tbl.axis('off')

ax_tbl.text(0.5, 0.97, 'Our Sample Distribution',
            ha='center', va='top', fontsize=12, fontweight='bold',
            color=BLUE, transform=ax_tbl.transAxes)
ax_tbl.text(0.5, 0.87, 'Differences in Chronic Absenteeism Rate  (d = Post − Pre)',
            ha='center', va='top', fontsize=10, color=MUTED, transform=ax_tbl.transAxes)

col_labels = ['n', 'mean', 'SD', 'min', 'Q₁', 'median', 'Q₃', 'max']
row_data = [[str(n), f'{d_bar:.3f}', f'{s_d:.3f}',
             f'{d.min():.3f}', f'{q1:.3f}', f'{med:.3f}',
             f'{q3:.3f}', f'{d.max():.3f}']]

tbl = ax_tbl.table(cellText=row_data, colLabels=col_labels,
                   cellLoc='center', loc='center', bbox=[0.05, 0.05, 0.90, 0.62])
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID_C)
    if r == 0:
        cell.set_facecolor('#3a4a60')
        cell.set_text_props(color=LIGHT, fontweight='bold')
    else:
        cell.set_facecolor('#2e3550')
        cell.set_text_props(color=LIGHT)
    cell.set_height(0.28)

ax_dot = fig.add_subplot(gs[1])
ax_dot.set_facecolor(PANEL)

np.random.seed(42)
jitter = np.random.uniform(-0.18, 0.18, n)
ax_dot.scatter(d, jitter, color=BLUE, s=62, zorder=3,
               edgecolors=LIGHT, linewidths=0.35, alpha=0.92)
ax_dot.axhline(0, color=GRID_C, linewidth=0.9, linestyle='--')
ax_dot.axvline(d_bar, color=MEAN_C, linewidth=2, linestyle='--',
               label=f'Mean = {d_bar:.3f}', zorder=4)
ax_dot.set_yticks([])
ax_dot.set_xlabel('Difference in Chronic Absenteeism Rate (Post − Pre)', fontsize=10)
ax_dot.set_title('Dot Plot of Differences', fontsize=11, fontweight='bold', color=LIGHT, pad=8)
ax_dot.set_xlim(-0.09, 0.16)
ax_dot.xaxis.set_minor_locator(MultipleLocator(0.01))
ax_dot.grid(axis='x', which='major', color=GRID_C, linestyle='-', alpha=0.8)
for s in ['top', 'right', 'left']:
    ax_dot.spines[s].set_visible(False)
ax_dot.spines['bottom'].set_edgecolor(GRID_C)
ax_dot.legend(loc='upper right', fontsize=9, facecolor=PANEL, edgecolor=GRID_C, labelcolor=LIGHT)

ax_box = fig.add_subplot(gs[2])
ax_box.set_facecolor(PANEL)

ax_box.boxplot(d, vert=False, patch_artist=True, widths=0.45,
               boxprops=dict(facecolor='#3a5f80', color=BLUE, linewidth=1.8),
               medianprops=dict(color=LIGHT, linewidth=2.5),
               whiskerprops=dict(color=BLUE, linewidth=1.5),
               capprops=dict(color=BLUE, linewidth=1.8),
               flierprops=dict(marker='o', color=MEAN_C, markersize=6, markeredgecolor=MEAN_C))
ax_box.axvline(d_bar, color=MEAN_C, linewidth=2, linestyle='--',
               label=f'Mean = {d_bar:.3f}', zorder=4)
ax_box.set_yticks([1])
ax_box.set_yticklabels(['Differences\n(Post − Pre)'], fontsize=10)
ax_box.set_xlabel('Difference in Chronic Absenteeism Rate (Post − Pre)', fontsize=10)
ax_box.set_title('Box Plot of Differences', fontsize=11, fontweight='bold', color=LIGHT, pad=8)
ax_box.set_xlim(-0.09, 0.16)
ax_box.xaxis.set_minor_locator(MultipleLocator(0.01))
ax_box.grid(axis='x', which='major', color=GRID_C, linestyle='-', alpha=0.8)
for s in ['top', 'right']:
    ax_box.spines[s].set_visible(False)
ax_box.spines['bottom'].set_edgecolor(GRID_C)
ax_box.spines['left'].set_edgecolor(GRID_C)
ax_box.legend(loc='upper right', fontsize=9, facecolor=PANEL, edgecolor=GRID_C, labelcolor=LIGHT)

result_text = (
    f"Matched Pairs t-Test   |   H₀: μ_d = 0    Hₐ: μ_d ≠ 0    α = 0.05\n"
    f"t = {t_stat:.3f}     df = {df_t}     p-value ≈ {p_value:.2e}     → REJECT H₀"
)
fig.text(0.5, 0.01, result_text, ha='center', va='bottom', fontsize=10.5, color=LIGHT,
         bbox=dict(boxstyle='round,pad=0.6', facecolor=PANEL, edgecolor=BLUE, linewidth=1.5),
         fontfamily='monospace')

plt.savefig('matched_pairs_v3.png', dpi=180, bbox_inches='tight', facecolor=BG)