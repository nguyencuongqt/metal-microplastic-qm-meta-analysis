#!/usr/bin/env python3
"""
Figure 3 - EMVP (Environmental Metal Vector Potential) screening, R_water framework.

Three panels:
  A  Forest/dot plot of R_water = EMVP / C_water at C_MP = 0.01 mg/L (95% HDI),
     with a secondary top axis showing % of C_water.
  B  Heat map of log10(R_water) across three C_MP scenarios (0.001, 0.01, 1 mg/L),
     with the "elevated" (0.01) column highlighted.
  C  Ranking panel: EMVP and C_water (both in ug/L, shared log axis) per metal,
     ordered by R_water; R_water annotated at right. The visual gap between the
     two bars *is* R_water -> ranking is driven by low C_water, not high EMVP.

Outputs: `figures/main/Figure_3.pdf`, `figures/main/Figure_3.png`, and
`results/environmental/08_rwater_asia_summary.csv`.

Data are the verified values from the manuscript Table S4
(C_water = Asia column, Zhou et al. 2020, Glob. Ecol. Conserv. 22:e00925).
R_water is a mass-balance contextual ratio; it is NOT a risk quotient and assumes
no desorption. No regulatory benchmark (EPA/WHO/CCME) is used.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, NullFormatter

# ----------------------------------------------------------------------
# Style (Water Research): Arial/Helvetica, small fonts, no surrounding box
# ----------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.unicode_minus": True,
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
})

TEAL   = "#1D9E75"   # EMVP / main points
BLUE   = "#185FA5"   # highlight / C_water
ORANGE = "#E08214"   # C_water bars
RED    = "#E24B4A"   # R_water = 1 reference

# ----------------------------------------------------------------------
# VERIFIED DATA (do not alter) - ordered by descending R_water
# ----------------------------------------------------------------------
metals = ['Cd', 'Pb', 'Ni', 'Hg', 'As', 'Mn', 'Cu', 'Cr', 'Zn']

C_water = {'Cd': 17.75, 'Pb': 92.70, 'Ni': 54.84, 'Hg': 4.17,
           'As': 178.3, 'Mn': 967.8, 'Cu': 345.9, 'Cr': 383.9, 'Zn': 889.6}

EMVP_mean = {'Cd': 0.0047, 'Pb': 0.0073, 'Ni': 0.0042, 'Hg': 0.00022,
             'As': 0.0049, 'Mn': 0.015, 'Cu': 0.0047, 'Cr': 0.0031, 'Zn': 0.0045}

R_water_mean = {'Cd': 2.60e-4, 'Pb': 7.90e-5, 'Ni': 7.70e-5, 'Hg': 5.30e-5,
                'As': 2.70e-5, 'Mn': 1.60e-5, 'Cu': 1.40e-5, 'Cr': 8.10e-6, 'Zn': 5.10e-6}

# 95% HDIs propagated from the primary-model posterior at the fixed baseline
# (aged PS, mean temperature), then divided by the total-water C_water values.
HDI_low  = {'Cd': 1.1492e-4, 'Pb': 3.6024e-5, 'Ni': 2.9104e-5, 'Hg': 2.1756e-6,
            'As': 4.4206e-6, 'Mn': 3.3672e-8, 'Cu': 6.1434e-6, 'Cr': 1.7682e-6,
            'Zn': 1.9490e-6}
HDI_high = {'Cd': 4.4595e-4, 'Pb': 1.3112e-4, 'Ni': 1.3762e-4, 'Hg': 2.5462e-4,
            'As': 6.6717e-5, 'Mn': 5.2032e-5, 'Cu': 2.2388e-5, 'Cr': 1.7120e-5,
            'Zn': 9.0150e-6}

# Heatmap: rows = metals, cols = scenarios (R_water scales linearly with C_MP)
factors = [0.1, 1.0, 100.0]                 # x relative to base 0.01 mg/L
scenario_top = ['Low', 'Elev.', 'Extr.']
scenario_bot = ['0.001', '0.01', '1']
heatmap = np.array([[np.log10(R_water_mean[m] * f) for f in factors] for m in metals])

# ----------------------------------------------------------------------
# Figure layout
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18 / 2.54, 11 / 2.54),
                         gridspec_kw={'width_ratios': [2.4, 1.45, 2.4]})
ax_a, ax_b, ax_c = axes
y = np.arange(len(metals))

# ===================== PANEL A: dot / forest plot =====================
r_means = np.array([R_water_mean[m] for m in metals])
xerr_lo = np.array([R_water_mean[m] - HDI_low[m] for m in metals])
xerr_hi = np.array([HDI_high[m] - R_water_mean[m] for m in metals])

for i, m in enumerate(metals):
    ax_a.errorbar(r_means[i], y[i], xerr=[[xerr_lo[i]], [xerr_hi[i]]],
                  fmt='o', ms=5, mfc=TEAL, mec=TEAL,
                  ecolor=TEAL, elinewidth=1.0, capsize=2.2, mew=1.0,
                  capthick=1.0, ls='none', zorder=3)

ax_a.axvline(1, color=RED, ls='--', lw=1.0, zorder=2)
ax_a.text(1, -0.35, 'R$_{water}$ = 1 ', color=RED, fontsize=7.5,
          va='bottom', ha='right', rotation=90)
ax_a.set_xscale('log')
ax_a.set_xlim(1e-7, 2e0)
ax_a.set_ylim(-0.6, len(metals) - 0.4)
ax_a.set_yticks(y)
ax_a.set_yticklabels(metals)
ax_a.invert_yaxis()
ax_a.set_xlabel('R$_{water}$ = EMVP / C$_{water}$', fontsize=8)
ax_a.xaxis.set_major_locator(LogLocator(base=10, numticks=9))
ax_a.xaxis.set_minor_formatter(NullFormatter())
ax_a.grid(axis='x', ls=':', lw=0.4, color='0.8', zorder=0)
for s in ['top', 'right']:
    ax_a.spines[s].set_visible(False)

# secondary top axis: % of C_water = R_water * 100
ax_a_top = ax_a.secondary_xaxis('top', functions=(lambda v: v * 100, lambda v: v / 100))
ax_a_top.set_xlabel('% of C$_{water}$', fontsize=8)
ax_a_top.set_xscale('log')

ax_a.text(0.04, 0.04, 'All metals < 10$^{-3}$\nMax: Cd = 2.6x10$^{-4}$',
          transform=ax_a.transAxes, fontsize=7, va='bottom', ha='left',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', lw=0.5))
ax_a.set_title('A', loc='left', fontweight='bold', fontsize=10)

# ===================== PANEL B: heatmap ===============================
im = ax_b.imshow(heatmap, aspect='auto', cmap='Blues',
                 vmin=-7, vmax=0, origin='upper')
ax_b.set_xticks([0, 1, 2])
ax_b.set_xticklabels([f'{t}\n{b}' for t, b in zip(scenario_top, scenario_bot)],
                     fontsize=7)
ax_b.set_xlabel('C$_{MP}$ (mg/L)', fontsize=8)
ax_b.set_yticks(y)
ax_b.set_yticklabels(metals, fontsize=8)
for i in range(len(metals)):
    for j in range(3):
        val = heatmap[i, j]
        ax_b.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7,
                  color=('white' if val > -2.5 else 'black'))
for s in ['top', 'bottom', 'left', 'right']:
    ax_b.spines[s].set_visible(False)
ax_b.tick_params(length=0)
# highlight the "elevated" (0.01 mg/L) column
ax_b.add_patch(Rectangle((0.5, -0.5), 1, len(metals), fill=False,
                         edgecolor=BLUE, lw=1.6, zorder=5))
cbar = plt.colorbar(im, ax=ax_b, shrink=0.55, pad=0.06)
cbar.set_label('log$_{10}$(R$_{water}$)', fontsize=7)
cbar.ax.tick_params(labelsize=6.5)
ax_b.set_title('B', loc='left', fontweight='bold', fontsize=10)

# ===================== PANEL C: EMVP vs C_water (shared ug/L) =========
emvp_vals = np.array([EMVP_mean[m] for m in metals])
cw_vals   = np.array([C_water[m] for m in metals])
h = 0.36
ax_c.barh(y - h / 2, emvp_vals, height=h, color=TEAL,   label='EMVP', zorder=3)
ax_c.barh(y + h / 2, cw_vals,   height=h, color=ORANGE, label='C$_{water}$', zorder=3)
ax_c.set_xscale('log')
ax_c.set_xlim(1e-4, 1e5)
ax_c.set_ylim(-0.6, len(metals) - 0.4)
ax_c.set_yticks(y)
ax_c.set_yticklabels(metals)
ax_c.invert_yaxis()
ax_c.set_xlabel('Concentration (µg/L, log scale)', fontsize=8)
ax_c.grid(axis='x', ls=':', lw=0.4, color='0.8', zorder=0)
for s in ['top', 'right']:
    ax_c.spines[s].set_visible(False)
# R_water annotation at the right margin of each cluster
for i, m in enumerate(metals):
    ax_c.text(0.99, y[i], f'R={R_water_mean[m]:.1e}', transform=ax_c.get_yaxis_transform(),
              va='center', ha='right', fontsize=6.2, color='0.25')
ax_c.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7,
            frameon=False, handlelength=1.1, columnspacing=1.2, borderaxespad=0.2)
ax_c.set_title('C', loc='left', fontweight='bold', fontsize=10)
ax_c.text(0.5, -0.20, 'Ranking driven by low C$_{water}$, not high EMVP',
          transform=ax_c.transAxes, ha='center', fontsize=7, style='italic')

# ----------------------------------------------------------------------
plt.tight_layout(pad=0.6, w_pad=2.0)
plt.subplots_adjust(bottom=0.16, top=0.88)
output_dir = Path(__file__).resolve().parents[1] / 'figures' / 'main'
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'Figure_3.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'Figure_3.png', dpi=300, bbox_inches='tight')

result_dir = Path(__file__).resolve().parents[1] / 'results' / 'environmental'
result_dir.mkdir(parents=True, exist_ok=True)
n_water_bodies = {'Cd': 129, 'Pb': 153, 'Ni': 89, 'Hg': 31, 'As': 60,
                  'Mn': 42, 'Cu': 113, 'Cr': 102, 'Zn': 122}
with (result_dir / '08_rwater_asia_summary.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['metal', 'c_water_asia_ug_L', 'n_water_bodies', 'emvp_mean_ug_L',
                     'r_water_mean', 'r_water_hdi95_lower', 'r_water_hdi95_upper',
                     'percent_c_water_mean'])
    for metal in metals:
        writer.writerow([metal, C_water[metal], n_water_bodies[metal], EMVP_mean[metal],
                         R_water_mean[metal], HDI_low[metal], HDI_high[metal],
                         R_water_mean[metal] * 100])

print(f'Saved Figure 3 to {output_dir} and R_water summary to {result_dir}')
