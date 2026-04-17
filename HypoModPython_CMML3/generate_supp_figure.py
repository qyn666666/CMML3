

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'lines.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def compute_isi_histogram(spike_times_ms, bin_size=5, max_isi=2000):
    if len(spike_times_ms) < 2:
        return np.array([]), np.array([])
    isis = np.diff(spike_times_ms)
    isis = isis[(isis > 0) & (isis <= max_isi)]
    bins = np.arange(0, max_isi + bin_size, bin_size)
    hist, edges = np.histogram(isis, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    return hist, centres


def normalise_histogram(hist, target=10000):
    s = hist.sum()
    return hist * target / s if s > 0 else hist.copy()


def compute_hazard(spike_times_ms, bin_size=5, max_isi=2000):
    if len(spike_times_ms) < 2:
        return np.array([]), np.array([])
    isis = np.diff(spike_times_ms)
    isis = isis[(isis > 0) & (isis <= max_isi)]
    bins = np.arange(0, max_isi + bin_size, bin_size)
    hist, edges = np.histogram(isis, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    survived = len(isis)
    hazard = np.zeros(len(hist), dtype=float)
    for i, count in enumerate(hist):
        hazard[i] = count / survived * 100 if survived > 0 else 0
        survived -= count
    return hazard, centres


# Load data 
print("Loading oxydata dap cells.xlsx...")
col_names_row = pd.read_excel('oxydata dap cells.xlsx', sheet_name='Sheet1',
                               header=None, nrows=1).iloc[0].tolist()
df = pd.read_excel('oxydata dap cells.xlsx', sheet_name='Sheet1',
                   header=None, skiprows=6)
df.columns = col_names_row

neurons = {}
for col in df.columns:
    vals = pd.to_numeric(df[col], errors='coerce').dropna().values
    vals = vals[vals > 0]
    neurons[col] = np.sort(vals) * 1000.0  # s → ms

# DAP detection: early/main ISI ratio (0-50ms bins vs 50-200ms bins) > 1.5
DISPLAY_MAX = 500

def dap_ratio(times):
    if len(times) < 2: return 0.0
    isis = np.diff(times); isis = isis[isis > 0]
    bins = np.arange(0, 2005, 5)
    hist, _ = np.histogram(isis, bins=bins)
    early = hist[:10].mean()
    main  = hist[10:40].mean()
    return early / main if main > 0 else 0.0

dap_labels = {}
for name, times in neurons.items():
    if len(times) > 1:
        isis = np.diff(times); isis = isis[isis > 0]
        pct = 100 * np.sum(isis < 50) / len(isis)
        dap_labels[name] = pct

# Plot: 4-row × 4-col layout (group 1: neurons 0-3, group 2: neurons 4-6) 
neuron_list  = list(neurons.items())
group1       = neuron_list[:4]   # rows 0-1
group2       = neuron_list[4:]   # rows 2-3  (3 neurons, last column empty)
N_COLS       = 4

fig = plt.figure(figsize=(20, 14))

# 5 rows: 0=hist1, 1=haz1, 2=spacer, 3=hist2, 4=haz2
gs = gridspec.GridSpec(5, N_COLS, figure=fig,
                       height_ratios=[1, 1, 0.35, 1, 1],
                       hspace=0.65, wspace=0.35)

# Collect axes so we can unify ylims afterwards
hist_axes = []
haz_axes  = []

def plot_neuron(row_hist, row_haz, col, name, times):
    pct_short = dap_labels.get(name, 0)
    is_dap    = dap_ratio(times) > 1.5
    bar_color = 'steelblue' if is_dap else '#999999'
    haz_color = 'tomato'    if is_dap else '#bbbbbb'

    hist, bins = compute_isi_histogram(times, bin_size=5)
    haz,  _    = compute_hazard(times, bin_size=5)
    hist_n     = normalise_histogram(hist)
    mask       = bins <= DISPLAY_MAX
    early_mask = (bins <= 50) & mask

    ax_h = fig.add_subplot(gs[row_hist, col])
    ax_z = fig.add_subplot(gs[row_haz,  col])
    hist_axes.append(ax_h)
    haz_axes.append(ax_z)

    # ISI histogram
    ax_h.bar(bins[mask], hist_n[mask], width=4.5, color=bar_color, alpha=0.85)
    if is_dap:
        ax_h.bar(bins[early_mask], hist_n[early_mask], width=4.5,
                 color='darkorange', alpha=0.9)
    title_str = name + ('\n★ DAP+' if is_dap else f'\n({pct_short:.1f}% ISI<50ms)')
    ax_h.set_title(title_str)
    ax_h.set_xlabel('ISI (ms)')
    ax_h.set_ylabel('Count (norm. to 10k)')
    ax_h.set_xlim(0, DISPLAY_MAX)

    n_spikes = len(times)
    if n_spikes > 1:
        mean_isi = np.mean(np.diff(times))
        freq     = 1000 / mean_isi if mean_isi > 0 else 0
        ax_h.text(0.97, 0.97, f'n={n_spikes}\n{freq:.1f} Hz',
                  transform=ax_h.transAxes, ha='right', va='top', fontsize=14,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='none', alpha=0.7))

    # Hazard function
    ax_z.bar(bins[mask], haz[mask], width=4.5, color=haz_color, alpha=0.85)
    if is_dap:
        ax_z.bar(bins[early_mask], haz[early_mask], width=4.5,
                 color='crimson', alpha=0.9)
    ax_z.set_xlabel('ISI (ms)')
    ax_z.set_ylabel('Hazard (%)')
    ax_z.set_xlim(0, DISPLAY_MAX)

# Group 1 (rows 0-1, cols 0-3)
for col, (name, times) in enumerate(group1):
    plot_neuron(0, 1, col, name, times)

# Group 2 (rows 3-4, cols 0-2; row 2 is spacer, col 3 left empty)
for col, (name, times) in enumerate(group2):
    plot_neuron(3, 4, col, name, times)

# Unify y-axis limits within plot type
hist_ymax = max(ax.get_ylim()[1] for ax in hist_axes)
haz_ymax  = max(ax.get_ylim()[1] for ax in haz_axes)
for ax in hist_axes:
    ax.set_ylim(0, hist_ymax)
for ax in haz_axes:
    ax.set_ylim(0, haz_ymax)

# Legend 
dap_patch  = mpatches.Patch(color='darkorange', label='DAP-positive early peak (≤50 ms)')
ctrl_patch = mpatches.Patch(color='#999999',    label='DAP-negative')
fig.legend(handles=[dap_patch, ctrl_patch],
           loc='lower center', ncol=2, fontsize=15,
           bbox_to_anchor=(0.5, -0.03))

fig.savefig('fig_supp_isi_all_neurons.png', dpi=300, bbox_inches='tight')
print("Saved: fig_supp_isi_all_neurons.png")

# Print table
print(f"\n{'Neuron':<14} {'Spikes':>7} {'Freq (Hz)':>10} {'%ISI<50ms':>10}  DAP?")
print('-' * 55)
for name, times in neurons.items():
    if len(times) > 1:
        isis = np.diff(times)
        isis = isis[isis > 0]
        freq = 1000 / np.mean(isis)
        pct  = 100 * np.sum(isis < 50) / len(isis)
        flag = '★ YES' if dap_ratio(times) > 1.5 else 'no'
        print(f"{name:<14} {len(times):>7d} {freq:>10.2f} {pct:>10.1f}  {flag}")
