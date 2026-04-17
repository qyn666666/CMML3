
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script use

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from copy import deepcopy

plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ISI analysis helpers 

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


# Integrate-and-fire model 

DEFAULT_PARAMS = {
    'runtime': 500, 'hstep': 1,
    'Vrest': -62.0, 'Vthresh': -50.0, 'halflifeMem': 7.5,
    'psprate': 300, 'pspratio': 1.0, 'pspmag': 3.0,
    'kHAP': 60.0, 'halflifeHAP': 8.0,
    'kAHP': 0.5,  'halflifeAHP': 500.0,
    'kDAP': 0.0,  'halflifeDAP': 50.0,
    'kNMDA': 0.0, 'halflifeNMDA': 80.0, 'MgConc': 1.0,
    'dap_mode': 'none',
}


def run_model(params, seed=None):
    if seed is not None:
        random.seed(seed)
    p = params
    runtime_ms = int(p['runtime']) * 1000
    Vrest, Vthresh, pspmag = p['Vrest'], p['Vthresh'], p['pspmag']
    absref = 2

    tauMem  = math.log(2) / p['halflifeMem']
    tauHAP  = math.log(2) / p['halflifeHAP']
    tauAHP  = math.log(2) / p['halflifeAHP']
    tauDAP  = math.log(2) / p['halflifeDAP']  if p['halflifeDAP']  > 0 else 1.0
    tauNMDA = math.log(2) / p['halflifeNMDA'] if p['halflifeNMDA'] > 0 else 1.0

    epsprate = p['psprate'] / 1000.0
    ipsprate = epsprate * p['pspratio']
    dap_mode = p['dap_mode']

    V = Vrest
    tPSP = tHAP = tAHP = tDAP = tNMDA = 0.0
    ttime = epspt = ipspt = 0.0
    spike_times = []

    for i in range(1, runtime_ms + 1):
        ttime += 1
        nepsp = nipsp = 0
        if epsprate > 0:
            while epspt < 1:
                nepsp += 1
                epspt += -math.log(1 - random.random()) / epsprate
            epspt -= 1
        if ipsprate > 0:
            while ipspt < 1:
                nipsp += 1
                ipspt += -math.log(1 - random.random()) / ipsprate
            ipspt -= 1
        inputPSP = nepsp * pspmag - nipsp * pspmag

        tPSP = tPSP + inputPSP - tPSP * tauMem
        tHAP = tHAP - tHAP * tauHAP
        tAHP = tAHP - tAHP * tauAHP

        if dap_mode == 'simple':
            tDAP = tDAP - tDAP * tauDAP
            V = Vrest + tPSP - tHAP - tAHP + tDAP
        elif dap_mode == 'nmda':
            tNMDA = tNMDA - tNMDA * tauNMDA
            V_base = Vrest + tPSP - tHAP - tAHP
            MgFactor = 1.0 / (1.0 + p['MgConc'] * math.exp(-0.062 * V_base))
            V = V_base + tNMDA * MgFactor
        else:
            V = Vrest + tPSP - tHAP - tAHP

        if V > Vthresh and ttime >= absref:
            spike_times.append(float(i))
            tHAP += p['kHAP']
            tAHP += p['kAHP']
            if dap_mode == 'simple':
                tDAP += p['kDAP']
            elif dap_mode == 'nmda':
                tNMDA += p['kNMDA']
            ttime = 0

    return np.array(spike_times)


def spike_freq(spike_times_ms, runtime_s):
    return len(spike_times_ms) / runtime_s if runtime_s > 0 else 0


# Load recorded data 

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

# DAP detection: early/main hazard ratio > 1.5 (0-50ms vs 50-200ms ISI count ratio)
def dap_ratio(times):
    if len(times) < 2:
        return 0.0
    isis = np.diff(times); isis = isis[isis > 0]
    bins = np.arange(0, 2005, 5)
    hist, _ = np.histogram(isis, bins=bins)
    early = hist[:10].mean()   # 0–50 ms bins
    main  = hist[10:40].mean() # 50–200 ms bins
    return early / main if main > 0 else 0.0

print(f"Loaded {len(neurons)} neurons:")
for name, times in neurons.items():
    if len(times) > 1:
        isis = np.diff(times); isis = isis[isis > 0]
        pct_short = 100 * np.sum(isis < 50) / len(isis)
        ratio = dap_ratio(times)
        dap_flag = "DAP+" if ratio > 1.5 else "DAP-"
        print(f"  {name}: {len(times)} spikes, "
              f"freq={1000/np.mean(isis):.2f} Hz, "
              f"{pct_short:.1f}% ISI<50ms, ratio={ratio:.2f} {dap_flag}")

# Identify DAP candidate: DAP+ neuron with highest %ISI<50ms
def pct_short_isi(times):
    if len(times) < 2: return 0.0
    isis = np.diff(times); isis = isis[isis > 0]
    return 100 * np.sum(isis < 50) / len(isis)

dap_cell = max(
    [n for n in neurons if dap_ratio(neurons[n]) > 1.5],
    key=lambda n: pct_short_isi(neurons[n])
)
print(f"\nDAP candidate selected: {dap_cell} "
      f"(ratio={dap_ratio(neurons[dap_cell]):.2f}, "
      f"{pct_short_isi(neurons[dap_cell]):.1f}% ISI<50ms)")
cell_times = neurons[dap_cell]

# Match baseline firing rate to the recorded cell
cell_isis = np.diff(cell_times)
cell_isis = cell_isis[cell_isis > 0]
cell_freq_hz = 1000.0 / np.mean(cell_isis)
print(f"  Cell mean freq: {cell_freq_hz:.2f} Hz")

# Run baseline model to roughly match firing rate 
# Sweep psprate to find one giving ~cell_freq_hz output
print("\nMatching baseline firing rate...")
best_prate = 300
best_diff = 1e9
for prate in range(200, 700, 25):
    p = deepcopy(DEFAULT_PARAMS)
    p['psprate'] = prate
    p['runtime'] = 200
    p['dap_mode'] = 'none'
    sp = run_model(p, seed=SEED)
    freq = spike_freq(sp, 200)
    diff = abs(freq - cell_freq_hz)
    if diff < best_diff:
        best_diff = diff
        best_prate = prate
print(f"  Best psprate for baseline match: {best_prate} Hz")

# Run models for Figure 1 
RUNTIME_FIG1 = 1000   # seconds per condition
N_REPS       = 2      # repeat runs (concatenated for smoother histograms)

def concat_runs(params, n=N_REPS, rt=RUNTIME_FIG1):
    p = deepcopy(params)
    p['runtime'] = rt
    all_sp = []
    for s in range(n):
        sp = run_model(p, seed=SEED + s)
        all_sp.append(sp + s * rt * 1000)
    return np.concatenate(all_sp)

print("\nRunning Figure 1 models (this may take a minute)...")

p_noDAP = deepcopy(DEFAULT_PARAMS)
p_noDAP['psprate']  = best_prate
p_noDAP['dap_mode'] = 'none'

p_simpDAP = deepcopy(DEFAULT_PARAMS)
p_simpDAP['psprate']     = best_prate
p_simpDAP['dap_mode']    = 'simple'
p_simpDAP['kDAP']        = 15.0
p_simpDAP['halflifeDAP'] = 20.0

p_nmdaDAP = deepcopy(DEFAULT_PARAMS)
p_nmdaDAP['psprate']      = best_prate
p_nmdaDAP['dap_mode']     = 'nmda'
p_nmdaDAP['kNMDA']        = 400.0   # large amplitude needed to overcome Mg2+ block at low firing rates
p_nmdaDAP['halflifeNMDA'] = 20.0
p_nmdaDAP['MgConc']       = 1.0

spk_noDAP   = concat_runs(p_noDAP)
print(f"  no-DAP:   {len(spk_noDAP)} spikes, "
      f"{spike_freq(spk_noDAP, RUNTIME_FIG1*N_REPS):.2f} Hz")

spk_simpDAP = concat_runs(p_simpDAP)
print(f"  simp-DAP: {len(spk_simpDAP)} spikes, "
      f"{spike_freq(spk_simpDAP, RUNTIME_FIG1*N_REPS):.2f} Hz")

spk_nmdaDAP = concat_runs(p_nmdaDAP)
print(f"  NMDA-DAP: {len(spk_nmdaDAP)} spikes, "
      f"{spike_freq(spk_nmdaDAP, RUNTIME_FIG1*N_REPS):.2f} Hz")

# Figure 1 
print("\nPlotting Figure 1...")
DISPLAY_MAX = 400

datasets = [
    (cell_times,  f'(A) Data: {dap_cell}', 'steelblue'),
    (spk_noDAP,   '(B) Model: No DAP',      '#888888'),
    (spk_simpDAP, '(C) Model: Simple DAP',  'darkorange'),
    (spk_nmdaDAP, '(D) Model: NMDA DAP',    'forestgreen'),
]

fig1, axes1 = plt.subplots(2, 4, figsize=(12, 6), sharey='row')
for ax in axes1.flat:
    ax.tick_params(labelleft=True)

for col, (spk, label, colour) in enumerate(datasets):
    label = label.split(') ', 1)[-1]  # strip "(A) " prefix
    hist, bins = compute_isi_histogram(spk, bin_size=5)
    haz, _     = compute_hazard(spk, bin_size=5)
    hist_n     = normalise_histogram(hist)
    mask       = bins <= DISPLAY_MAX

    ax_h = axes1[0, col]
    ax_z = axes1[1, col]

    ax_h.bar(bins[mask], hist_n[mask], width=4.5, color=colour, alpha=0.85)
    ax_h.set_title(label)
    ax_h.set_xlabel('ISI (ms)')
    if col == 0:
        ax_h.set_ylabel('Count (norm.)')

    ax_z.bar(bins[mask], haz[mask], width=4.5, color=colour, alpha=0.85)
    ax_z.set_xlabel('ISI (ms)')
    if col == 0:
        ax_z.set_ylabel('Hazard (%)')

plt.tight_layout()
fig1.savefig('figure1_isi_fits.png', dpi=300, bbox_inches='tight')
print("  Saved: figure1_isi_fits.png")

# F-I curves 
print("\nComputing F-I curves...")
PSP_RATES  = [100, 150, 200, 250, 300, 350, 400, 450, 500]
RUNTIME_FI = 200  # seconds per point

results_noDAP   = []
results_simpDAP = []
results_nmdaDAP = []

for prate in PSP_RATES:
    # No DAP
    p = deepcopy(DEFAULT_PARAMS)
    p['runtime'] = RUNTIME_FI; p['psprate'] = prate; p['dap_mode'] = 'none'
    sp = run_model(p, seed=SEED)
    results_noDAP.append(spike_freq(sp, RUNTIME_FI))

    # Simple DAP
    p['dap_mode'] = 'simple'; p['kDAP'] = 15.0; p['halflifeDAP'] = 20.0
    sp = run_model(p, seed=SEED)
    results_simpDAP.append(spike_freq(sp, RUNTIME_FI))

    # NMDA DAP
    p['dap_mode'] = 'nmda'; p['kNMDA'] = 400.0; p['halflifeNMDA'] = 20.0; p['MgConc'] = 1.0
    sp = run_model(p, seed=SEED)
    results_nmdaDAP.append(spike_freq(sp, RUNTIME_FI))
    print(f"  psprate={prate}: noDAP={results_noDAP[-1]:.2f} "
          f"simpDAP={results_simpDAP[-1]:.2f} "
          f"nmdaDAP={results_nmdaDAP[-1]:.2f} Hz")

# Secretion model 
print("\nRunning secretion model...")

SP = {
    'kE': 1.5, 'halflifeE': 100, 'Eth': 12,
    'Bbase': 0.5, 'kB': 0.021, 'halflifeB': 2000,
    'kC': 0.0003, 'halflifeC': 20000, 'Cth': 0.14,
    'alpha': 0.003 / 1000,
    'Pmax': 5000, 'Rmax': 1000000, 'Rinit': 1000000,
    'beta': 120, 'VolPlasma': 100,
    'tauClear': math.log(2) / 68 / 1000,
}


def run_secretion(spike_times_ms, runtime_s=500):
    runtime_ms = runtime_s * 1000
    tauE = math.log(2) / SP['halflifeE']
    tauB = math.log(2) / SP['halflifeB']
    tauC = math.log(2) / SP['halflifeC']
    Ethpow = SP['Eth'] ** 5
    Cthpow = SP['Cth'] ** 3

    tE = tB = 0.0; tC = 0.03
    tP = SP['Pmax']; tR = SP['Rinit']; tPlasma = 0.0

    spike_set = set(spike_times_ms.astype(int))
    plasma_trace = np.zeros(runtime_s)

    for i in range(1, runtime_ms + 1):
        tE = tE - tE * tauE
        tB = tB - tB * tauB
        tC = tC - tC * tauC
        EKpow = tE ** 5; CKpow = tC ** 3
        Einh  = 1 - EKpow / (EKpow + Ethpow)
        Cinh  = 1 - CKpow / (CKpow + Cthpow)
        CaEnt = Einh * Cinh * (tB + SP['Bbase'])
        secX  = tE ** 2 * SP['alpha'] * tP
        fillP = SP['beta'] * tR / SP['Rmax'] if tP < SP['Pmax'] else 0
        tP = tP - secX + fillP
        tR = tR - fillP
        tPlasma = tPlasma + secX - tPlasma * SP['tauClear']
        if i in spike_set:
            tE += SP['kE'] * CaEnt
            tB += SP['kB']
            tC += SP['kC'] * CaEnt
        if i % 1000 == 0:
            plasma_trace[i // 1000 - 1] = tPlasma / SP['VolPlasma']

    return np.arange(runtime_s), plasma_trace


RT_SEC = 500
p_sec_base = deepcopy(DEFAULT_PARAMS)
p_sec_base['runtime'] = RT_SEC; p_sec_base['psprate'] = best_prate; p_sec_base['dap_mode'] = 'none'

p_sec_simp = deepcopy(DEFAULT_PARAMS)
p_sec_simp['runtime'] = RT_SEC; p_sec_simp['psprate'] = best_prate
p_sec_simp['dap_mode'] = 'simple'; p_sec_simp['kDAP'] = 15.0; p_sec_simp['halflifeDAP'] = 20.0

spk_sec_base = run_model(p_sec_base, seed=SEED)
spk_sec_simp = run_model(p_sec_simp, seed=SEED)

t_base, plasma_base = run_secretion(spk_sec_base, runtime_s=RT_SEC)
t_simp, plasma_simp = run_secretion(spk_sec_simp, runtime_s=RT_SEC)

pct_inc = 100 * (plasma_simp.sum() - plasma_base.sum()) / (plasma_base.sum() + 1e-9)
print(f"  Secretion increase with Simple DAP: {pct_inc:.1f}%")

# Varying-input helpers 

def run_model_varying(params, psprate_array, seed=None):
    """Run the integrate-and-fire model with a time-varying PSP rate (Hz/step)."""
    if seed is not None:
        random.seed(seed)
    p = params
    runtime_ms = len(psprate_array)
    Vrest, Vthresh, pspmag = p['Vrest'], p['Vthresh'], p['pspmag']
    absref = 2

    tauMem  = math.log(2) / p['halflifeMem']
    tauHAP  = math.log(2) / p['halflifeHAP']
    tauAHP  = math.log(2) / p['halflifeAHP']
    tauDAP  = math.log(2) / p['halflifeDAP']  if p['halflifeDAP']  > 0 else 1.0
    tauNMDA = math.log(2) / p['halflifeNMDA'] if p['halflifeNMDA'] > 0 else 1.0
    dap_mode = p['dap_mode']

    V = Vrest
    tPSP = tHAP = tAHP = tDAP = tNMDA = 0.0
    ttime = epspt = ipspt = 0.0
    spike_times = []

    for i in range(runtime_ms):
        ttime += 1
        epsprate = psprate_array[i] / 1000.0
        ipsprate = epsprate * p['pspratio']

        nepsp = nipsp = 0
        if epsprate > 0:
            while epspt < 1:
                nepsp += 1
                epspt += -math.log(1 - random.random()) / epsprate
            epspt -= 1
        if ipsprate > 0:
            while ipspt < 1:
                nipsp += 1
                ipspt += -math.log(1 - random.random()) / ipsprate
            ipspt -= 1
        inputPSP = nepsp * pspmag - nipsp * pspmag

        tPSP = tPSP + inputPSP - tPSP * tauMem
        tHAP = tHAP - tHAP * tauHAP
        tAHP = tAHP - tAHP * tauAHP

        if dap_mode == 'simple':
            tDAP = tDAP - tDAP * tauDAP
            V = Vrest + tPSP - tHAP - tAHP + tDAP
        elif dap_mode == 'nmda':
            tNMDA = tNMDA - tNMDA * tauNMDA
            V_base = Vrest + tPSP - tHAP - tAHP
            MgFactor = 1.0 / (1.0 + p['MgConc'] * math.exp(-0.062 * V_base))
            V = V_base + tNMDA * MgFactor
        else:
            V = Vrest + tPSP - tHAP - tAHP

        if V > Vthresh and ttime >= absref:
            spike_times.append(float(i + 1))
            tHAP += p['kHAP']
            tAHP += p['kAHP']
            if dap_mode == 'simple':
                tDAP += p['kDAP']
            elif dap_mode == 'nmda':
                tNMDA += p['kNMDA']
            ttime = 0

    return np.array(spike_times)


def compute_rate_timeseries(spike_times_ms, runtime_ms, bin_ms=1000):
    """Bin spike times into a 1-s firing rate time series (Hz)."""
    n_bins = runtime_ms // bin_ms
    rate = np.zeros(n_bins)
    for b in range(n_bins):
        t_start = b * bin_ms
        t_end   = t_start + bin_ms
        rate[b] = np.sum((spike_times_ms >= t_start) & (spike_times_ms < t_end))
    return rate


# Varying-input simulation (Figure 2C) 
print("\nRunning varying-input simulation (Figure 2C)...")
VAR_RUNTIME_S  = 400
var_runtime_ms = VAR_RUNTIME_S * 1000
t_ms_var = np.arange(var_runtime_ms, dtype=float)
psprate_arr = 300.0 + 150.0 * np.sin(2 * np.pi * 0.05 * t_ms_var / 1000.0)
psprate_arr = np.maximum(psprate_arr, 0.0)

p_v_noDAP = deepcopy(DEFAULT_PARAMS)
p_v_noDAP['dap_mode'] = 'none'

p_v_simpDAP = deepcopy(DEFAULT_PARAMS)
p_v_simpDAP['dap_mode']    = 'simple'
p_v_simpDAP['kDAP']        = 15.0
p_v_simpDAP['halflifeDAP'] = 20.0

p_v_nmdaDAP = deepcopy(DEFAULT_PARAMS)
p_v_nmdaDAP['dap_mode']     = 'nmda'
p_v_nmdaDAP['kNMDA']        = 400.0
p_v_nmdaDAP['halflifeNMDA'] = 20.0
p_v_nmdaDAP['MgConc']       = 1.0

spk_v_noDAP   = run_model_varying(p_v_noDAP,   psprate_arr, seed=SEED)
spk_v_simpDAP = run_model_varying(p_v_simpDAP, psprate_arr, seed=SEED)
spk_v_nmdaDAP = run_model_varying(p_v_nmdaDAP, psprate_arr, seed=SEED)

rate_v_noDAP   = compute_rate_timeseries(spk_v_noDAP,   var_runtime_ms)
rate_v_simpDAP = compute_rate_timeseries(spk_v_simpDAP, var_runtime_ms)
rate_v_nmdaDAP = compute_rate_timeseries(spk_v_nmdaDAP, var_runtime_ms)
t_bins_v = np.arange(VAR_RUNTIME_S)

for name, rate in [('no-DAP',   rate_v_noDAP),
                   ('simpDAP',  rate_v_simpDAP),
                   ('nmdaDAP',  rate_v_nmdaDAP)]:
    amp = (rate.max() - rate.min()) / 2
    print(f"  {name}: mean={rate.mean():.2f} Hz, modulation amp={amp:.2f} Hz")

# Figure 2 (3-panel: A=F-I, B=Secretion, C=Varying-input) 
print("\nPlotting Figure 2...")
fig2, (ax1, ax2, ax_var) = plt.subplots(1, 3, figsize=(13, 4))

ax1.plot(PSP_RATES, results_noDAP,   'o-', color='#888888',     lw=2, ms=7, label='No DAP')
ax1.plot(PSP_RATES, results_simpDAP, 's-', color='darkorange',  lw=2, ms=7, label='Simple DAP')
ax1.plot(PSP_RATES, results_nmdaDAP, '^-', color='forestgreen', lw=2, ms=7, label='NMDA DAP')
ax1.set_xlabel('Input PSP rate (Hz)')
ax1.set_ylabel('Output firing rate (Hz)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t_base, plasma_base, color='#888888',    lw=2, label='No DAP')
ax2.plot(t_simp, plasma_simp, color='darkorange', lw=2, label='Simple DAP')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Plasma oxytocin (arb. units)')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax_var.plot(t_bins_v, rate_v_noDAP,   color='#888888',    lw=1.5, label='No DAP')
ax_var.plot(t_bins_v, rate_v_nmdaDAP, color='forestgreen', lw=1.5, label='NMDA DAP')
ax_var.plot(t_bins_v, rate_v_simpDAP, color='darkorange',  lw=1.5, label='Simple DAP')
ax_var.set_xlabel('Time (s)')
ax_var.set_ylabel('Output firing rate (Hz)')
ax_var.legend(loc='upper right')
ax_var.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('figure2_signal_processing.png', dpi=300, bbox_inches='tight')
print("  Saved: figure2_signal_processing.png")

# Print summary 
print("\n=== Summary ===")
print(f"DAP cell analysed: {dap_cell}")
print(f"Fitted psprate (baseline match): {best_prate} Hz")
print(f"Simple DAP params: kDAP=15.0, halflifeDAP=20 ms")
print(f"NMDA DAP params:   kNMDA=400, halflifeNMDA=20 ms, MgConc=1.0")
print(f"Secretion increase (Simple DAP): {pct_inc:.1f}%")

gain_noDAP   = np.polyfit(PSP_RATES, results_noDAP,   1)[0]
gain_simpDAP = np.polyfit(PSP_RATES, results_simpDAP, 1)[0]
print(f"F-I gain (no DAP):    {gain_noDAP:.4f} Hz/Hz")
print(f"F-I gain (simple DAP):{gain_simpDAP:.4f} Hz/Hz")
print(f"\nFigure 1 →figure1_isi_fits.png")
print(f"Figure 2 →figure2_signal_processing.png")
print("Done.")
