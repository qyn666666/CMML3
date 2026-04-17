# CMML3 ICA2

Integrate-and-fire spiking model of the depolarising afterpotential (DAP) in oxytocin neurons, implementing two DAP mechanisms and testing their effects on spike patterning and signal processing.

---

## Project Background

Magnocellular oxytocin neurons of the hypothalamus secrete oxytocin into the bloodstream. Their hormone output depends critically on spike rate and patterning, shaped by intrinsic electrophysiological properties. A proportion of oxytocin neurons exhibit a **depolarising afterpotential (DAP)** – a transient voltage rise following each spike that promotes burst firing. This project models the DAP using two mechanisms (Simple DAP and NMDA-based DAP), fits them to recorded spike data, and tests their effects on signal processing and oxytocin secretion.

---

## Repository Structure

```
├── README.md
├── HypoModPython_CMML3/
    ├── HypoModPy/
    ├── data/
    ├── SpikeModPython.py             # GUI entry point
    ├── generate_figures.py           # Main analysis script
    ├── generate_supp_figure.py       # Supplementary figure script
    ├── spikemod.py                   # HypoMod GUI extension – DAP model
    └── spikepanels.py                # HypoMod GUI extension – DAP parameter panels
```

---

## Script Overview

### `spikemod.py`

Extends the HypoMod integrate-and-fire model with two DAP implementations:

- **Simple DAP**: after each spike, adds a fixed voltage increment (`kDAP`) that decays exponentially with half-life `halflifeDAP`
- **NMDA DAP**: models an NMDA receptor-mediated current with Mg²⁺ voltage-dependent block, gated by η(V) = 1 / (1 + [Mg²⁺]·exp(−0.062·V))

This file cannot run standalone – it is loaded by the HypoMod GUI.

---

### `spikepanels.py`

Defines the GUI control panels for the DAP parameters (`kDAP`, `halflifeDAP`, `kNMDA`, `halflifeNMDA`, `MgConc`), displayed in the HypoMod interface alongside existing HAP/AHP controls.

This file cannot run standalone – it is loaded by the HypoMod GUI.

---

### `HypoModPython_CMML3/SpikeModPython.py` (GUI)

Launches the HypoMod graphical interface with the DAP extensions loaded. Allows interactive real-time tuning of all model parameters including the DAP, with live ISI histogram and firing rate display.

Requires `wxPython`:

```bash
pip install wxPython
cd HypoModPython_CMML3
python SpikeModPython.py
```

---

## `generate_figures.py`

The main analysis script. Running it will:

1. Load spike recordings from `oxydata dap cells.xlsx` and identify DAP-positive neurons using an early-peak ratio criterion (ratio > 1.5)
2. Simulate the No DAP, Simple DAP, and NMDA DAP models using confirmed best-fit parameters for neuron MAL11E
3. Plot ISI histograms and hazard functions comparing model output to recorded data
4. Compute F–I curves (output firing rate vs input PSP rate) for all three models
5. Run the oxytocin secretion model to compare plasma hormone levels
6. Simulate responses to a sinusoidally varying input signal (300 ± 150 Hz, 0.05 Hz)

**Output:** `figure1_isi_fits.png`, `figure2_signal_processing.png`

```bash
python generate_figures.py
```

---

### `generate_supp_figure.py`

Loads all neurons from `oxydata dap cells.xlsx` and plots their ISI histograms and hazard functions in a single figure, with DAP-positive neurons automatically flagged.

**Output:** `fig_supp_isi_all_neurons.png`

```bash
python generate_supp_figure.py
```

---

### Requirements

- Python 3.10+
- `numpy`, `pandas`, `matplotlib`, `openpyxl`, `scipy`
- `wxPython` (GUI only, optional)

```bash
pip install numpy pandas matplotlib openpyxl scipy
```

> `oxydata dap cells.xlsx` must be in the **same folder** as the scripts when running them.

