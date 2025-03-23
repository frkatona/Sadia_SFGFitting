# SFG Spectrum Fitting with Lorentzian-Gaussian Peaks

This Python script fits experimental Sum-Frequency Generation (SFG) spectral data using a model composed of Lorentzian-Gaussian hybrid peaks. It uses non-linear least-squares optimization to extract the parameters that best represent the data, outputting a plot of the results as well as a fit report CSV.

### 📊 Example Output

![example fit](exports/figure.png)
---

## 📁 Files

- `SFG_fitter.py` — Main script that performs the fit, plots results, and exports data.
- `data.csv` — Input data file containing two tab-separated columns: `x` (frequency) and `y` (intensity).
- `parameters.csv` — Input parameter file with initial values, bounds, and labels.
- `fit_output.csv` — Output file containing original data, predicted fit, and individual peak contributions.

---

## 📊 Model Description

The model consists of:

- A constant offset (green light background)
- A nonresonant background term
- A sum of `N` peaks, each represented by a Lorentzian-Gaussian hybrid function with four parameters:
  - **A**: Amplitude
  - **wr**: Center frequency
  - **Tau**: Linewidth
  - **Sigma**: Gaussian broadening

When `Sigma <= 10`, the peak is treated as purely Lorentzian for computational efficiency.

---

## 🧾 Parameter File Format (`parameters.csv`)

The file includes four columns:

| Parameter Label         | Parameter | Lower Bound | Upper Bound |
|-------------------------|-----------|-------------|-------------|
| Offset                  | 0         | 0           | 1           |
| Nonresonant background  | 0.1       | -0.2        | 0.2         |
| Peak1 Amplitude         | 0         | -1          | 1           |
| Peak1 Center            | 2850.6    | 2845        | 2860        |
| Peak1 Tau               | 14.3      | 0           | 20          |
| Peak1 Sigma             | 0         | 0           | 1e-15       |
| Peak2 Amplitude         | -0.4      | -1          | 2           |
| ...             | ...    | ...        | ...        |
| ...             | ...    | ...        | ...        |

Note that after the first two parameters (Offset and Nonresonant background), the following values are grouped in sets of four.  Each represents the Gaussian-Lorentzian profile parameters for a distinct peak, as discussed in the model description above.

The values in the Parameter column are used as initial guesses for the optimization. The Lower Bound and Upper Bound columns define the constraints for each parameter during the fitting process.

Note that when the Parameter value falls outside of its corresponding bounds, an error is printed to the terminal describing which value to address.

Also note that the number of peaks is determined automatically within the script with the variable `num_peaks`. Conversely, the optimization cycles are a set value of 50, but this can be changed in the script if desired.  Along with parameter values and bounds, modification of this is the primary means of influencing the fit when it appears poor.

---

## 🔁 Script Workflow

1. Load experimental data (`data.csv`)
2. Load initial parameters and bounds (`parameters.csv`)
3. Fit the model using SciPy's `least_squares` over multiple optimization cycles
4. Decompose the result into individual peaks
5. Plot data, fit, and component peaks
6. Save output to `fit_output.csv`

---

## 📈 Output

The script generates:

- A plot showing:
  - Raw data (red dots)
  - Fitted signal (blue line)
  - Individual peaks (dashed lines)
- A CSV (`fit_output.csv`) with columns:
  - `x`, `y`, `y_predicted`, `peak_1`, `peak_2`, ..., `peak_N`

---

## ⚙️ Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- SciPy

Install dependencies with:

```bash
pip install numpy pandas matplotlib scipy
```

---

## ▶️ Running the Script

Open the script's root folder in a code editor like VS Code and run it.  To run from the command line, open a terminal in the script's root folder and execute:

```bash
python SFG_fitter.py
```

Make sure `data.csv` and `parameters.csv` are in the same directory.

---

## 🧠 Notes

- To use CSVs stored in a different directory, modify the file paths in the script using their absolute paths.
- The parameter bounds are enforced during optimization.
- If any initial parameter is out of bounds, a warning is printed.
- Peak components are summed in complex space and squared in magnitude to get intensity.
