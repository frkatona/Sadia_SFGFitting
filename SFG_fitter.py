import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# === Load data from CSVs ===

data_df = pd.read_csv('data.csv', header=0, delimiter='\t')  # Assumes first row as column headers
params_df = pd.read_csv('parameters.csv', header=0) # consolidated params and bounds: param, LB, UB

x = data_df['x'].values
y = data_df['y'].values
p = params_df['parameter'].values
p_lb = params_df['lower_bound'].values
p_ub = params_df['upper_bound'].values

num_peaks = (len(p) - 2) // 4

# === Troubleshooting ===

for i, (val, lb, ub) in enumerate(zip(p, p_lb, p_ub)):
    if not (lb <= val <= ub):
        print(f"❌ Parameter {i}: {val} is outside bounds [{lb}, {ub}]")

#! note:
# I edited a line of the original parameters/bounds: '-0.5, -1, 2' was originally '-0.5, 1, 2' and idk if this is the appropriate change


# === Function Definitions ===

def SFG_Lorentzian(A, wr, w, Tau):
    return A / (w - wr + 1j * Tau)


def SFG_Lorentzian_Gaussian(A, wr, w, Tau, sigma):
    if sigma <= 10:
        return SFG_Lorentzian(A, wr, w, Tau)
    
    ki = np.zeros_like(w, dtype=np.complex128)
    num_per_sigma = 30
    max_sigma = 3
    norm = 0

    for n in range(-num_per_sigma * max_sigma, num_per_sigma * max_sigma + 1):
        weight = np.exp(-n * n / (num_per_sigma * num_per_sigma))
        shifted_wr = wr - n / num_per_sigma * sigma
        ki += SFG_Lorentzian(A, shifted_wr, w, Tau) * weight
        norm += weight

    return ki / norm


def SFG_signal_sum(params, frequency):
    ki = np.zeros_like(frequency, dtype=np.complex128)
    y_out = np.zeros_like(frequency)

    num_peaks = (len(params) - 2) // 4
    for i in range(num_peaks):
        idx = 2 + i * 4
        A = params[idx]
        wr = params[idx + 1]
        Tau = params[idx + 2]
        sigma = params[idx + 3]
        ki += SFG_Lorentzian_Gaussian(A, wr, frequency, Tau, sigma)

    ki += params[1]  # non-resonant background (imaginary part assumed negligible)
    y_out = np.abs(ki) ** 2
    y_out += params[0]  # background from green light scattering
    return y_out


def relative_residual(y_pred, y_true):
    residual = y_pred - y_true
    return np.sqrt(np.sum(residual ** 2) / np.sum(y_true ** 2)), residual


def residual_func(params, x, y):
    return SFG_signal_sum(params, x) - y


# === Optimization Loop ===

n_cycles = 50
residuals_list = []

for i in range(n_cycles):
    print(f'******** Starting optimization cycle #{i+1} ********')
    result = least_squares(residual_func, p, args=(x, y), bounds=(p_lb, p_ub),
                           max_nfev=500000, ftol=1e-9)
    p = result.x
    rel_error, residuals = relative_residual(SFG_signal_sum(p, x), y)
    residuals_list.append(residuals)
    print(f'Relative error = {rel_error:.6f}')

# Plotting residuals over cycles
plt.figure(figsize=(10, 6))
colors = plt.cm.plasma(np.linspace(0, 1, n_cycles))

for i in range(n_cycles):
    plt.plot(x, residuals_list[i], color=colors[i], label=f'Cycle {i+1}')

fontsize = 25
plt.xlabel('frequency (cm⁻¹)', fontsize=fontsize)
plt.ylabel('residuals', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.title('Residuals over Optimization Cycles', fontsize=fontsize)
# plt.colorbar(plt.cm.ScalarMappable(cmap='plasma'), label='Cycle')
plt.tight_layout()

# === Final fit and individual peaks ===

y_predicted = SFG_signal_sum(p, x)
peaks = np.zeros((len(x), num_peaks))

for i in range(num_peaks):
    idx = 2 + i * 4
    A = p[idx]
    wr = p[idx + 1]
    Tau = p[idx + 2]
    sigma = p[idx + 3]
    peaks[:, i] = np.abs(SFG_Lorentzian_Gaussian(A, wr, x, Tau, sigma)) ** 2

# === Plotting ===

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='#d23838', label='Data', s=30)
plt.plot(x, y_predicted, color='#03045e', label='fit', linewidth=4, alpha=0.9)

# colors = ['r', 'g', 'c', 'm', 'y', 'k']
colors = plt.cm.viridis(np.linspace(0, 1, num_peaks))

for i in range(num_peaks):
    plt.plot(x, peaks[:, i], '--', label=f'Peak {i+1}', color=colors[i % len(colors)])

plt.legend()
plt.xlabel('frequency (cm⁻¹)', fontsize=fontsize)
plt.ylabel('signal intensity', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plt.title('SFG Fit and Components')
plt.tight_layout()



plt.show()

# === Save Output ===

output_df = pd.DataFrame({
    'x': x,
    'y': y,
    'y_predicted': y_predicted
})
for i in range(num_peaks):
    output_df[f'peak_{i+1}'] = peaks[:, i]

output_df.to_csv('fit_output.csv', index=False)
print('Saved fit results to fit_output.csv')