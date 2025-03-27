import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# === Load data from CSVs ===
data_df = pd.read_csv('data.csv', header=0, delimiter='\t')
params_df = pd.read_csv('parameters.csv', header=0)

x = data_df['x'].values
y = data_df['y'].values
p = params_df['parameter'].values
p_lb = params_df['lower_bound'].values
p_ub = params_df['upper_bound'].values
labels = params_df['label'].values

num_peaks = (len(p) - 2) // 5

# === Define Pseudo-Voigt function ===
def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    sigma_g = sigma / np.sqrt(2 * np.log(2))
    lorentz = (gamma**2) / ((x - center)**2 + gamma**2)
    gaussian = np.exp(-(x - center)**2 / (2 * sigma_g**2))
    return amplitude * (eta * lorentz + (1 - eta) * gaussian)

# === Sum SFG signals ===
def SFG_signal_sum(params, frequency):
    y_out = np.zeros_like(frequency, dtype=np.float64)

    for i in range(num_peaks):
        idx = 2 + i * 5
        amplitude, center, sigma, gamma, eta = params[idx:idx+5]
        y_out += pseudo_voigt(frequency, amplitude, center, sigma, gamma, eta)

    y_out += params[0]  # Offset
    y_out += params[1]  # Nonresonant background

    return y_out

# === Residual function ===
def residual_func(params, x, y):
    return SFG_signal_sum(params, x) - y

# === Optimization ===
result = least_squares(residual_func, p, args=(x, y), bounds=(p_lb, p_ub),
                       max_nfev=10000, ftol=1e-9)
p_optimized = result.x

# === Compute AIC and BIC ===
n = len(y)
k = len(p_optimized)
rss = np.sum(result.fun**2)

aic = 2 * k + n * np.log(rss / n)
bic = k * np.log(n) + n * np.log(rss / n)

print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")

# === Parameter uncertainties ===
covariance = np.linalg.pinv(result.jac.T @ result.jac)
parameter_errors = np.sqrt(np.diag(covariance))

print("Optimized parameters with uncertainties:")
for label, param, err in zip(labels, p_optimized, parameter_errors):
    print(f"{label}: {param:.6f} ± {err:.6f}")

# === Final fit and decomposed peaks ===
y_predicted = SFG_signal_sum(p_optimized, x)
peaks = np.zeros((len(x), num_peaks))

for i in range(num_peaks):
    idx = 2 + i * 5
    amplitude, center, sigma, gamma, eta = p_optimized[idx:idx+5]
    peaks[:, i] = pseudo_voigt(x, amplitude, center, sigma, gamma, eta)

# === Plot results ===
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='#d23838', label='Data', s=20)
plt.plot(x, y_predicted, color='#03045e', label='Fit', linewidth=3, alpha=0.8)

colors = plt.cm.tab10(np.linspace(0, 1, num_peaks))
for i in range(num_peaks):
    plt.plot(x, peaks[:, i], '--', color=colors[i], label=f'Peak {i+1}', linewidth=1)

plt.legend()
plt.xlabel('cm⁻¹', fontsize=16)
plt.ylabel('Abs', fontsize=16)
plt.title('Optimized Pseudo-Voigt Fit with Decomposed Peaks', fontsize=16)
plt.tight_layout()

plt.savefig('optimized_fit.png', dpi=300)
plt.show()

# === Save optimized results ===
output_df = pd.DataFrame({'x': x, 'y': y, 'y_predicted': y_predicted})
for i in range(num_peaks):
    output_df[f'peak_{i+1}'] = peaks[:, i]

output_df.to_csv('optimized_fit_output.csv', index=False)

print('Saved optimized fit results to optimized_fit_output.csv')
