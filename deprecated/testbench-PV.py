import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import wofz

# === Load data from CSVs ===
data_df = pd.read_csv('data.csv', header=0, delimiter='\t')
params_df = pd.read_csv('parameters_test.csv', header=0)

x = data_df['x'].values
y = data_df['y'].values
p = params_df['parameter'].values
p_lb = params_df['lower_bound'].values
p_ub = params_df['upper_bound'].values

num_peaks = (len(p) - 2) // 5  # Adjusted for eta

# === Troubleshooting parameter bounds ===
for i, (val, lb, ub) in enumerate(zip(p, p_lb, p_ub)):
    if not (lb <= val <= ub):
        print(f"❌ Parameter {i}: {val} is outside bounds [{lb}, {ub}]")

# === Define Pseudo-Voigt function ===
def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    sigma_g = sigma / np.sqrt(2 * np.log(2))
    lorentz = (gamma**2) / ((x - center)**2 + gamma**2)
    gaussian = np.exp(-(x - center)**2 / (2 * sigma_g**2))
    return amplitude * (eta * lorentz + (1 - eta) * gaussian)

# === Sum SFG signals using Pseudo-Voigt ===
def SFG_signal_sum(params, frequency):
    y_out = np.zeros_like(frequency, dtype=np.float64)

    for i in range(num_peaks):
        idx = 2 + i * 5
        amplitude = params[idx]
        center = params[idx + 1]
        sigma = params[idx + 2]
        gamma = params[idx + 3]
        eta = params[idx + 4]
        y_out += pseudo_voigt(frequency, amplitude, center, sigma, gamma, eta)

    y_out += params[0]  # background from green light scattering
    y_out += params[1]  # non-resonant background

    return y_out

# === Relative residual calculation ===
def relative_residual(y_pred, y_true):
    residual = y_pred - y_true
    return np.sqrt(np.sum(residual ** 2) / np.sum(y_true ** 2)), residual

# === Residual function for optimization ===
def residual_func(params, x, y):
    return SFG_signal_sum(params, x) - y

# === Optimization Loop ===
result = least_squares(residual_func, p, args=(x, y), bounds=(p_lb, p_ub),
                       max_nfev=500000, ftol=1e-9)
p = result.x

# Calculate uncertainties
jacobian = result.jac
covariance = np.linalg.pinv(jacobian.T @ jacobian)
parameter_errors = np.sqrt(np.diag(covariance))
correlation_matrix = covariance / np.outer(parameter_errors, parameter_errors)

# Print uncertainties and correlations
print("Parameter uncertainties:")
for i, err in enumerate(parameter_errors):
    label = params_df['label'][i]
    print(f"{label}: {p[i]:.2f} ± {err:.2f}")

print("Correlation matrix:")
print(correlation_matrix)

# === Final fit and individual peaks ===
y_predicted = SFG_signal_sum(p, x)
peaks = np.zeros((len(x), num_peaks))

for i in range(num_peaks):
    idx = 2 + i * 5
    amplitude = p[idx]
    center = p[idx + 1]
    sigma = p[idx + 2]
    gamma = p[idx + 3]
    eta = p[idx + 4]
    peaks[:, i] = pseudo_voigt(x, amplitude, center, sigma, gamma, eta)

# === Final Plot ===
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data', s=20)
plt.plot(x, y_predicted, color='blue', label='Fit', linewidth=2)

colors = plt.cm.viridis(np.linspace(0, 1, num_peaks))
for i in range(num_peaks):
    plt.plot(x, peaks[:, i], '--', label=f'Peak {i+1}', color=colors[i])

plt.legend()
plt.xlabel('frequency (cm⁻¹)', fontsize=16)
plt.ylabel('signal intensity', fontsize=16)
plt.title('Pseudo-Voigt Fit of SFG Data', fontsize=16)
plt.tight_layout()

plt.savefig('fit_plot_pseudo_voigt.png', dpi=300)
plt.show()

# === Save Results ===
output_df = pd.DataFrame({
    'x': x,
    'y': y,
    'y_predicted': y_predicted
})

for i in range(num_peaks):
    output_df[f'peak_{i+1}'] = peaks[:, i]

output_df.to_csv('fit_output_pseudo_voigt.csv', index=False)
print('Saved fit results to fit_output_pseudo_voigt.csv')
