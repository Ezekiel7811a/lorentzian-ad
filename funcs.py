import numpy as np


def log_knee_power_law(x, a, k, c):
    return a - np.log10(k + x**c)


def log_free_roll_lorentzian(x, a, fc, c):
    return a - np.log10(1 + (x / fc) ** (-c))


def log_power_law(x, a, c):
    return a - np.log10(x**c)


def power_law(x, a, b):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    m = np.isfinite(x) & (x > 0)
    out[m] = a / (x[m] ** b)
    return out


def lorentzian(x, a, fc):
    return a / (1 + (x / fc) ** 4)


def free_roll_lorentzian(x, a, fc, exp):
    return a / (1 + (x / fc) ** (-exp))


def gaussian_function(xs, *params):
    ys = np.zeros_like(xs)
    if np.isnan(params[0]):
        return ys
    for ii in range(0, len(params), 3):
        ctr, hgt, wid = params[ii : ii + 3]
        ys = ys + hgt * np.exp(-((xs - ctr) ** 2) / (2 * wid**2))

    return ys


def aic_mse(mse, k, n):
    return n * np.log(mse) + 2 * k


def mean_squared_error(residuals: np.array):
    return np.mean(residuals**2)
