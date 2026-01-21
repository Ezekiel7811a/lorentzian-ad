import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from funcs import *


def simple_ap_fit(
    x,
    y,
    method=log_knee_power_law,
    ap_bounds=([-np.inf, 0, -np.inf], [np.inf, 60, np.inf]),
    **kwargs,
):
    """
    Comes from fooof https://fooof-tools.github.io/fooof/
    """
    off_guess = y[0]
    knee_guess = 30
    exp_guess = np.abs(y[-1] - y[0]) / (x[-1] - x[0])
    guess = [off_guess, knee_guess, exp_guess]
    ap_params, ap_pcov = curve_fit(
        method, x, y, p0=guess, bounds=ap_bounds, maxfev=10000
    )

    if kwargs.get("return_ap_pcov", False):
        return ap_params, ap_pcov
    return ap_params


def drop_peak_cf(guess, bw_std_edge=1, freq_range=(7, 45)):
    """
    comes from fooof https://fooof-tools.github.io/fooof/
    """
    cf_params = [item[0] for item in guess]
    bw_params = [item[2] * bw_std_edge for item in guess]
    out_of_bounds = np.logical_or(
        np.array(cf_params) < freq_range[0], np.array(cf_params) > freq_range[1]
    )
    keep_peak = (
        (np.abs(np.subtract(cf_params, freq_range[0])) > bw_params)
        & (np.abs(np.subtract(cf_params, freq_range[1])) > bw_params)
        & ~out_of_bounds
    )
    guess = np.array([gu for (gu, keep) in zip(guess, keep_peak) if keep])

    return guess


def drop_peak_overlap(guess, overlap_thresh=0.75):
    """
    comes from fooof https://fooof-tools.github.io/fooof/
    """
    guess = sorted(guess, key=lambda x: float(x[0]))
    bounds = [
        [peak[0] - peak[2] * overlap_thresh, peak[0] + peak[2] * overlap_thresh]
        for peak in guess
    ]
    drop_inds = []
    for ind, b_0 in enumerate(bounds[:-1]):
        b_1 = bounds[ind + 1]
        if b_0[1] > b_1[0]:
            drop_inds.append(
                [ind, ind + 1][np.argmin([guess[ind][1], guess[ind + 1][1]])]
            )
    keep_peak = [not ind in drop_inds for ind in range(len(guess))]
    guess = np.array([gu for (gu, keep) in zip(guess, keep_peak) if keep])

    return guess


def fit_peak_guess(
    guess,
    gauss_std_limits,
    x,
    y,
    cf_bound=1.5,
    freq_range=(3, 45),
    gauss_bounds=([7, 0, 0.5], [50, np.inf, 8]),
):
    """
    comes from fooof https://fooof-tools.github.io/fooof/
    """
    freq_range = (gauss_bounds[0][0], gauss_bounds[1][0])
    lo_bound = [
        [peak[0] - 2 * cf_bound * peak[2], 0, gauss_std_limits[0]] for peak in guess
    ]
    hi_bound = [
        [peak[0] + 2 * cf_bound * peak[2], np.inf, gauss_std_limits[1]]
        for peak in guess
    ]
    lo_bound = [
        bound if bound[0] > freq_range[0] else [freq_range[0], *bound[1:]]
        for bound in lo_bound
    ]
    hi_bound = [
        bound if bound[0] < freq_range[1] else [freq_range[1], *bound[1:]]
        for bound in hi_bound
    ]
    gaus_param_bounds = (
        tuple([item for sublist in lo_bound for item in sublist]),
        tuple([item for sublist in hi_bound for item in sublist]),
    )
    guess = np.ndarray.flatten(guess)
    gaussian_params, _ = curve_fit(
        gaussian_function, x, y, p0=guess, maxfev=5000, bounds=gaus_param_bounds
    )
    return np.array(gaussian_params).reshape(-1, 3)


def fit_peaks(
    max_n_peaks,
    x,
    y,
    min_peak_height=0.0,
    threshold=2,
    peak_width_limits=(2, 8),
    gauss_bounds=([4, 0, 0], [45, np.inf, np.inf]),
):
    """
    Comes from fooof https://fooof-tools.github.io/fooof/
    """
    inside_gauss_bounds = np.logical_and(
        x >= gauss_bounds[0][0], x <= gauss_bounds[1][0]
    )
    residuals = y.copy()
    freq_res = x[1] - x[0]
    gauss_std_limits = tuple([bwl / 2 for bwl in peak_width_limits])
    guess = np.empty([0, 3])
    threshold = threshold * np.std(y[inside_gauss_bounds])

    # Get indices where bounds condition is True
    valid_indices = np.where(inside_gauss_bounds)[0]

    while len(guess) < max_n_peaks:
        # Find max within bounds
        y_bounded = y[inside_gauss_bounds]
        if len(y_bounded) == 0:
            break

        max_ind_relative = np.argmax(y_bounded)
        max_ind = valid_indices[max_ind_relative]  # Map back to original indices
        max_height = y[max_ind]

        if max_height <= threshold:
            break

        guess_freq = x[max_ind]
        guess_height = max_height

        if not guess_height > min_peak_height:
            break

        half_height = guess_height / 2
        le_ind = next(
            (val for val in range(max_ind - 1, -1, -1) if y[val] <= half_height), None
        )
        ri_ind = next(
            (val for val in range(max_ind + 1, len(y), 1) if y[val] <= half_height),
            None,
        )

        try:
            short_side = min(
                [abs(ind - max_ind) for ind in [le_ind, ri_ind] if ind is not None]
            )
            fwhm = short_side * 2 * freq_res
            guess_std = fwhm / (2 * np.sqrt(2 * np.log(2)))
        except ValueError:
            guess_std = np.mean(peak_width_limits)

        if guess_std < gauss_std_limits[0]:
            guess_std = gauss_std_limits[0]
        if guess_std > gauss_std_limits[1]:
            guess_std = gauss_std_limits[1]

        guess = np.vstack((guess, [guess_freq, guess_height, guess_std]))

        peak_gauss = gaussian_function(x, guess_freq, guess_height, guess_std)
        y = y - peak_gauss

    guess = drop_peak_cf(guess, freq_range=(gauss_bounds[0][0], gauss_bounds[1][0]))
    guess = drop_peak_overlap(guess)

    if len(guess) > 0:
        gaussian_params = fit_peak_guess(
            guess, gauss_std_limits, x, residuals, gauss_bounds=gauss_bounds
        )
        gaussian_params = gaussian_params[gaussian_params[:, 0].argsort()]
    else:
        gaussian_params = np.empty([0, 3])

    return gaussian_params


def correct_curve(x, residuals, n_endpoints=4):
    def affine_func(x, a, b):
        return a * x + b

    idx = np.r_[0:n_endpoints, -n_endpoints:]
    x_end = x[idx]
    res_end = residuals[idx]

    coeffs = np.polyfit(x_end, res_end, 1)
    a, b = coeffs

    linear_corrected_residuals = residuals - affine_func(x, a, b)
    return linear_corrected_residuals


def low_cap(x, residuals):
    residuals[residuals < 0] = 0
    return residuals


def robust_ap_fit(x, y, n_endpoints=4, estimate_curve=True):
    ap_bounds = ([-np.inf, 7, -np.inf], [np.inf, 45, np.inf])
    popt = simple_ap_fit(x, y, method=log_free_roll_lorentzian, ap_bounds=ap_bounds)
    initial_fit = log_free_roll_lorentzian(x, *popt)
    residuals = y - initial_fit

    if estimate_curve:
        linear_corrected_residuals = correct_curve(
            x, residuals, n_endpoints=n_endpoints
        )
    else:
        linear_corrected_residuals = low_cap(x, residuals)

    ignore_mask = np.percentile(linear_corrected_residuals, 30)
    threshold_mask = linear_corrected_residuals <= ignore_mask

    ap_params, _ = curve_fit(
        log_free_roll_lorentzian,
        x[threshold_mask],
        y[threshold_mask],
        p0=popt,
        bounds=ap_bounds,
        maxfev=5000,
    )
    return ap_params


def weighted_affine_fit(
    x, y, idx=None, min_seg=2, method="huber", c=1.345, max_iter=25, tol=1e-6
):
    if idx is None:
        idx = len(x)
    x = np.asarray(x[:idx])
    y = np.asarray(y[:idx])
    n = len(x)
    if n < max(min_seg, 2):
        raise ValueError("Not enough points for a weighted affine fit")

    X = np.column_stack((np.ones(n), x))

    a, b = np.linalg.lstsq(X, y, rcond=None)[0]

    def mad(z):
        return 1.4826 * np.median(np.abs(z - np.median(z))) + 1e-12

    for _ in range(max_iter):
        r = y - (a + b * x)
        s = mad(r)
        u = r / (s * c)

        if method == "huber":
            w = np.where(np.abs(u) <= 1, 1.0, 1.0 / np.abs(u))
        elif method == "tukey":
            mask = np.abs(u) < 1
            w = np.zeros_like(u)
            w[mask] = (1 - u[mask] ** 2) ** 2
        else:
            raise ValueError("method must be 'huber' or 'tukey'")
        sw = np.sqrt(w)
        a_new, b_new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]

        if np.hypot(a_new - a, b_new - b) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    # Outputs
    y_hat = a + b * x
    sse = np.sum(w * (y - (a + b * x)) ** 2)
    dof = max(n - 2, 1)
    mse = sse / dof
    return {
        "intercept": a,
        "slope": b,
        "weights": w,
        "y_hat": y_hat,
        "sse": sse,
        "mse": mse,
        "n_used": n,
    }


def robust_power_law_fit(x, r_, verbose=False):
    win = 5
    poly = 3

    x = np.asarray(x)
    r_ = np.asarray(r_)

    valid = np.isfinite(x) & np.isfinite(r_) & (x > 0) & (r_ > 0)
    x = x[valid]
    r_ = r_[valid]

    if len(r_) < 2:
        if verbose:
            print("Not enough valid points for fitting")
        return np.array([np.nan, np.nan])

    r = np.log10(r_)

    if len(r) < win:
        if verbose:
            print(f"Not enough points ({len(r)}) for smoothing")
        d1 = np.gradient(r)
    else:
        rsm = savgol_filter(r, window_length=(win | 1), polyorder=poly)
        d1 = np.gradient(rsm)

    d10 = d1[0]
    if not np.isfinite(d10) or d10 == 0:
        if verbose:
            print("Bad derivative normalization (d1[0] is 0 or non-finite)")
        return np.array([np.nan, np.nan])

    d1_norm = d1 / d10

    idxs = np.argwhere(d1_norm < 0.1).flatten()
    idx = int(idxs[0]) if len(idxs) else (len(r) - 1)

    try:
        if verbose:
            print(f"Using {idx} points for robust power law fit")
        res = weighted_affine_fit(np.log10(x), r, idx=idx, method="tukey")
    except ValueError:
        return np.array([np.nan, np.nan])

    return np.array([res["intercept"], res["slope"]])


def sequential_fit(
    x,
    y,
    physiological_mask=None,
    max_n_peaks=3,
    **kwargs,
):
    FIT_PEAKS = {
        "threshold": kwargs.get("threshold", 1),
    }

    y = y.copy()[x > 0]
    x = x.copy()[x > 0]
    y_log = np.log10(y)
    if physiological_mask is None:
        physiological_mask = (x >= 4) & (x <= 45)
    pl_mask = x <= x[physiological_mask][-1]
    l_params = robust_ap_fit(
        x=x[physiological_mask],
        y=y_log[physiological_mask],
        estimate_curve=False,
    )

    residuals = y - 10 ** (log_free_roll_lorentzian(x, *l_params))
    log_residuals = y_log - log_free_roll_lorentzian(x, *l_params)
    try:
        peak_params = fit_peaks(
            max_n_peaks,
            x,
            log_residuals,
            gauss_bounds=([6, 0, 0], [45, np.inf, 8]),
            **FIT_PEAKS,
        )
    except:
        peak_params = np.full((1, 3), np.nan)

    pl_params = robust_power_law_fit(
        x=x[pl_mask],
        r_=residuals[pl_mask] - gaussian_function(x[pl_mask], *peak_params.flatten()),
    )
    return np.concatenate((pl_params, l_params, peak_params.flatten()))


def adaptable_func(x, *params, add=None):
    params_ = list(params)
    ap_params = params_[2:5]
    peak_params = params_[5:]

    if not np.isnan(params_[0]):
        ap_params[0] = 10 ** ap_params[0]
        pl_params = params_[:2]
        pl_params[0] = 10 ** pl_params[0]
        pl_params[1] = np.abs(pl_params[1])
        return (
            np.log10(power_law(x, *pl_params) + free_roll_lorentzian(x, *ap_params))
            + gaussian_function(x, *peak_params)
            + (add if add is not None else 0)
        )
    else:
        return (
            log_free_roll_lorentzian(x, *ap_params)
            + gaussian_function(x, *peak_params)
            + (add if add is not None else 0)
        )
