import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.signal import find_peaks


def compute_SpectralExponent(XX, YY):
    X = np.log10(XX)
    Y = np.log10(YY)

    XXi = np.logspace(X[0], X[-1], len(X) * 4)
    Xi = np.log10(XXi)
    Yi = np.interp(
        Xi,
        X,
        Y,
    )
    YYi = 10 ** (Yi)

    # STEP 1, FIT 1st LINE
    linfit0 = linregress(Xi, Yi)  # alternative = 'less'???
    slope0, intercept0 = linfit0[0], linfit0[1]
    YPred0 = Xi * slope0 + intercept0
    YRes0 = Yi - YPred0
    sigma0 = np.sqrt(np.mean(YRes0**2))

    # FIND DEVIANT RESIDUALS
    threRes = 0
    boolYdev = YRes0 > threRes
    idxs_dev = np.argwhere(np.diff(boolYdev, prepend=False, append=False)).reshape(
        -1, 2
    )
    idxs_peaks = find_peaks(Yi)[0]
    threResPks = np.median(np.absolute(YRes0 - np.median(YRes0))) * 1

    idxs_peaks = idxs_peaks[YRes0[idxs_peaks] > threResPks]
    boolYdev2 = np.zeros_like(boolYdev)
    for idx_start, idx_end in idxs_dev:
        range_dev = np.arange(idx_start, idx_end)
        boolYdev2[range_dev] = np.any(np.intersect1d(idxs_peaks, range_dev))

    # STEP 2, FIT 2nd LINE WITHOUT DEVIANT RESIDUALS
    linfit = linregress(Xi[~boolYdev2], Yi[~boolYdev2])
    slope, intercept = linfit[0], linfit[1]
    YPred = Xi * slope + intercept
    YRes = Yi[~boolYdev2] - YPred[~boolYdev2]
    sigma = np.sqrt(np.mean(YRes**2))

    stats = dict(
        slope=slope,
        slope0=slope0,
        inter=intercept,
        inter0=intercept0,
        r=linfit[2],
        r0=linfit0[2],
        sigma=sigma,
        sigma0=sigma0,
        threResPks=threResPks,
    )
    vectors = dict(
        f_resamp=XXi,
        observed=YYi,
        predicted=YPred,
        resid=YRes,
        resid0=YRes0,
        reject=boolYdev2,
    )

    return slope, intercept, stats, vectors


def spectral_exponent_per_channel(freqs, psds, fmin=15.0, fmax=50.0):
    mask = (freqs >= fmin) & (freqs <= fmax)
    XX = freqs[mask]

    n_ch = psds.shape[0]
    slopes = np.full(n_ch, np.nan, dtype=float)
    intercepts = np.full(n_ch, np.nan, dtype=float)
    stats_list = [None] * n_ch

    for ch in range(n_ch):
        YY = psds[ch, mask]
        if np.any(~np.isfinite(YY)) or np.max(YY) <= 0 or np.sum(YY > 0) < 5:
            continue
        slope, inter, stats, vectors = compute_SpectralExponent(XX, YY)
        slopes[ch] = slope
        intercepts[ch] = inter
        stats_list[ch] = stats

    return pd.DataFrame(
        {
            "slope": slopes,
            "intercept": intercepts,
        }
    )
