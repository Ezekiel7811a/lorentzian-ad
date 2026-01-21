from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from fit import sequential_fit, adaptable_func
from funcs import *


def _process_one(psd_, f, psd_er=None):
    try:
        params_ = sequential_fit(x=f, y=psd_)
        physiological_mask = (f > 0) & (f <= 45)
        residuals = np.log10(psd_) - adaptable_func(f, *params_)
        if not np.isnan(params_[0]):
            params_[0] = 10 ** params_[0]
            params_[1] = np.abs(params_[1])
        params_[2] = 10 ** params_[2]

        aic = aic_mse(
            mean_squared_error(residuals[physiological_mask]),
            len(params_) + 1,
            physiological_mask.sum() - 1,
        )

        g_p = params_[5:]
        n_peaks = len(g_p) // 3
        if len(g_p) > 9:
            g_p = g_p[0:9]
        elif len(g_p) < 9:
            g_p = np.concatenate((g_p, np.full(9 - len(g_p), np.nan)))

        return (
            aic,
            params_[0:2].tolist(),
            params_[2:5].tolist(),
            g_p.tolist(),
            n_peaks,
        )
    except Exception:
        return (
            np.nan,
            [np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan] * 9,
            np.nan,
        )


def compute_params_parallel(
    psd, f, n_jobs=-1, backend="loky", prefer=None, batch_size="auto"
):
    results = Parallel(
        n_jobs=n_jobs, backend=backend, prefer=prefer, batch_size=batch_size
    )(delayed(_process_one)(psd_, f) for psd_ in psd)
    aics_pl_l, params_pl, params_l, params_g, n_peaks_s = map(np.array, zip(*results))

    return pd.DataFrame(
        {
            "aic": aics_pl_l,
            "pl_a": params_pl[:, 0],
            "pl_exp": params_pl[:, 1],
            "l_a": params_l[:, 0],
            "l_fc": params_l[:, 1],
            "l_exp": params_l[:, 2],
            "g1_cf": params_g[:, 0],
            "g1_mu": params_g[:, 1],
            "g1_s": params_g[:, 2],
            "g2_cf": params_g[:, 3],
            "g2_mu": params_g[:, 4],
            "g2_s": params_g[:, 5],
            "g3_cf": params_g[:, 6],
            "g3_mu": params_g[:, 7],
            "g3_s": params_g[:, 8],
            "n_peaks": n_peaks_s,
        }
    )
