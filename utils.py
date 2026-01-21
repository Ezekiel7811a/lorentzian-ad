import numpy as np
import pandas as pd


def summarize_mixedlm(res, model_name: str):
    params = res.params
    bse = res.bse
    stat = res.tvalues  # Wald z in MixedLM (naming quirk)
    pval = res.pvalues

    out = pd.DataFrame(
        {
            "model": model_name,
            "term": params.index,
            "coef": params.values,
            "se": bse.values,
            "stat": stat.values,
            "p": pval.values,
        }
    )

    ci = res.conf_int(alpha=0.05)
    ci.columns = ["ci_low", "ci_high"]
    out = out.merge(
        ci.reset_index().rename(columns={"index": "term"}), on="term", how="left"
    )
    out["n_obs"] = int(res.nobs)
    if hasattr(res.model, "n_groups") and res.model.n_groups is not None:
        n_groups = int(res.model.n_groups)
    else:
        n_groups = len(getattr(res.model, "group_labels", []))
    out["n_groups"] = n_groups

    out["llf"] = float(res.llf)
    out["aic"] = float(res.aic) if res.aic is not None else None
    out["bic"] = float(res.bic) if res.bic is not None else None

    return out


def fmt_p(p):
    if pd.isna(p):
        return ""
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"
