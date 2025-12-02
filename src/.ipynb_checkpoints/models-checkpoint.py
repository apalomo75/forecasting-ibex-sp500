import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera

def estimate_arima(series, max_p=5, max_q=5, d=0, criterion="aic", label="Series"):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(series, order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
                    best_model = model
            except Exception:
                continue

    print(f"{label}: Best ARIMA order {best_order} with AIC={best_aic:.2f}")
    return best_model


def validate_residuals(model, label="Series"):
    resid = model.resid

    # Ljung–Box test for autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
    lb_p = lb_test['lb_pvalue'].iloc[0]

    # Jarque–Bera test for normality
    jb_stat, jb_p = jarque_bera(resid)

    df = pd.DataFrame({
        "Model": [label],
        "LjungBox_p": [lb_p],
        "JarqueBera_p": [jb_p]
    })
    return df



def forecast_arima(model, steps=[1, 5], label="Series"):
    forecasts = {}
    for step in steps:
        pred = model.get_forecast(steps=step)
        forecasts[step] = pred.predicted_mean.iloc[-1]
    df = pd.DataFrame(list(forecasts.items()), columns=["Horizon", "Forecast"])
    df["Model"] = label
    return df
