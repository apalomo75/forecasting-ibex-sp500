# IBEX–SP500 Forecasting & Risk Analysis  
**Volatility, Forecast Stability, EGARCH Modelling and Tail-Risk Evaluation**

This project compares the IBEX-35 and the S&P500 using daily data to understand their volatility behaviour, downside risk, shock asymmetry and short-term forecastability.  
A unified pipeline is applied to both indices: cleaning, EDA, ARIMA/Prophet forecasting, EGARCH volatility estimation, VaR/ES analysis, walk-forward stability tests, rolling correlations and Power BI integration.

## Repository Structure
**data**
Raw and processed datasets.
All consolidated results for visualization and analysis are stored in `data/analysis_panel.csv`, which serves as the primary dataset for the Power BI dashboard.

**notebooks**
Notebooks 00–09 containing the full workflow: preprocessing, EDA, forecasting, volatility modelling, VaR/ES, walk-forward stability, classification, rolling correlations and the final report.

**figures**
Key plots exported automatically:
drawdowns, conditional volatility, VaR exceptions, rolling correlations, forecast-error rolling windows.

**src**
Auxiliary functions for cleaning, forecasting,
exporting datasets and reproducibility utilities.

**powerbi**
Dashboard file (.pbix) built from analysis_panel.csv

## Methods Used

- **ARIMA & Prophet forecasting** with residual diagnostics and model comparison.  
- **Walk-forward evaluation** (rolling-origin) to measure forecast stability over time.  
- **EGARCH(1,1)-t volatility modelling** to capture clustering and asymmetric shock effects.  
- **Dynamic Value-at-Risk (VaR95/99) & Expected Shortfall (ES99)** derived from conditional variance.  
- **VaR Backtesting** using Kupiec (POF) and Christoffersen (Independence) tests.  
- **Rolling correlations & drawdown analysis** to study co-movement and downside-risk intensity.  
- **Stability tests** (CUSUM, CUSUMSQ) and rolling error monitoring.  
- **Machine-learning classifiers** (Random Forest, XGBoost) for regime and risk-state prediction.

## Main Findings (Brief)

- The **S&P500** shows smoother volatility regimes, fewer VaR exceptions and lower forecast errors.  
- The **IBEX-35** displays stronger volatility clustering, deeper drawdowns and sharper negative-shock amplification.  
- Rolling correlations are **time-varying**, rising in crises and weakening during calm periods.  
- Predictability is higher for the S&P500; the IBEX is structurally more unstable and noisier.  

Detailed explanations appear in `09_final_report_and_conclusions.ipynb`.

## How to Reproduce

1. Clone the repository  
2. Create an environment and install dependencies:

pip install -r requirements.txt

3. Run notebooks 00–09 in ascending order  
4. Key figures will be saved automatically in `/figures`  
5. The Power BI dataset is exported as:

data/analysis_panel.csv

## Data Sources

- IBEX-35 — Yahoo Finance (`^IBEX`)  
- S&P500 — Yahoo Finance (`^GSPC`)  
- Daily frequency, cleaned and aligned

## Author  
**Alejandro Palomo Morales**  
Quantitative Finance • Market Risk • Data Science





