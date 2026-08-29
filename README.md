# 📈 Quantitative Multi-Strategy Portfolio Optimizer
### Markowitz Mean-Variance Efficient Frontier | Conditional Value-at-Risk (Min CVaR) | Ledoit-Wolf Covariance | SciPy SLSQP

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Quant Finance](https://img.shields.io/badge/Portfolio-Convex%20Optimization-success.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A quantitative asset allocation platform implementing **Markowitz Mean-Variance Efficient Frontier**, **Black-Litterman Bayesian allocation**, and convex **Conditional Value-at-Risk (CVaR)** optimization across multi-asset equity universes.

---

## 📌 Optimization Objectives & Risk Formulations

### 1. Markowitz Maximum Sharpe Program:
$$\max_{\mathbf{w}} \quad \frac{\mathbf{w}^T \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}} \quad \text{Subject to: } \sum w_i = 1, \; w_i \ge 0$$

### 2. Tail Risk Minimum CVaR (95% Expected Shortfall):
$$\min_{\mathbf{w}, \alpha} \quad \alpha + \frac{1}{(1-\beta)S} \sum_{s=1}^S \max\left(0, \; -\mathbf{w}^T \mathbf{r}_s - \alpha\right)$$

---

## 📊 Benchmark Optimization Performance
* **Asset Universe:** 5 Years of Daily Price Histories (1,259 trading days) across 6 institutional assets (`AAPL`, `MSFT`, `JPM`, `XOM`, `JNJ`, `TLT`).
* **Markowitz Max Sharpe Portfolio:**
  * Expected Return: **19.36%**
  * Annual Volatility: **16.83%**
  * **Sharpe Ratio: 0.91**
* **Minimum CVaR Portfolio:** Reduces tail risk volatility to **8.20%**.

---

## 📂 Repository Structure
```
Multi-Strategy-Quantitative-Portfolio-Optimizer/
├── src/
│   ├── portfolio_optimizer.py      # Markowitz, Black-Litterman & CVaR solver
│   └── data_loader.py              # Equity historical price loader
├── Quantitative_Portfolio_Optimization.ipynb # Interactive evaluation notebook
├── run_pipeline.py                 # Pipeline execution script
├── test_portfolio_optimizer.py     # Unit testing suite (4/4 passing)
└── requirements.txt                # Production dependencies
```

---

## 🚀 Quickstart & Reproducibility
```bash
git clone https://github.com/SurajChouhan14/Multi-Strategy-Quantitative-Portfolio-Optimizer.git
cd Multi-Strategy-Quantitative-Portfolio-Optimizer
pip install -r requirements.txt
python run_pipeline.py
python -m unittest test_portfolio_optimizer.py
```
