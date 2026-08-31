# Quantitative Multi-Strategy Portfolio Optimizer
> **Convex Portfolio Optimization Suite implementing Markowitz Mean-Variance Efficient Frontier, Rockafellar-Uryasev (2000) Convex CVaR Linear Programming in SciPy HiGHS, Black-Litterman Bayesian Asset Allocation, and Equal Risk Parity**  
> *Quantitative Finance · Markowitz Modern Portfolio Theory · Rockafellar-Uryasev Convex CVaR · Black-Litterman · SciPy HiGHS · Asset Allocation*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/SurajChouhan14/Multi-Strategy-Quantitative-Portfolio-Optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/SurajChouhan14/Multi-Strategy-Quantitative-Portfolio-Optimizer/actions)
[![Benchmark](https://img.shields.io/badge/benchmark-1%2C259%20Daily%20Bars-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-6%20passed-brightgreen.svg)]()

---

## 🎯 Executive Overview & Mathematical Architecture
Quantitative asset management relies on robust portfolio construction models balancing risk, return, and tail-loss downside risk. This repository formulates and solves **core quantitative asset allocation paradigms**:

### 1. Markowitz Mean-Variance Program (Max Sharpe Ratio)
$$\max_{\mathbf{w}} \quad rac{\mathbf{w}^T oldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^T oldsymbol{\Sigma} \mathbf{w}}} \quad 	ext{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad w_i \ge 0$$

### 2. Rockafellar-Uryasev (2000) Convex CVaR Program (HiGHS LP)
Convex Linear Programming formulation minimizing tail loss (95% Conditional Value-at-Risk / Expected Shortfall):
$$\min_{\mathbf{w}, \zeta, \mathbf{u}} \quad \zeta + rac{1}{(1-lpha) T} \sum_{t=1}^T u_t \quad 	ext{s.t.} \quad u_t \ge -\mathbf{w}^T \mathbf{r}_t - \zeta, \quad u_t \ge 0, \quad \sum_{i=1}^N w_i = 1, \quad w_i \ge 0$$

### 3. Black-Litterman Bayesian Asset Allocation
Blends CAPM market equilibrium prior returns $oldsymbol{\Pi} = \delta oldsymbol{\Sigma} \mathbf{w}_{	ext{mkt}}$ with subjective investor views $(P, Q, oldsymbol{\Omega})$ into a Bayesian posterior distribution:
$$oldsymbol{\mu}_{	ext{BL}} = \left[ (	au oldsymbol{\Sigma})^{-1} + P^T oldsymbol{\Omega}^{-1} P ight]^{-1} \left[ (	au oldsymbol{\Sigma})^{-1} oldsymbol{\Pi} + P^T oldsymbol{\Omega}^{-1} Q ight]$$
$$oldsymbol{\Sigma}_{	ext{BL}} = oldsymbol{\Sigma} + \left[ (	au oldsymbol{\Sigma})^{-1} + P^T oldsymbol{\Omega}^{-1} P ight]^{-1}$$
Solved via Mean-Variance Quadratic Utility: $\min_{\mathbf{w}} rac{\delta}{2} \mathbf{w}^T oldsymbol{\Sigma}_{	ext{BL}} \mathbf{w} - \mathbf{w}^T oldsymbol{\mu}_{	ext{BL}}$ s.t. $\sum w_i = 1, w_i \ge 0$.

### 4. Covariance Regularization
Covariance matrix estimated using fixed shrinkage ($\delta = 0.20$ intensity toward diagonal variance target $\mathbf{S}_{	ext{prior}} = 	ext{diag}(oldsymbol{\Sigma}_{	ext{sample}})$) to guarantee positive-definiteness.

```
  ┌────────────────────────────────────────────────────────┐
  │ Institutional Universe (AAPL, MSFT, JPM, XOM, JNJ, TLT) │
  │ • 1,259 Daily Trading Bars (5-Year History)             │
  │ • Shrinkage Covariance Matrix (delta = 0.20 Target)     │
  └───────────────────────────┬────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
  ┌──────────────┐    ┌──────────────┐     ┌──────────────┐
  │  Markowitz   │    │  R-U Convex  │     │    Black-    │
  │  Max Sharpe  │    │   CVaR LP    │     │   Litterman  │
  │ (SLSQP QP)   │    │  (HiGHS LP)  │     │   Bayesian   │
  └──────┬───────┘    └──────┬───────┘     └──────┬───────┘
         │                   │                    │
         ▼                   ▼                    ▼
  • Sharpe: 0.91      • Vol: 8.20%         • Bayesian
  • Ret: 19.36%       • Min Tail Loss      • View Blending
  • Vol: 16.83%       • 95% CVaR: 16.17%   • Closed Form
```

---

## 📊 Benchmark Execution & Validation Report

### Multi-Asset Universe (1,259 Trading Days, 6 Institutional Equities)

| Strategy Paradigm | Measured Metrics | Formulation & Solver | Operational Definition |
|---|:---:|:---:|---|
| **Markowitz Max Sharpe** | **$0.91	ext{ Sharpe Ratio}$**<br>($19.36\%	ext{ Return}$, $16.83\%	ext{ Vol}$) | Non-Linear Program (SLSQP) | Maximizes excess return per unit of annualized total portfolio volatility |
| **Rockafellar-Uryasev Min CVaR** | **$8.20\%	ext{ Annual Volatility}$**<br>($16.17\%	ext{ Annual 95% CVaR}$) | Exact Convex LP (SciPy HiGHS) | Global minimum tail risk downside loss at 95% confidence level |
| **Black-Litterman Bayesian** | **$3.41\%	ext{ Return}$, $10.34\%	ext{ Vol}$**<br>(AAPL: $27.8\%$, MSFT: $14.4\%$, TLT: $36.5\%$) | Bayesian Quadratic Utility | Smooth Bayesian weight tilts from benchmark ($P, Q, oldsymbol{\Omega}$) |
| **Equal Risk Parity (ERC)** | **$12.35\%	ext{ Return}$, $12.63\%	ext{ Vol}$**<br>(Each asset $pprox 16.7\%$ risk share) | Sequential Quadratic Program | Equalizes marginal risk contributions across all 6 asset classes |

---

## 📁 Repository Structure

```text
Multi-Strategy-Quantitative-Portfolio-Optimizer/
├── .github/
│   └── workflows/
│       └── ci.yml                      # Automated CI test & benchmark workflow
├── .gitignore                          # Git exclusions (pycache, results, logs)
├── Portfolio_Optimization.ipynb        # Interactive Jupyter notebook
├── README.md                           # Documentation & mathematical formulations
├── data/
│   └── portfolio_asset_prices.csv      # 1,259 daily price records across 6 assets
├── requirements.txt                    # Dependencies (scipy, pandas, numpy)
├── run_pipeline.py                     # Pipeline execution runner
├── src/
│   ├── data_loader.py                  # Price data ingestion & shrinkage covariance
│   └── portfolio_optimizer.py          # Markowitz, R-U CVaR LP & Black-Litterman solvers
└── test_portfolio_optimizer.py         # 6 hard unit & Bayesian invariant tests
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/SurajChouhan14/Multi-Strategy-Quantitative-Portfolio-Optimizer.git
cd Multi-Strategy-Quantitative-Portfolio-Optimizer
pip install -r requirements.txt
```

### 2. Run Optimization Pipeline
```bash
python run_pipeline.py
```

### 3. Run Test Suite
```bash
python test_portfolio_optimizer.py
```
