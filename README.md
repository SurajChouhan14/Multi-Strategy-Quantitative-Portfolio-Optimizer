# Multi-Strategy Quantitative Portfolio Optimization Engine

An institutional-grade Quantitative Portfolio Optimization system implementing **Markowitz Modern Portfolio Theory (Mean-Variance QP)**, **Hierarchical Equal Risk Parity (ERC)**, **Post-Modern Portfolio Theory (PMPT Sortino)**, and **Conditional Value-at-Risk (Min CVaR / 95% Expected Shortfall)** with Ledoit-Wolf covariance shrinkage.

---

## 1. System Architecture

```
                                 +-------------------------------------+
                                 | Multi-Asset Daily Price Histories   |
                                 | (AAPL, MSFT, JPM, XOM, JNJ, TLT)    |
                                 +------------------+------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
              +--------------------+                                +--------------------+
              | Expected Returns   |                                | Ledoit-Wolf        |
              | mu = E[R_i] * 252  |                                | Covariance Sigma   |
              +----------+---------+                                +---------+----------+
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                                                    v
                                 +-------------------------------------+
                                 | Quantitative Strategy Optimizers    |
                                 | (SLSQP / Quadratic Programming)     |
                                 +------------------+------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                          |                          |
                         v                          v                          v
              +--------------------+     +--------------------+     +--------------------+
              | Markowitz Sharpe   |     | Equal Risk Parity  |     | PMPT Sortino & CVaR|
              | (Max Yield / Risk) |     | (Equal Marginal RC)|     | (Downside Tail Opt)|
              +--------------------+     +--------------------+     +--------------------+
```

---

## 2. Mathematical Formulation

### **1. Markowitz Mean-Variance Optimization (Max Sharpe)**:
$$\max_{w} \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}} \quad \text{s.t. } \sum_{i=1}^{N} w_i = 1, \quad w_i \ge 0$$

### **2. Equal Risk Contribution (Risk Parity)**:
$$RC_i = w_i \frac{(\Sigma w)_i}{\sigma_p} = \frac{1}{N} \sigma_p \quad \forall i=1, \dots, N$$

### **3. Post-Modern Portfolio Theory (Max Sortino)**:
$$\max_{w} \frac{\mu_p - r_f}{\sqrt{\frac{1}{T} \sum_{t=1}^{T} \min(0, R_{p,t} - r_f)^2}}$$

### **4. 95% Conditional Value at Risk (Expected Shortfall)**:
$$\min_{w} \text{CVaR}_{0.95}(w) = -\mathbb{E}[R_p \mid R_p \le \text{VaR}_{0.95}]$$

---

## 3. Exact Computed Benchmark Results (5-Year Multi-Asset Universe)

```
===============================================================================================
MULTI-STRATEGY QUANTITATIVE PORTFOLIO OPTIMIZER
===============================================================================================
Strategy                       | Exp Return   | Volatility   | Sharpe     | Sortino   
-----------------------------------------------------------------------------------------------
Markowitz Max Sharpe           | 19.36%       | 16.83%       | 0.91       | -         
Equal Risk Parity (ERC)        | 6.02%        | 8.82%        | 0.23       | -         
PMPT Max Sortino               | 19.60%       | 17.10%       | -          | 0.93      
Tail Risk Min CVaR (95% ES)    | 2.68%        | 8.20%        | -          | -         
===============================================================================================

Discrete $100,000 Portfolio Execution:
  * Buy 64 shares of AAPL ($9,600.00)
  * Buy 229 shares of JPM ($80,150.00)
  * Buy 6 shares of XOM ($600.00)
  * Total Allocated: $99,883.16 | Uninvested Cash: $116.84
===============================================================================================
```

---

## 4. Quick Start & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run multi-strategy portfolio optimization pipeline
python run_pipeline.py

# 3. Run automated unit tests
python test_portfolio_optimizer.py
```

---

## 5. Master Placement Resume Description

> **Multi-Strategy Quantitative Portfolio Optimizer (SLSQP / QP)**
> * Engineered an institutional multi-strategy portfolio allocation engine implementing Markowitz Mean-Variance, Equal Risk Parity (ERC), PMPT Max Sortino, and Min CVaR (95% Expected Shortfall).
> * Implemented Ledoit-Wolf covariance shrinkage matrix regularization to prevent matrix inversion singularity and out-of-sample portfolio instability.
> * Designed a greedy discrete cash allocation engine converting continuous optimal weights into executable integer share lots for a \$100,000 capital book.

---

## License
MIT License. Open for academic research and portfolio demonstration.