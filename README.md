# Black–Scholes Option Pricing

Minimal Python implementation of **European option pricing** using the Black–Scholes–Merton model.  
This repository provides functions to calculate both **call** and **put** option prices, along with a sample usage.

---

## 📘 Model Overview
The Black–Scholes model assumes:
- European exercise (only at maturity)
- Constant risk-free rate \( r \)
- Constant volatility \( \sigma \)
- No dividends
- Frictionless markets

Formulas:

\[
d_1=\frac{\ln(S/K)+(r+\tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad
d_2=d_1-\sigma\sqrt{T}
\]

\[
C = S \, \Phi(d_1) - K e^{-rT} \Phi(d_2), \quad
P = K e^{-rT} \Phi(-d_2) - S \Phi(-d_1)
\]

Where:
- \( S \) = current stock price  
- \( K \) = strike price  
- \( T \) = time to maturity (in years)  
- \( r \) = risk-free interest rate  
- \( \sigma \) = volatility  
- \( \Phi \) = cumulative distribution function of standard normal  

---

## ⚙️ Requirements
Install the dependencies:
```bash
pip install numpy scipy

## 🔮 Extensions

This repo can be expanded with:
- **Greeks:** delta, gamma, theta, vega, rho  
- **Implied Volatility:** solve for σ given market price  
- **Dividends:** adjust for continuous dividend yield  
- **Stochastic Volatility:** plug in GARCH/EWMA estimates for σ  
- **Volatility Surface:** build surfaces across strikes and maturities  

---

## ✅ Quick Checks

- **Put–Call Parity:**  
  \[
  C - P = S - K e^{-rT}
  \]

- **Sanity:**  
  If \( S \gg K \), call ↑, put ↓.  
  If \( T \to 0 \), prices → intrinsic value.  

## 🛠️ Next Steps / Roadmap

Planned improvements and contributions for this repo:

- [ ] Add **Greeks** (Δ, Γ, Θ, ϒ, ρ) for risk management & hedging  
- [ ] Implement **Implied Volatility solver** (Newton–Raphson / bisection)  
- [ ] Extend model to handle **dividends** (continuous yield adjustment)  
- [ ] Integrate **stochastic volatility models** (e.g., GARCH, Heston)  
- [ ] Build a **volatility surface** visualization tool  
- [ ] Add **unit tests** for core functions  
- [ ] Create a **CLI tool** (`python blackscholes.py --S 105 --K 107 --T 0.25 --r 0.04 --sigma 0.10`)  
- [ ] Jupyter Notebook with examples and plots  
- [ ] Package as a **PyPI library** for easy install  

---

