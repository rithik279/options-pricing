# Options Pricing Models

A clean, tested Python library for pricing European and American options using the **Black-Scholes** model and **binomial (CRR) trees**.

---

## What It Does

- **Black-Scholes model**: Closed-form European call/put pricing with Greeks (delta, gamma, theta, vega)
- **Binomial tree (Cox-Ross-Rubinstein)**: European and American option pricing with convergence to Black-Scholes
- **Put-call parity** verification built in
- Well-tested against textbook examples and known relationships

---

## How to Run

```bash
# Install dependencies
pip install numpy scipy

# Run pricing examples
python logic/blackscholes.py
python logic/binomialtree.py

# Run tests
pip install pytest
pytest tests/
```

---

## Example Usage

```python
from logic.blackscholes import price_call, price_put, delta_call, gamma
from logic.binomialtree import binomial_price

# Black-Scholes European call
C = price_call(S=105, K=107, T=0.25, r=0.04, sigma=0.10)
print(f"Call price: ${C:.4f}")   # ~3.74

# Binomial American put
P = binomial_price(S=105, K=107, T=0.25, r=0.04, sigma=0.10, is_put=True, is_american=True)
print(f"American put: ${P:.4f}")

# Greeks
print(f"Delta: {delta_call(105, 107, 0.25, 0.04, 0.10):.4f}")
print(f"Gamma: {gamma(105, 107, 0.25, 0.04, 0.10):.4f}")
```

---

## Extensions

| Feature | Status |
|---|---|
| Greeks (delta, gamma, theta, vega) | ✅ Included |
| American options (binomial) | ✅ Included |
| Put-call parity check | ✅ Included |
| Implied volatility solver | 🔜 Planned |
| Dividend adjustment | 🔜 Planned |
| Volatility surface | 🔜 Planned |
| CLI interface | 🔜 Planned |
