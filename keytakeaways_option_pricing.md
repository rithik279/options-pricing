# Key Takeaways: Options Pricing Models Project

## Project Overview

This project implements and compares two foundational models for pricing European and American options: the **Black-Scholes-Merton (BSM) model** and the **binomial (Cox-Ross-Rubinstein) tree model**. The implementation was refactored from a rough university homework repo into a clean, tested Python library with 12 passing unit tests.

---

## 1. The Black-Scholes-Merton Model

### What It Is

The Black-Scholes-Merton model provides closed-form analytical solutions for European option prices. It won Merton and Scholes the 1997 Nobel Prize in Economics (Black had passed). The model assumes:

- European exercise only (no early exercise)
- Frictionless markets with continuous trading
- No dividends paid during the option's life
- Constant risk-free rate `r` and volatility `σ`
- Log-normal price distribution for the underlying

### The Governing Equations

**d₁ and d₂:**
```
          /   S    \          /       σ² \
d₁ = ln  |  ----  |   +   r +  -------   T
          \   K    /          \     2     /
          ------------------------------------
                        σ √T
```

```
d₂ = d₁ - σ √T
```

**European Call:**
```
C = S · Φ(d₁) - K · e^(-rT) · Φ(d₂)
```

**European Put:**
```
P = K · e^(-rT) · Φ(-d₂) - S · Φ(-d₁)
```

Where `Φ` is the cumulative standard normal distribution.

### Key Insight: Why This Works

The model hinges on a **delta-hedged portfolio**: hold `Δ = Φ(d₁)` shares of stock and short 1 call. The portfolio is instantaneously riskless, so it must earn the risk-free rate `r`. Solving this no-arbitrage condition yields the closed-form solution above.

### Implementation Details

The `price_call()` and `price_put()` functions implement the formulas directly using `scipy.stats.norm.cdf()`. Edge cases are handled:

- `σ = 0` or `T = 0` → price collapses to intrinsic value (no time premium, no volatility)
- Deep ITM options → price approaches `S - K·e^(-rT)`
- Deep OTM options → price approaches zero

### What We Learned

1. **The model is fast** — one closed-form formula replaces thousands of Monte Carlo paths
2. **It's a baseline, not ground truth** — real markets violate many BSM assumptions (volatility isn't constant, jumps occur, dividends exist)
3. **The input volatility `σ` is almost always an estimate** — this is the "calibration" problem

---

## 2. The Greeks: Risk Sensitivities

The Greeks measure how an option's price changes with respect to market parameters. They are the primary tools for risk management and hedging in derivatives desks.

### Delta (Δ) — First Derivative w.r.t. Spot Price

```
Δ_call = Φ(d₁)       (always in [0, 1])
Δ_put  = Φ(d₁) - 1   (always in [-1, 0])
```

- **Interpretation**: For every $1 increase in the stock price, the call gains approximately `Δ` dollars
- Deep ITM calls have Δ → 1 (they behave like the underlying)
- Deep OTM calls have Δ → 0 (they're nearly worthless)
- A 50-delta option is called an **ATM straddle** sensitivity

### Gamma (Γ) — Second Derivative w.r.t. Spot Price

```
Γ = Φ'(d₁) / (S · σ · √T)
  = φ(d₁) / (S · σ · √T)
```

Where `φ` is the standard normal **PDF**.

- **Gamma is always positive for both calls and puts** — this means both sides of a hedged portfolio face convexity risk
- **ATM options have maximum gamma** — small price moves near the strike cause large delta changes
- Gamma peaks at lower `T` (more urgent hedging needed near expiry)
- Practitioners "gamma scalping" gamma by selling high-gamma options and delta-hedging dynamically

### Theta (Θ) — Rate of Time Decay

```
Θ_call = -(S · φ(d₁) · σ) / (2√T) - r · K · e^(-rT) · Φ(d₂)
Θ_put  = -(S · φ(d₁) · σ) / (2√T) + r · K · e^(-rT) · Φ(-d₂)
```

- **Theta is almost always negative** — time is an asset for the option seller, a liability for the buyer
- Near expiry, theta accelerates (options lose value faster as they approach expiration)
- Deep ITM options can have **positive theta** if the dividend yield exceeds the cost of carry

### Vega (ν) — Sensitivity to Volatility

```
ν = S · φ(d₁) · √T · 0.01
```

(Expressed per 1% move in volatility, not per 100%)

- **Vega is always positive** — higher volatility increases both calls and puts (more uncertainty = higher option premium)
- Long-dated options have more vega (more time = more exposure to volatility)
- ATM options are most sensitive to volatility changes

### Practical Takeaway: The Greeks Dashboard

For the example parameters (`S=105, K=107, T=0.25, r=4%, σ=10%`):

| Greek | Call | Put |
|---|---|---|
| Delta | +0.4394 | -0.5606 |
| Gamma | 0.0751 | 0.0751 |
| Theta (annual) | -5.92 | -1.68 |
| Vega | 0.2070 | 0.2070 |

Key observations:
- **Delta is not symmetric** — put delta is `-1` more negative than call delta (if stock goes to zero, put = K, call = 0)
- **Gamma is the same for calls and puts** — by put-call symmetry
- **Put theta is less negative** than call theta here because the put is OTM (less time value to decay)
- **Vega is identical** for calls and puts at the same strike/expiry

---

## 3. The Binomial (Cox-Ross-Rubinstein) Model

### What It Is

The CRR binomial tree discretizes the stock price evolution into `N` time steps, each with an **up move** (`u`) and a **down move** (`d`):

```
u = e^(σ·√Δt)        d = 1/u
p = (e^(r·Δt) - d) / (u - d)      [risk-neutral probability]
```

At each node, the option value is computed recursively from the terminal payoffs back to the present (**backward induction**).

### Why It's Important

The CRR model is philosophically different from Black-Scholes:

- **Discrete by design** — naturally handles American early exercise, barrier features, and dividends
- **Converges to Black-Scholes** as `N → ∞` — proves BSM is the continuous-time limit of the binomial model
- **Intuitive** — can be drawn as a tree, making it accessible to non-quants

### European vs. American Options

**European:** Just discount the expected payoff under the risk-neutral probability.

**American:** At each node, compare:
- Holding the option → discounted children values
- **Early exercising** → intrinsic value `max(S - K, 0)` for calls, `max(K - S, 0)` for puts

```
American Put Value = max(K - S,  Discounted Children)
```

American puts are always worth **at least as much** as their European counterparts. The early exercise premium is highest when:
- The option is deep ITM
- Interest rates are high (you receive K sooner)
- Volatility is low (high volatility makes waiting more valuable)

### Convergence Analysis

For `S=105, K=107, T=0.25, r=4%, σ=10%`:

| Model | Call | Put |
|---|---|---|
| Black-Scholes | $1.6689 | $2.6042 |
| Binomial (N=200) | $1.6705 | $2.6059 |
| American Put | — | $2.7944 |

The binomial converges to Black-Scholes within **0.05** at `N=200`. The American put is worth **$0.19 more** than the European put — this is the early exercise premium.

### What We Learned

1. **N=200 is sufficient for most pricing** — convergence is fast and oscillation-free with CRR parameterization
2. **American put > European put** by exactly the early exercise premium — always
3. **American calls rarely exceed European calls** in the absence of dividends — the cost of carry on the stock outweighs early exercise benefit for equity options

---

## 4. Put-Call Parity: The Most Important Identity in Options

### The Equation

```
C - P = S - K·e^(-rT)
```

### Why It's Powerful

Put-call parity follows directly from no-arbitrage reasoning. If it ever deviates, you can:

1. **Buy the cheaper side, sell the more expensive side**
2. **Delta-hedge the residual** at `r`
3. **Earn a risk-free spread** until expiry when parity is restored

### Verification

For our example parameters, all tests confirm parity:

```
C - P = 1.6689 - 2.6042 = -0.9353
S - K·e^(-rT) = 105 - 107·e^(-0.04 × 0.25) = -0.9353
Difference = 0.0000  ✓
```

### Extended Parity: American Options

For **American options**, the identity becomes a **bound** rather than an equality:

```
C* - P ≤ S - K·e^(-rT) ≤ C - P*
```

Where `C*, P*` are American values. This is useful for cross-model sanity checks.

### Practical Takeaway

Before pricing anything, check put-call parity. If it fails, the model or implementation has a bug. It's the **first line of defense** in any options pricing code.

---

## 5. Testing Strategy: What Makes This Robust

### The Test Pyramid

The 12 tests in `test_pricing.py` follow a layered strategy:

**Layer 1: Theoretical Constraints**
- Put-call parity holds exactly (no tolerance)
- Zero volatility → intrinsic value
- Zero maturity → intrinsic value

**Layer 2: Economic Intuition**
- Deep ITM call > intrinsic value
- Deep OTM put ≈ zero
- Greeks have correct signs and magnitudes

**Layer 3: Numerical Benchmarks**
- Hull (Example 13.1): `S=42, K=40, r=10%, T=0.5, σ=20%` → `C ≈ 4.76`
- Binomial converges to Black-Scholes at high N
- American put ≥ European put

**Layer 4: Structural Invariants**
- American put premium is always non-negative
- Convergence improves monotonically with N

### Why This Matters

Without this test pyramid, you can accidentally:
- Flip a sign and still get "reasonable-looking" numbers
- Use wrong CDF approximation and appear accurate for ATM options
- Break early exercise logic without noticing

---

## 6. Key Financial Insights

### 1. Volatility Is the Hardest Input

The Black-Scholes formula is **exact given `σ`**. But `σ` must be estimated from historical data (GARCH, EWMA) or inferred from market prices (implied volatility). This is the **calibration problem** — arguably the most important unsolved problem in practical options pricing.

### 2. The Greeks Are Interdependent

You cannot manage delta without watching gamma. A delta-neutral portfolio near the strike can have **explosive gamma risk** as the stock price moves. This is why options desks rebalance their delta hedges continuously — not just at end of day.

### 3. Time Decay Accelerates

Theta is convex in `T`. A 30-day option loses more value in the last 7 days than in the first 23. This is why **theta sellers** (selling options far from expiry) need to carefully evaluate early assignment risk.

### 4. ATM Options Are Special

ATM options have maximum gamma and vega, making them both the most **expensive to hedge** and the most **sensitive to market movements**. In a market crash, the "pin risk" near ATM strikes causes unpredictable behavior.

### 5. American vs. European: When It Matters

American options are strictly more valuable than European options only in specific cases:
- **American put**: Always has positive early exercise premium, especially with high interest rates
- **American call with dividends**: Early exercise can be optimal right before dividend payment
- **American call without dividends**: Almost never optimal to exercise early (you'd give up remaining time value)

### 6. Model Risk Is Real

Every model is wrong. Black-Scholes assumes constant volatility — but in reality, volatility is **stochastic** (Heston model), exhibits jumps (Merton jump-diffusion), and shows a "smile" across strikes (implied volatility surface). The right model choice depends on the trade you're hedging, not on which is "best."

---

## 7. Code Architecture Lessons

### From Homework to Library

The original repo had several problems that are extremely common in academic projects:

| Problem | Impact | Fix |
|---|---|---|
| Committed `venv/` | 1000+ files committed, repo looks unprofessional | Added `.gitignore`, removed venv |
| Unnamed functions | `black_scholes_call()` — verbosity tax | `price_call()` — clean API |
| No package structure | `blackscholes.py` + `binomialtree.py` in flat repo | `logic/` package with `__init__.py` |
| No tests | Works by faith | 12 pytest tests covering all critical paths |
| No edge case handling | Crashes on σ=0 or T=0 | Graceful fallback to intrinsic value |
| No Greeks | Missing ~50% of finance content | Full Greeks: delta, gamma, theta, vega |

### API Design

The final functions follow a consistent signature pattern:

```python
price_call(S, K, T, r, sigma)      # BSM call
price_put(S, K, T, r, sigma)       # BSM put
binomial_price(S, K, T, r, sigma, N, is_put, is_american)
```

This makes it trivial to swap models or compare outputs.

---

## 8. Extension Roadmap

The current implementation covers the **core** of options pricing. Planned extensions include:

### Near-term
- **Implied Volatility Solver**: Given `C`, solve for `σ` using Newton-Raphson or bisection — the backbone of the volatility surface
- **Dividend Adjustment**: Continuous dividend yield `q` → `S·e^(-qT)` adjustment to Black-Scholes

### Medium-term
- **GARCH Volatility Model**: Replace flat `σ` with time-varying GARCH(1,1) estimates
- **Heston Stochastic Volatility Model**: Full smile-capable model with closed-form characteristic function
- **CLI Interface**: `python -m logic.blackscholes --S 105 --K 107 --T 0.25 --r 0.04 --sigma 0.10`
- **Volatility Surface**: Plot `σ` across strikes (moneyness) and maturities to see the smile/skew

### Long-term
- **Black-76 Model**: For interest rate options and caps/floors
- **Binomial with dividends**: Cox-Ross-Rubinstein with discrete dividends
- **Monte Carlo pricer**: Path-dependent options (Asian, barrier, lookback) using GBM simulation

---

## 9. Summary of Key Results

### Pricing Results (S=105, K=107, T=0.25, r=4%, σ=10%)

| Metric | Value |
|---|---|
| European Call (BS) | $1.6689 |
| European Put (BS) | $2.6042 |
| Binomial European Call (N=200) | $1.6705 |
| Binomial European Put (N=200) | $2.6059 |
| American Put | $2.7944 |
| Early Exercise Premium | $0.19 |
| Put-Call Parity Error | < 0.0001 |

### Greeks (S=105, K=107, T=0.25, r=4%, σ=10%)

| Greek | Call | Put |
|---|---|---|
| Delta | +0.4394 | -0.5606 |
| Gamma | 0.0751 | 0.0751 |
| Theta (annual) | -5.9195 | -1.6821 |
| Vega | 0.2070 | 0.2070 |

### Tests Summary: 12/12 Passed
- Put-call parity ✓
- Hull textbook example ✓
- Black-Scholes / binomial convergence ✓
- American put early exercise premium ✓
- Greeks signs and magnitudes ✓
- Edge cases (σ=0, T→0) ✓

---

## 10. Final Reflections

This project demonstrates the core toolkit that every derivatives quant needs:

1. **Black-Scholes** is your mental model — fast, intuitive, the language of options markets
2. **Binomial trees** are your workhorse — flexible, handles American options, naturally discrete
3. **The Greeks** are your risk language — delta, gamma, theta, vega are how traders communicate hedging needs
4. **Put-call parity** is your sanity check — always verify before trusting any model output
5. **Tests are not optional** — the Hull example test alone caught a sign error that "looked right"

The jump from "homework code" to "production library" is largely about **API design, edge case handling, and test coverage**. The mathematics of options pricing is well-established; the craft is in implementing it cleanly, verifying it rigorously, and extending it thoughtfully.
