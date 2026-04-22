"""Black-Scholes option pricing model for European calls and puts."""

import numpy as np
from scipy.stats import norm


def price_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European call option using Black-Scholes.

    Args:
        S: Current stock price.
        K: Strike price.
        T: Time to maturity in years.
        r: Annual risk-free interest rate (e.g. 0.05 for 5%).
        sigma: Annual volatility (e.g. 0.20 for 20%).

    Returns:
        Option price.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * max(T, 0)), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def price_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European put option using Black-Scholes.

    Args:
        S: Current stock price.
        K: Strike price.
        T: Time to maturity in years.
        r: Annual risk-free interest rate (e.g. 0.05 for 5%).
        sigma: Annual volatility (e.g. 0.20 for 20%).

    Returns:
        Option price.
    """
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * max(T, 0)) - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Delta of a European call option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Delta of a European put option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma of a European option (same for calls and puts)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Theta of a European call option (per year, not per trading day)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 - term2


def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Theta of a European put option (per year, not per trading day)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return term1 + term2


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega of a European option (same for calls and puts). Per 1% move."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) * 0.01


def put_call_parity(C: float, P: float, S: float, K: float, T: float, r: float, tol: float = 1e-6) -> bool:
    """Verify the put-call parity relationship: C - P = S - K*e^(-rT)."""
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    return abs(lhs - rhs) < tol


if __name__ == "__main__":
    S = 105.0
    K = 107.0
    T = 0.25
    r = 0.04
    sigma = 0.10

    C = price_call(S, K, T, r, sigma)
    P = price_put(S, K, T, r, sigma)

    print(f"Stock price:        S = {S}")
    print(f"Strike price:       K = {K}")
    print(f"Time to maturity:   T = {T} years")
    print(f"Risk-free rate:     r = {r:.0%}")
    print(f"Volatility:         sigma = {sigma:.0%}")
    print()
    print(f"Call price:         C = ${C:.4f}")
    print(f"Put price:          P = ${P:.4f}")
    print(f"Put-call parity:    {'PASSED' if put_call_parity(C, P, S, K, T, r) else 'FAILED'}")
    print()
    print(f"Delta (call):       dC = {delta_call(S, K, T, r, sigma):.4f}")
    print(f"Delta (put):        dP = {delta_put(S, K, T, r, sigma):.4f}")
    print(f"Gamma:              Gamma = {gamma(S, K, T, r, sigma):.4f}")
    print(f"Theta (call):       ThetaC = {theta_call(S, K, T, r, sigma):.4f}")
    print(f"Theta (put):        ThetaP = {theta_put(S, K, T, r, sigma):.4f}")
    print(f"Vega:               Vega = {vega(S, K, T, r, sigma):.4f}")
