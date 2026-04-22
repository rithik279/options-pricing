"""Binomial tree option pricing model for European and American options."""

import numpy as np


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 100,
    is_put: bool = False,
    is_american: bool = False,
) -> float:
    """Price a European or American option using a binomial (Cox-Ross-Rubinstein) tree.

    Args:
        S: Current stock price.
        K: Strike price.
        T: Time to maturity in years.
        r: Annual risk-free interest rate.
        sigma: Annual volatility.
        N: Number of time steps (higher N → more accurate, slower).
        is_put: If True, price a put; if False, price a call.
        is_american: If True, allow early exercise; if False, European exercise.

    Returns:
        Option price.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Terminal stock prices
    ST = np.array([S * (u ** (N - i)) * (d**i) for i in range(N + 1)])

    # Terminal payoffs
    if is_put:
        V = np.maximum(K - ST, 0.0)
    else:
        V = np.maximum(ST - K, 0.0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        V = discount * (p * V[:-1] + (1 - p) * V[1:])
        if is_american:
            Si = np.array([S * (u ** (i - j)) * (d**j) for j in range(i + 1)])
            if is_put:
                V = np.maximum(K - Si, V)
            else:
                V = np.maximum(Si - K, V)

    return float(V[0])


def _verify_parity(S=100, K=100, T=1, r=0.05, sigma=0.20, N=200):
    """Quick sanity check: European call - put should equal S - K*e^(-rT)."""
    c = binomial_price(S, K, T, r, sigma, N, is_put=False)
    p = binomial_price(S, K, T, r, sigma, N, is_put=True)
    parity = c - p
    synthetic = S - K * np.exp(-r * T)
    print(f"Call = {c:.4f}  Put = {p:.4f}  Diff = {parity:.4f}")
    print(f"S - K*e^(-rT) = {synthetic:.4f}  |error| = {abs(parity - synthetic):.6f}")


if __name__ == "__main__":
    S = 105.0
    K = 107.0
    T = 0.25
    r = 0.04
    sigma = 0.10
    N = 200

    c = binomial_price(S, K, T, r, sigma, N, is_put=False)
    p = binomial_price(S, K, T, r, sigma, N, is_put=True)
    c_am = binomial_price(S, K, T, r, sigma, N, is_put=False, is_american=True)
    p_am = binomial_price(S, K, T, r, sigma, N, is_put=True, is_american=True)

    print(f"Stock price:    S = {S}")
    print(f"Strike price:   K = {K}")
    print(f"Time:           T = {T} years")
    print(f"Rate:           r = {r:.0%}")
    print(f"Volatility:     sigma = {sigma:.0%}")
    print(f"Steps:          N = {N}")
    print()
    print(f"European Call:   C = ${c:.4f}")
    print(f"European Put:    P = ${p:.4f}")
    print(f"American Call:  C* = ${c_am:.4f}")
    print(f"American Put:   P* = ${p_am:.4f}")
    print()
    print("Put-call parity check (binomial):")
    _verify_parity(S, K, T, r, sigma, N)
