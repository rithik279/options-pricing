"""Tests for Black-Scholes and binomial tree pricing."""

import numpy as np
import pytest
from logic.blackscholes import (
    price_call,
    price_put,
    put_call_parity,
    delta_call,
    delta_put,
    gamma,
    theta_call,
    theta_put,
    vega,
)
from logic.binomialtree import binomial_price


class TestBlackScholes:
    """Black-Scholes sanity checks."""

    def test_call_put_parity(self):
        """C - P = S - K*e^(-rT) for European options."""
        S, K, T, r, sigma = 105.0, 107.0, 0.25, 0.04, 0.10
        C = price_call(S, K, T, r, sigma)
        P = price_put(S, K, T, r, sigma)
        assert put_call_parity(C, P, S, K, T, r), "Put-call parity violated"

    def test_call_in_the_money(self):
        """Deep ITM call should be worth approximately S - K*e^(-rT)."""
        S, K, T, r, sigma = 150.0, 100.0, 0.5, 0.05, 0.20
        C = price_call(S, K, T, r, sigma)
        intrinsic = S - K * np.exp(-r * T)
        assert C > intrinsic, "Call must be worth at least intrinsic value"
        assert C < S, "Call cannot exceed spot price"

    def test_put_out_of_the_money(self):
        """Deep OTM put should be worth close to zero."""
        # S > K: put is OTM
        S, K, T, r, sigma = 100.0, 50.0, 0.5, 0.05, 0.20
        P = price_put(S, K, T, r, sigma)
        assert P < 0.10, "Deep OTM put should be nearly worthless"

    def test_call_put_near_zero_maturity(self):
        """At T=0, price should equal intrinsic value."""
        S, K, T, r, sigma = 105.0, 100.0, 1e-6, 0.05, 0.20
        C = price_call(S, K, T, r, sigma)
        P = price_put(S, K, T, r, sigma)
        assert abs(C - max(S - K, 0)) < 0.01, "Call near expiry should equal intrinsic"
        assert abs(P - max(K - S, 0)) < 0.01, "Put near expiry should equal intrinsic"

    def test_zero_volatility_call(self):
        """With σ=0, call = max(S - K*e^(-rT), 0)."""
        S, K, T, r = 110.0, 100.0, 1.0, 0.05
        C = price_call(S, K, T, r, sigma=0.0)
        discounted_K = K * np.exp(-r * T)
        expected = max(S - discounted_K, 0)
        assert abs(C - expected) < 1e-9, "σ=0 call should equal discounted intrinsic"

    def test_greeks_signs(self):
        """Delta is positive for calls, negative for puts; gamma is always positive."""
        S, K, T, r, sigma = 105.0, 107.0, 0.25, 0.04, 0.10
        assert delta_call(S, K, T, r, sigma) > 0, "Call delta must be positive"
        assert delta_put(S, K, T, r, sigma) < 0, "Put delta must be negative"
        assert gamma(S, K, T, r, sigma) > 0, "Gamma must be positive"
        assert vega(S, K, T, r, sigma) > 0, "Vega must be positive"

    def test_known_market_values(self):
        """Reproduce a well-known textbook example (Hull, Example 13.1)."""
        # S=42, K=40, r=0.1, T=0.5, σ=0.2
        # Expected: C ≈ 4.76
        C = price_call(S=42, K=40, T=0.5, r=0.1, sigma=0.2)
        assert abs(C - 4.76) < 0.02, f"Expected ~4.76, got {C:.4f}"

    def test_theta_negative(self):
        """Theta should be negative (time decay) for vanilla options."""
        S, K, T, r, sigma = 105.0, 107.0, 0.25, 0.04, 0.10
        assert theta_call(S, K, T, r, sigma) < 0, "Call theta should be negative"


class TestBinomialTree:
    """Binomial tree sanity checks."""

    def test_european_call_matches_black_scholes(self):
        """Binomial should approximate Black-Scholes for European calls."""
        S, K, T, r, sigma = 105.0, 107.0, 0.25, 0.04, 0.10
        bs_call = price_call(S, K, T, r, sigma)
        bin_call = binomial_price(S, K, T, r, sigma, N=500, is_put=False)
        assert abs(bin_call - bs_call) < 0.05, "Binomial should match Black-Scholes"

    def test_european_put_parity(self):
        """C - P = S - K*e^(-rT) for binomial European options."""
        S, K, T, r, sigma = 105.0, 107.0, 0.25, 0.04, 0.10
        c = binomial_price(S, K, T, r, sigma, N=500, is_put=False)
        p = binomial_price(S, K, T, r, sigma, N=500, is_put=True)
        synthetic = S - K * np.exp(-r * T)
        assert abs((c - p) - synthetic) < 0.01, "Put-call parity failed for binomial"

    def test_american_put_above_european(self):
        """American put >= European put (early exercise premium)."""
        S, K, T, r, sigma = 50.0, 100.0, 0.5, 0.05, 0.20
        p_eu = binomial_price(S, K, T, r, sigma, N=200, is_put=True, is_american=False)
        p_am = binomial_price(S, K, T, r, sigma, N=200, is_put=True, is_american=True)
        assert p_am >= p_eu - 1e-6, "American put must be >= European put"

    def test_convergence_with_steps(self):
        """Price should converge as N increases."""
        S, K, T, r, sigma = 105.0, 107.0, 0.25, 0.04, 0.10
        bs_call = price_call(S, K, T, r, sigma)
        prices = [binomial_price(S, K, T, r, sigma, N=n, is_put=False) for n in [10, 50, 200]]
        for p in prices:
            assert abs(p - bs_call) < 0.5, f"Price {p:.4f} deviates too much at N={n}"
