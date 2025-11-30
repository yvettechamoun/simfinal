"""
 - generate_samples.py - Contains all functions for random sample generation
"""
import numpy as np


# -----------------------------
# Generate Arrival times
# -----------------------------
def get_arrival_times(n_samples, arrival_lambda=18) -> np.ndarray:
    arrival_times = np.cumsum(np.random.exponential(scale=60 / arrival_lambda, size=n_samples))

    return arrival_times


# -----------------------------
# Stochastic recovery time based on surgery length
# -----------------------------
def get_recovery_time(surgery_length) -> float:
    base = 8 + 0.25 * surgery_length
    noise = np.random.gamma(shape=2, scale=14)

    return base + noise


# -----------------------------
# Surgery time distribution sampling using Log-Normal 
# mu=26.123 and sigma=1.862
# From "Integrating Data Mining and Optimization Techniques on Surgery Scheduling"
# DOI:10.1007/978-3-642-35527-1_49
# -----------------------------
def get_surgery_durations(n_samples, mu=26.123, sigma=1.862):
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)

    return samples
