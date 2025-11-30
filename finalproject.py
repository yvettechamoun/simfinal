import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# -----------------------------
# Stochastic recovery time based on surgery length
# -----------------------------
def recovery_time(surgery_length):
    base = 8 + 0.25 * surgery_length
    noise = np.random.gamma(shape=2, scale=14)
    return base + noise

# -----------------------------
# Surgery distribution sampling using truncated normal
# -----------------------------
def sample_surgery_durations_truncnorm(n_samples, mean=160, std=40, lower=30, upper=300):
    # Convert bounds to truncated normal parameters (surgeries range from 30-300 minutes)
    a = (lower - mean) / std
    b = (upper - mean) / std
    # Sample from truncated normal
    samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_samples)
    return samples

# -----------------------------
# Simulation Parameters
# -----------------------------
n_samples = 216
n_or = 3
n_recovery = 2
day_length = 12 * 60  # minutes

arrival_lambda = 18
arrival_times = np.cumsum(np.random.exponential(scale=60 / arrival_lambda, size=n_samples))

# Use truncated normal to sample surgery durations
surgery_durations = sample_surgery_durations_truncnorm(n_samples, mean=150, std=40, lower=30, upper=300)
recovery_durations = np.array([recovery_time(s) for s in surgery_durations])

# -----------------------------
# Simulation With Correct Overflow Rule
# -----------------------------
def simulate_with_wait_arrays(arrivals, surgeries, recoveries, n_or, n_recovery, day_length):
    or_available = np.zeros(n_or)
    rec_available = np.zeros(n_recovery)

    wait_or = []
    wait_rec = []

    or_busy_within_day = np.zeros(n_or)

    overflow_count = 0
    scheduled_count = 0

    for i in range(len(arrivals)):
        arrival = arrivals[i]
        s_time = surgeries[i]
        r_time = recoveries[i]

        # OR scheduling
        idx_or = np.argmin(or_available)
        start_or = max(arrival, or_available[idx_or])
        finish_or = start_or + s_time

        if start_or >= day_length:
            overflow_count += 1
            continue

        scheduled_count += 1
        wait_or.append(start_or - arrival)
        busy_within_day = max(0.0, min(finish_or, day_length) - start_or)
        or_busy_within_day[idx_or] += busy_within_day
        or_available[idx_or] = finish_or

        # Recovery scheduling
        idx_rec = np.argmin(rec_available)
        start_rec = max(finish_or, rec_available[idx_rec])
        finish_rec = start_rec + r_time
        wait_rec.append(start_rec - finish_or)
        rec_available[idx_rec] = finish_rec

    total_day_capacity = n_or * day_length
    utilization = np.sum(or_busy_within_day) / total_day_capacity
    overflow_prob = overflow_count / len(arrivals)

    wait_or_arr = np.array(wait_or) if len(wait_or) > 0 else np.array([])
    wait_rec_arr = np.array(wait_rec) if len(wait_rec) > 0 else np.array([])

    return wait_or_arr, wait_rec_arr, utilization, overflow_prob, scheduled_count

# -----------------------------
# Run Simulation
# -----------------------------
wait_or, wait_rec, or_util, overflow_prob, scheduled_count = simulate_with_wait_arrays(
    arrival_times, surgery_durations, recovery_durations, n_or, n_recovery, day_length
)

avg_or_wait = np.nan if wait_or.size == 0 else np.mean(wait_or)
avg_rec_wait = np.nan if wait_rec.size == 0 else np.mean(wait_rec)

print(f"Scheduled patients (started before day end): {scheduled_count}/{n_samples}")
print(f"Average OR Wait: {avg_or_wait:.2f} minutes")
print(f"Average Recovery Wait: {avg_rec_wait:.2f} minutes")
print(f"OR Utilization (within day window): {or_util*100:.2f}%")
print(f"Probability of Overflow (couldn't start): {overflow_prob*100:.2f}%")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(wait_or)
plt.title("OR Wait Times (scheduled patients)")
plt.xlabel("Scheduled Patient Index")
plt.ylabel("Wait Time (minutes)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(wait_rec)
plt.title("Recovery Bed Wait Times (scheduled patients)")
plt.xlabel("Scheduled Patient Index")
plt.ylabel("Wait Time (minutes)")
plt.grid(True)
plt.show()
