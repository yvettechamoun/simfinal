import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


# -----------------------------
# Complex Surgery Duration Function
# -----------------------------
def surgery_duration_function(x):
    """Target unnormalized distribution for surgery durations."""
    return np.exp(-np.sin(3 * x ** 3 - 3 * np.cos(x)))


# -----------------------------
# Metropolis-Hastings Sampler (truncated normal proposal)
# -----------------------------
def sample_surgery_durations_mh_truncnorm(n_samples, x0=150,domain=(30, 300),proposal_std=40,
                                          burn_in=100,
                                          thinning=5,
                                          random_seed=42):
    np.random.seed(random_seed)
    samples = []
    x_current = x0
    low, high = domain
    total_samples_needed = n_samples * thinning + burn_in

    for _ in range(total_samples_needed):
        # Truncated normal proposal around current sample
        a, b = (low - x_current) / proposal_std, (high - x_current) / proposal_std
        x_proposal = truncnorm.rvs(a, b, loc=x_current, scale=proposal_std)

        # Acceptance probability
        p_current = surgery_duration_function(x_current)
        p_proposal = surgery_duration_function(x_proposal)
        alpha = min(1, p_proposal / p_current)

        # Accept/reject
        if np.random.rand() < alpha:
            x_current = x_proposal

        samples.append(x_current)

    # Apply burn-in and thinning
    samples = np.array(samples)
    samples = samples[burn_in::thinning]
    return samples


# -----------------------------
# Stochastic Recovery Time
# -----------------------------
def recovery_time(surgery_length):
    """Recovery time depends on surgery length plus stochastic gamma noise."""
    base = 60 + 0.25 * surgery_length  # base time in minutes
    noise = np.random.gamma(shape=2, scale=7)
    return base + noise


# -----------------------------
# Simulation of OR and Recovery Scheduling
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

        # -----------------------------
        # OR Scheduling
        # -----------------------------
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

        # -----------------------------
        # Recovery Bed Scheduling
        # -----------------------------
        idx_rec = np.argmin(rec_available)
        start_rec = max(finish_or, rec_available[idx_rec])
        finish_rec = start_rec + r_time
        wait_rec.append(start_rec - finish_or)
        rec_available[idx_rec] = finish_rec

    # -----------------------------
    # Performance Metrics
    # -----------------------------
    total_day_capacity = n_or * day_length
    utilization = np.sum(or_busy_within_day) / total_day_capacity
    overflow_prob = overflow_count / len(arrivals)

    wait_or_arr = np.array(wait_or) if len(wait_or) > 0 else np.array([])
    wait_rec_arr = np.array(wait_rec) if len(wait_rec) > 0 else np.array([])

    return wait_or_arr, wait_rec_arr, utilization, overflow_prob, scheduled_count


# -----------------------------
# Simulation Parameters
# -----------------------------
n_samples = 180  # number of patients in one day
n_or = 3  # number of ORs
n_recovery = 2  # number of recovery beds
day_length = 12 * 60  # 12 hours in minutes
arrival_lambda = 15  # average patient arrivals per hour

# Generate patient arrival times
arrival_times = np.cumsum(np.random.exponential(scale=60 / arrival_lambda, size=n_samples))

# Sample surgery durations using Metropolis-Hastings
surgery_durations = sample_surgery_durations_mh_truncnorm(
    n_samples=n_samples,
    x0=150,
    domain=(30, 300),
    proposal_std=40,
    burn_in=100,
    thinning=5,
    random_seed=42
)

# Sample recovery durations based on surgery durations
recovery_durations = np.array([recovery_time(s) for s in surgery_durations])

# Run the simulation
wait_or, wait_rec, or_util, overflow_prob, scheduled_count = simulate_with_wait_arrays(
    arrival_times, surgery_durations, recovery_durations, n_or, n_recovery, day_length
)

# -----------------------------
# Results
# -----------------------------
avg_or_wait = np.nan if wait_or.size == 0 else np.mean(wait_or)
avg_rec_wait = np.nan if wait_rec.size == 0 else np.mean(wait_rec)

print(f"Scheduled patients (started before day end): {scheduled_count}/{n_samples}")
print(f"Average OR Wait: {avg_or_wait:.2f} minutes")
print(f"Average Recovery Wait: {avg_rec_wait:.2f} minutes")
print(f"OR Utilization (within day window): {or_util * 100:.2f}%")
print(f"Probability of Overflow (couldn't start): {overflow_prob * 100:.2f}%")

# -----------------------------
# Visualization
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
