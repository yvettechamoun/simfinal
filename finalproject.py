import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Complex Surgery Duration Function
# -----------------------------
def surgery_duration_function(x):
    return np.exp(-np.sin(3 * x ** 3 - 3 * np.cos(x)))


# Stochastic recovery time based on surgery length
def recovery_time(surgery_length):
    base = 8 + 0.25 * surgery_length
    noise = np.random.gamma(shape=2, scale=14)
    return base + noise


# Surgery distribution sampling
def sample_surgery_durations(n_samples, x_range=(0, 1)):
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = surgery_duration_function(x)
    y = y / np.sum(y)

    samples = np.random.choice(x, size=n_samples, p=y)

    # Scale to realistic minutes
    samples = 120 + 60 * samples
    samples = np.clip(samples, 30, 300)
    return samples


# -----------------------------
# Simulation Parameters
# -----------------------------
n_samples = 2000
n_or = 3
n_recovery = 2
day_length = 12 * 60  # minutes

arrival_lambda = 2
arrival_times = np.cumsum(np.random.exponential(scale=60 / arrival_lambda, size=n_samples))

surgery_durations = sample_surgery_durations(n_samples)
recovery_durations = np.array([recovery_time(s) for s in surgery_durations])


# -----------------------------
# Simulation with Utilization + Overflow
# -----------------------------
def simulate_with_wait_arrays(arrivals, surgeries, recoveries, n_or, n_recovery):
    or_available = np.zeros(n_or)
    rec_available = np.zeros(n_recovery)

    wait_or = []
    wait_rec = []

    or_busy_time = np.zeros(n_or)
    overflow_count = 0

    for i in range(len(arrivals)):
        arrival = arrivals[i]
        s_time = surgeries[i]
        r_time = recoveries[i]

        # OR assignment
        idx_or = np.argmin(or_available)
        start_or = max(arrival, or_available[idx_or])
        finish_or = start_or + s_time

        # OR overflow: cannot complete before day ends
        if finish_or > day_length:
            overflow_count += 1
            continue

        wait_or.append(start_or - arrival)
        or_busy_time[idx_or] += s_time
        or_available[idx_or] = finish_or

        # Recovery assignment
        idx_rec = np.argmin(rec_available)
        start_rec = max(finish_or, rec_available[idx_rec])
        finish_rec = start_rec + r_time

        # Recovery overflow
        if start_rec > day_length:
            overflow_count += 1
            continue

        wait_rec.append(start_rec - finish_or)
        rec_available[idx_rec] = finish_rec

    # OR utilization
    total_or_capacity = n_or * day_length
    total_or_busy = np.sum(or_busy_time)
    utilization = total_or_busy / total_or_capacity

    overflow_prob = overflow_count / len(arrivals)

    return (
        np.array(wait_or),
        np.array(wait_rec),
        utilization,
        overflow_prob
    )


# -----------------------------
# Run Simulation
# -----------------------------
wait_or, wait_rec, or_util, overflow_prob = simulate_with_wait_arrays(
    arrival_times, surgery_durations, recovery_durations, n_or, n_recovery
)

print(f"Average OR Wait: {np.mean(wait_or):.2f} minutes")
print(f"Average Recovery Wait: {np.mean(wait_rec):.2f} minutes")
print(f"OR Utilization: {or_util*100:.2f}%")
print(f"Probability of Queue Overflow: {overflow_prob*100:.2f}%")


# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(wait_or)
plt.title("OR Wait Times")
plt.xlabel("Patient Index")
plt.ylabel("Wait Time (minutes)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(wait_rec)
plt.title("Recovery Bed Wait Times")
plt.xlabel("Patient Index")
plt.ylabel("Wait Time (minutes)")
plt.grid(True)
plt.show()
