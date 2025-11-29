import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Complex Surgery Duration Function
# -----------------------------
def surgery_duration_function(x):
    return np.exp(-np.sin(3 * x ** 3 - 3 * np.cos(x)))


# Stochastic recovery time based on surgery length
def recovery_time(surgery_length):
    """
    Recovery bed time depends on the length of the surgery.
    Longer surgeries → longer recovery.
    Includes gamma noise for biological variability.
    """
    base = 8 + 0.25 * surgery_length  # deterministic component
    noise = np.random.gamma(shape=2, scale=10)  # stochastic component
    return base + noise


# Sample surgery durations
def sample_surgery_durations(n_samples, x_range=(0, 1)):
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = surgery_duration_function(x)
    y = y / np.sum(y)

    samples = np.random.choice(x, size=n_samples, p=y)

    # Scale to minutes
    samples = 120 + 60 * samples
    samples = np.clip(samples, 30, 300)
    return samples


# -----------------------------
# Simulation Parameters
# -----------------------------
n_samples = 2000
n_or = 3
n_recovery = 3
day_length = 12 * 60  # minutes in 12 hours

# Exponential arrivals (Poisson process)
arrival_lambda = 2  # per hour
arrival_times = np.cumsum(np.random.exponential(scale=60 / arrival_lambda, size=n_samples))

# Sample surgery durations
surgery_durations = sample_surgery_durations(n_samples)

# Use stochastic recovery times based on surgery length
recovery_durations = np.array([recovery_time(s) for s in surgery_durations])


# -----------------------------
# Simulation Returning Wait Arrays
# -----------------------------
def simulate_with_wait_arrays(arrivals, surgeries, recoveries, n_or, n_recovery):
    or_available = np.zeros(n_or)
    rec_available = np.zeros(n_recovery)

    wait_or = []
    wait_rec = []

    for i in range(len(arrivals)):
        arrival = arrivals[i]
        s_time = surgeries[i]
        r_time = recoveries[i]

        # OR scheduling
        idx_or = np.argmin(or_available)
        start_or = max(arrival, or_available[idx_or])
        finish_or = start_or + s_time

        if finish_or > day_length:
            continue

        wait_or.append(start_or - arrival)
        or_available[idx_or] = finish_or

        # Recovery bed
        idx_rec = np.argmin(rec_available)
        start_rec = max(finish_or, rec_available[idx_rec])

        if start_rec > day_length:
            continue

        wait_rec.append(start_rec - finish_or)
        rec_available[idx_rec] = start_rec + r_time

    return np.array(wait_or), np.array(wait_rec)


# -----------------------------
# Run Simulation
# -----------------------------
wait_or, wait_rec = simulate_with_wait_arrays(arrival_times, surgery_durations, recovery_durations, n_or, n_recovery)

print("Average OR Wait:", np.mean(wait_or))
print("Average Recovery Wait:", np.mean(wait_rec))

# -----------------------------
# Graphs
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(wait_or)
plt.title("OR Wait Times (per patient)")
plt.xlabel("Patient Index")
plt.ylabel("Wait Time (minutes)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(wait_rec)
plt.title("Recovery Bed Wait Times (per patient)")
plt.xlabel("Patient Index")
plt.ylabel("Wait Time (minutes)")
plt.grid(True)
plt.show()
