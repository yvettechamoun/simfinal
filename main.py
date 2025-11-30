"""
 - main.py - main program file
"""
from generate_samples import get_arrival_times, get_surgery_durations, get_recovery_time
from simulation import simulate_with_wait_arrays
import numpy as np
import matplotlib.pyplot as plt


def main():
    # -----------------------------
    # Simulation Parameters
    # -----------------------------
    n_samples = 216
    n_or = 3
    n_recovery = 2
    day_length = 12 * 60  # minutes

    arrival_lambda = 18 # Unit?
    arrival_times = get_arrival_times(n_samples, arrival_lambda=arrival_lambda)

    # Use log-normal to sample surgery durations
    surgery_durations = get_surgery_durations(n_samples)
    recovery_durations = np.array([get_recovery_time(s) for s in surgery_durations])

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


if __name__ == '__main__':
    main()
