"""
 - simulation.py - Contains all simulation functions of the program
"""

import numpy as np

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

