import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


#define complex surgery duration function
def surgery_duration_function(x):
    # Three modes: quick procedures, standard, and complex surgeries
    mode1 = 0.5 * np.exp(-((x - 50)**2) / (2 * 10**2))   # Quick: ~45 min
    mode2 = 1.1 * np.exp(-((x - 120)**2) / (2 * 50**2))  # Standard: ~120 min
    mode3 = 0.6 * np.exp(-((x - 240)**2) / (2 * 20**2))  # Complex: ~240 min
    return mode1 + mode2 + mode3

# generate sample surgery times using MH algorithm
# domain 30-300 minutes
# Switched to normal distribution -> symmetric
def sample_surgery_durations_mh(n_samples, x0=150,domain=(30, 300),
                                          proposal_std=40,
                                          burn_in=1000,
                                          thinning=5,
                                          random_seed=42):
    np.random.seed(random_seed)
    samples = []
    x_current = x0
    low, high = domain
    total_samples_needed = n_samples * thinning + burn_in

    for _ in range(total_samples_needed):
        # normal proposal around current sample
        a, b = (low - x_current) / proposal_std, (high - x_current) / proposal_std
        x_proposal = np.random.normal(x_current, proposal_std)

        if x_proposal < low or x_proposal > high:
            samples.append(x_current)
            continue

        # Acceptance probability
        alpha = min(1, surgery_duration_function(x_proposal) / surgery_duration_function(x_current))

        # Accept/reject
        if np.random.rand() < alpha:
            x_current = x_proposal

        samples.append(x_current)

    # Apply burn-in and thinning
    samples = np.array(samples)
    samples = samples[burn_in::thinning]
    return samples


#recovery time is based on surgery length
def recovery_time(surgery_length):
    base = 60 + 0.25 * surgery_length  # the base recovery time is one hour, plus 1/4 the surgery duration
    noise = np.random.gamma(shape=2, scale=7)  # add a random component for natural variability in recovery time
    return base + noise


#simulate with overflow allowance (surgeries that start before day is over will finish even if the day ends)
def simulate_with_wait_arrays(arrivals, surgeries, recoveries, n_or, n_recovery, day_length):
    or_available = np.zeros(n_or)
    rec_available = np.zeros(n_recovery)
    # initialize blank lists to store waiting times
    wait_or = []
    wait_rec = []
    # track the time the ORs are being used to calculate utilization later
    or_busy_within_day = np.zeros(n_or)

    overflow_count = 0 # a count for how many surgeries cannot be scheduled before the day ends
    scheduled_count = 0 # count for how many surgeries are scheduled before the day ends

    for i in range(len(arrivals)): #loop for each patient arrival
        arrival = arrivals[i] #generate arrival time
        s_time = surgeries[i] #generate surgery time
        r_time = recoveries[i] #generate recovery time

        # recovery room scheduling -> Must happen before OR is scheduled
        # -> OR cant be assigned without recovery bed available
        idx_rec = np.argmin(rec_available)  # assign patient to earliest available bed

        # OR scheduling
        idx_or = np.argmin(or_available)  # assign patient to first available OR
        # surgery start: arrival time, first available OR, or first available Recovery Bed
        start_or = max(arrival, or_available[idx_or], rec_available[idx_rec])  
        finish_or = start_or + s_time  # compute the finishing time

        # if the next start time would be after the day is over, add to overflow count
        if start_or >= day_length:
            overflow_count += 1
            continue
        # If not, the next surgery can be scheduled
        scheduled_count += 1
        wait_or.append(start_or - arrival) #append waiting time to list (surgery start time - arrival time)

        # Calculate how much the OR is busy, within the 12-hour period
        busy_within_day = max(0.0, min(finish_or, day_length) - start_or)
        or_busy_within_day[idx_or] += busy_within_day

        # Update availability
        or_available[idx_or] = finish_or

        # Recovery room scheduling
        # Recovery time begins after surgery or whenever bed is available
        start_rec = max(finish_or, rec_available[idx_rec])  # Should alwasys be the same number, keeping logic

        finish_rec = start_rec + r_time  # calculate when recovery time finishes
        # wait should always be 0 with current algorithm
        wait_rec.append(start_rec - finish_or)  # calculate how long patient waits for bed after surgery
        rec_available[idx_rec] = finish_rec  # update recovery bed availability

    # calculate performance metrics for the day
    total_day_capacity = n_or * day_length
    utilization = np.sum(or_busy_within_day) / total_day_capacity
    overflow_prob = overflow_count / len(arrivals)

    #convert to arrays for convenience
    wait_or_arr = np.array(wait_or) if len(wait_or) > 0 else np.array([])
    wait_rec_arr = np.array(wait_rec) if len(wait_rec) > 0 else np.array([])

    return wait_or_arr, wait_rec_arr, utilization, overflow_prob, scheduled_count


#simulation parameters
n_samples = 18000  # number of patients in one day
n_or = 3  # number of ORs
n_recovery = 2  # number of recovery beds
day_length = 12 * 60  # 12 hours in minutes
arrival_lambda = 15  # average patient arrivals per hour

# generate patient arrival times
arrival_times = np.cumsum(np.random.exponential(scale=60 / arrival_lambda, size=n_samples))

# sample surgery durations using Metropolis-Hastings
surgery_durations = sample_surgery_durations_mh(
    n_samples=n_samples,
    x0=150,
    domain=(30, 300),
    proposal_std=40,
    burn_in=1000,
    thinning=5,
    random_seed=42
)

# sample recovery durations based on surgery durations
recovery_durations = np.array([recovery_time(s) for s in surgery_durations])

# run the simulation
wait_or, wait_rec, or_util, overflow_prob, scheduled_count = simulate_with_wait_arrays(
    arrival_times, surgery_durations, recovery_durations, n_or, n_recovery, day_length
)

#calculate average waiting time
avg_or_wait = np.nan if wait_or.size == 0 else np.mean(wait_or)
avg_rec_wait = np.nan if wait_rec.size == 0 else np.mean(wait_rec)

print(f"Scheduled patients (started before day end): {scheduled_count}/{n_samples}")
print(f"Average OR Wait: {avg_or_wait:.2f} minutes")
print(f"Average Recovery Wait: {avg_rec_wait:.2f} minutes")
print(f"OR Utilization (within day window): {or_util * 100:.2f}%")
print(f"Probability of Overflow (couldn't start): {overflow_prob * 100:.2f}%")

#plot results from simulation
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

#histogram to show how MHMH performed
plt.hist(surgery_durations, bins=40, color='steelblue')
plt.title("Distribution of Surgery Durations (MH Samples)")
plt.xlabel("Minutes")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

#scatterplot of surgery duration vs recovery duration
plt.scatter(surgery_durations, recovery_durations, alpha=0.5)
plt.title("Surgery Duration vs Recovery Duration")
plt.xlabel("Surgery Duration (min)")
plt.ylabel("Recovery Duration (min)")
plt.grid(True)
plt.show()


plt.plot(surgery_durations[:180])
plt.title("MH Trace Plot (180 samples)")
plt.xlabel("Iteration")
plt.ylabel("Sample Value")
plt.show()
