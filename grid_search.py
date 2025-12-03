import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from multiprocessing import Pool, cpu_count
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Define complex surgery duration function
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

# Recovery time based on surgery length
def recovery_time(surgery_length, seed=None):
    if seed is not None:
        np.random.seed(seed)
    base = 60 + 0.25 * surgery_length
    noise = np.random.gamma(shape=2, scale=7)
    return base + noise

# Simulation with overflow allowance
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

        idx_rec = np.argmin(rec_available)
        idx_or = np.argmin(or_available)
        start_or = max(arrival, or_available[idx_or], rec_available[idx_rec])
        finish_or = start_or + s_time

        if start_or >= day_length:
            overflow_count += 1
            continue

        scheduled_count += 1
        wait_or.append(start_or - arrival)
        busy_within_day = max(0.0, min(finish_or, day_length) - start_or)
        or_busy_within_day[idx_or] += busy_within_day
        or_available[idx_or] = finish_or

        start_rec = max(finish_or, rec_available[idx_rec])
        finish_rec = start_rec + r_time
        wait_rec.append(start_rec - finish_or)
        rec_available[idx_rec] = finish_rec

    total_day_capacity = n_or * day_length
    utilization = np.sum(or_busy_within_day) / total_day_capacity if total_day_capacity > 0 else 0
    overflow_prob = overflow_count / len(arrivals) if len(arrivals) > 0 else 0

    wait_or_arr = np.array(wait_or) if len(wait_or) > 0 else np.array([])
    wait_rec_arr = np.array(wait_rec) if len(wait_rec) > 0 else np.array([])

    return wait_or_arr, wait_rec_arr, utilization, overflow_prob, scheduled_count

# Worker function for parallel processing
def run_single_config(args):
    """Run simulation for a single (n_or, n_recovery) configuration with multiple replications."""
    n_or, n_recovery, n_samples, day_length, arrival_lambda, n_replications = args
    
    avg_waits = []
    utilizations = []
    overflow_probs = []
    throughputs = []
    
    for rep in range(n_replications):
        seed = n_or * 1000 + n_recovery * 100 + rep
        np.random.seed(seed)
        
        # Generate arrivals
        arrival_times = np.cumsum(np.random.exponential(scale=60 / arrival_lambda, size=n_samples))
        
        # Sample surgery durations
        surgery_durations = sample_surgery_durations_mh(
            n_samples=n_samples, x0=150, domain=(30, 300),
            proposal_std=40, burn_in=100, thinning=5, random_seed=seed
        )
        
        # Sample recovery durations
        np.random.seed(seed + 5000)
        recovery_durations = np.array([recovery_time(s) for s in surgery_durations])
        
        # Run simulation
        wait_or, wait_rec, or_util, overflow_prob, scheduled = simulate_with_wait_arrays(
            arrival_times, surgery_durations, recovery_durations, n_or, n_recovery, day_length
        )
        
        avg_wait = np.mean(wait_or) if len(wait_or) > 0 else 0
        avg_waits.append(avg_wait)
        utilizations.append(or_util)
        overflow_probs.append(overflow_prob)
        throughputs.append(scheduled)
    
    return (n_or, n_recovery, 
            np.mean(avg_waits), 
            np.mean(utilizations), 
            np.mean(overflow_probs),
            np.mean(throughputs))

def main():
    # Simulation parameters
    n_samples = 180
    day_length = 12 * 60
    arrival_lambda = 15
    n_replications = 10  # Multiple replications for stability
    
    # Grid search parameters
    min_range, max_range = 1, 31
    values = max_range - min_range
    or_range = range(min_range, max_range)
    recovery_range = range(min_range, max_range)
    
    # Create all configurations
    configs = [
        (n_or, n_rec, n_samples, day_length, arrival_lambda, n_replications)
        for n_or, n_rec in product(or_range, recovery_range)
    ]
    
    print(f"Running grid search over {len(configs)} configurations...")
    print(f"Using {cpu_count()} CPU cores for parallel processing")
    
    # Run parallel simulations
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(run_single_config, configs)
    
    print("Simulations complete. Processing results...")
    
    # Organize results into grids
    temp = (values, values)
    or_grid = np.zeros(temp)
    rec_grid = np.zeros(temp)
    wait_grid = np.zeros(temp)
    util_grid = np.zeros(temp)
    overflow_grid = np.zeros(temp)
    throughput_grid = np.zeros(temp)
    
    for result in results:
        n_or, n_rec, avg_wait, util, overflow, throughput = result
        i, j = n_or - 1, n_rec - 1
        or_grid[i, j] = n_or
        rec_grid[i, j] = n_rec
        wait_grid[i, j] = avg_wait
        util_grid[i, j] = util * 100  # Convert to percentage
        overflow_grid[i, j] = overflow * 100  # Convert to percentage
        throughput_grid[i, j] = throughput
    
    # Create figure with 4 contour plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('OR & Recovery Bed Grid Search Analysis\n(180 patients/day, 12-hour window)', 
                 fontsize=14, fontweight='bold')
    
    X, Y = np.meshgrid(range(min_range, max_range), range(min_range, max_range))
    
    # Plot 1: Average OR Wait Time
    ax1 = axes[0, 0]
    cs1 = ax1.contourf(X, Y, wait_grid.T, levels=20, cmap='RdYlGn_r')
    ax1.contour(X, Y, wait_grid.T, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    cbar1 = fig.colorbar(cs1, ax=ax1)
    cbar1.set_label('Minutes')
    ax1.set_xlabel('Number of ORs')
    ax1.set_ylabel('Number of Recovery Beds')
    ax1.set_title('Average OR Wait Time')
    ax1.set_xticks(range(2, max_range, 2))
    ax1.set_yticks(range(2, max_range, 2))
    
    # Plot 2: OR Utilization
    ax2 = axes[0, 1]
    cs2 = ax2.contourf(X, Y, util_grid.T, levels=20, cmap='RdYlGn')
    ax2.contour(X, Y, util_grid.T, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    cbar2 = fig.colorbar(cs2, ax=ax2)
    cbar2.set_label('Percentage')
    ax2.set_xlabel('Number of ORs')
    ax2.set_ylabel('Number of Recovery Beds')
    ax2.set_title('OR Utilization (%)')
    ax2.set_xticks(range(2, max_range, 2))
    ax2.set_yticks(range(2, max_range, 2))
    
    # Plot 3: Overflow Probability
    ax3 = axes[1, 0]
    cs3 = ax3.contourf(X, Y, overflow_grid.T, levels=20, cmap='RdYlGn_r')
    ax3.contour(X, Y, overflow_grid.T, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    cbar3 = fig.colorbar(cs3, ax=ax3)
    cbar3.set_label('Percentage')
    ax3.set_xlabel('Number of ORs')
    ax3.set_ylabel('Number of Recovery Beds')
    ax3.set_title('Overflow Probability (% Not Scheduled)')
    ax3.set_xticks(range(2, max_range, 2))
    ax3.set_yticks(range(2, max_range, 2))
    
    # Plot 4: Throughput
    ax4 = axes[1, 1]
    cs4 = ax4.contourf(X, Y, throughput_grid.T, levels=20, cmap='RdYlGn')
    ax4.contour(X, Y, throughput_grid.T, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    cbar4 = fig.colorbar(cs4, ax=ax4)
    cbar4.set_label('Patients')
    ax4.set_xlabel('Number of ORs')
    ax4.set_ylabel('Number of Recovery Beds')
    ax4.set_title('Daily Throughput (Patients Scheduled)')
    ax4.set_xticks(range(2, max_range, 2))
    ax4.set_yticks(range(2, max_range, 2))
    
    plt.tight_layout()
    plt.savefig('or_grid_search_contour.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("GRID SEARCH SUMMARY")
    print("="*60)
    
    # Find optimal configurations
    min_wait_idx = np.unravel_index(np.argmin(wait_grid), wait_grid.shape)
    max_util_idx = np.unravel_index(np.argmax(util_grid), util_grid.shape)
    min_overflow_idx = np.unravel_index(np.argmin(overflow_grid), overflow_grid.shape)
    max_throughput_idx = np.unravel_index(np.argmax(throughput_grid), throughput_grid.shape)
    
    print(f"\nOptimal Configurations:")
    print(f"  Min Wait Time: {min_wait_idx[0]+1} ORs, {min_wait_idx[1]+1} Recovery Beds "
          f"({wait_grid[min_wait_idx]:.1f} min)")
    print(f"  Max Utilization: {max_util_idx[0]+1} ORs, {max_util_idx[1]+1} Recovery Beds "
          f"({util_grid[max_util_idx]:.1f}%)")
    print(f"  Min Overflow: {min_overflow_idx[0]+1} ORs, {min_overflow_idx[1]+1} Recovery Beds "
          f"({overflow_grid[min_overflow_idx]:.1f}%)")
    print(f"  Max Throughput: {max_throughput_idx[0]+1} ORs, {max_throughput_idx[1]+1} Recovery Beds "
          f"({throughput_grid[max_throughput_idx]:.0f} patients)")
    
    # Find efficient frontier (good balance)
    print(f"\nSample Configurations (OR=5):")
    for rec in [2, 5, 10, 15, 20]:
        print(f"  5 ORs, {rec} Rec: Wait={wait_grid[4, rec-1]:.1f}min, "
              f"Util={util_grid[4, rec-1]:.1f}%, Overflow={overflow_grid[4, rec-1]:.1f}%")
    
    print(f"\nContour plot saved to or_grid_search_contour.png")

if __name__ == "__main__":
    main()