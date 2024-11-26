import numpy as np
import cupy as cp
import time

# CPU simulation function
def simulate_on_cpu(start_states, num_steps):
    states_cpu = np.copy(start_states)
    for _ in range(num_steps):
        states_cpu = states_cpu * np.random.rand(len(states_cpu))
    return states_cpu

# GPU simulation function
def simulate_on_gpu(start_states, num_steps):
    states_gpu = cp.asarray(start_states)
    for _ in range(num_steps):
        states_gpu *= cp.random.rand(len(states_gpu))
    return cp.asnumpy(states_gpu)

# Function to run the full simulation
def run_simulation(total_samples, num_steps, final_steps, s_size, interval, fig, ax1, ax2):
    step_counts = []
    cpu_times = []
    gpu_times = []

    # Function to update the plots in each frame
    def update(frame):
        current_step = num_steps + frame * s_size
        start_states = np.random.rand(total_samples)

        # Time CPU simulation
        start_time = time.time()
        results_cpu = simulate_on_cpu(start_states, current_step)
        cpu_time = time.time() - start_time

        # Time GPU simulation
        start_time = time.time()
        results_gpu = simulate_on_gpu(start_states, current_step)
        gpu_time = time.time() - start_time

        # Append data for time analysis
        step_counts.append(current_step)
        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)

        # Clear and update the histogram plot
        ax1.clear()
        ax1.hist(results_cpu, bins=30, alpha=0.5, label="CPU", color='blue', edgecolor='black', linewidth=1.2, histtype='stepfilled')
        ax1.hist(results_gpu, bins=30, alpha=0.5, label="GPU", color='red', edgecolor='black', linewidth=1.2, linestyle='--', fill=False)
        ax1.set_title(f"Monte Carlo Simulation Results (Steps: {current_step})")
        ax1.set_xlabel("State Value")
        ax1.set_ylabel("Frequency")
        ax1.legend(loc="upper right")

        # Update time analysis plot
        ax2.clear()
        ax2.plot(step_counts, cpu_times, label="CPU Time", color='blue', marker='o')
        ax2.plot(step_counts, gpu_times, label="GPU Time", color='red', marker='s')
        ax2.legend()
        ax2.set_title("Simulation Time vs. Steps")
        ax2.set_xlabel("Number of Steps")
        ax2.set_ylabel("Time (seconds)")

        # Print debug information
        mean_cpu = np.mean(results_cpu)
        mean_gpu = np.mean(results_gpu)
        print(f"Frame={frame + 1}: {current_step} simulation steps completed.")
        print(f"CPU Time: {cpu_time:.2e} seconds, GPU Time: {gpu_time:.2e} seconds")
        print(f"\n")

    return update
