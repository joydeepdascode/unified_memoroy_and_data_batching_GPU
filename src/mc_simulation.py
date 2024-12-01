import numba
from numba import cuda
import cupy as cp
import numpy as np
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

# # Kernel function
# @cuda.jit
# def monte_carlo_kernel(data, result):
#     idx = cuda.grid(1)  # Get the thread index in a 1D grid
#     if idx < data.size:  # Ensure we're within bounds
#         result[idx] = data[idx] + 1  # Simulating some Monte Carlo-like operation

# # Function to run the simulation on the GPU
# def run_simulation(data):
#     # Allocate memory for result
#     result = np.zeros_like(data)

#     # Configure threads and blocks
#     threads_per_block = 32
#     blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block

#     # Transfer data to device (GPU)
#     d_data = cuda.to_device(data)
#     d_result = cuda.to_device(result)

#     # Launch the kernel
#     monte_carlo_kernel[blocks_per_grid, threads_per_block](d_data, d_result)

#     # Copy the result back to the host (CPU)
#     d_result.copy_to_host(result)

#     return result


def monte_carlo_kernel_cpu(grid, particle_positions, particle_directions, particle_energies, grid_size, num_steps):
    num_particles = particle_positions.shape[0]
    positions_over_time = []  # To store particle positions for animation
    energies_over_time = []   # To store particle energies for visualization
    grids_over_time = []      # To store grid states for visualization

    for step in range(num_steps):
        # --------- STORE PARTICLE INFO
        # Store particle positions and energies for this time step
        positions_over_time.append(np.copy(particle_positions))
        energies_over_time.append(np.copy(particle_energies))
        grids_over_time.append(np.copy(grid))  # Store a snapshot of the grid

        for i in range(num_particles):
            # --------- LOCAL VARIABLE TO STORE POS, DIR, EG
            # Get the current particle position and direction
            position = particle_positions[i]
            direction = particle_directions[i]
            energy = particle_energies[i]

            # --------- UPDATE POS AND BOUND THEM
            # Perform a simple simulation step (random walk and energy deposit)
            new_position = position + direction
            new_position = np.clip(new_position, 0, np.array(grid_size) - 1)  # Keep within grid bounds

            # --------- UPDATE GRID ARRAY WITH ENERGY
            # Update grid with the energy deposited by the particle
            grid[tuple(new_position.astype(int))] += energy

            # --------- UPDATE POS
            # Update the particle's position
            particle_positions[i] = new_position

            # --------- UPDATE EG
            # Simulate energy loss (simple model)
            particle_energies[i] *= 0.98  # Particle loses energy each step

    return positions_over_time, energies_over_time, grids_over_time
