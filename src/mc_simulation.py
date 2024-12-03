import numba
import cupy as cp
import numpy as np
import time
import sys
import os
from numba import cuda
# -----------------------------------------------------------------------

# Add the 'input' directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'input'))
# Now import from the 'input' directory
from input import *  # Import simulation parameters from input.py

# -----------------------------------------------------------------------

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
            position = position + direction
            position = np.clip(position, 0, np.array(grid_size) - 1)  # Keep within grid bounds

            # --------- UPDATE GRID ARRAY WITH ENERGY
            # Update grid with the energy deposited by the particle
            grid[tuple(position.astype(int))] += energy

            # --------- UPDATE POS
            # Update the particle's position
            particle_positions[i] = position

            # --------- UPDATE EG
            # Simulate energy loss (simple model)
            particle_energies[i] *= 0.98  # Particle loses energy each step

    return positions_over_time, energies_over_time, grids_over_time

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# 
# ---------------------------------- A SMALL TESTING FUNCTION (NOT PART OF MAIN CODE) --------------------------------
# 
# --------------------------------------------------------------------------------------------------------------------

def monte_carlo_kernel_test(grid, particle_positions, particle_directions, particle_energies, particle_positions_gpu):
    positions_over_time = []  # To store particle positions for animation
    energies_over_time = []   # To store particle energies for visualization
    grids_over_time = []      # To store grid states for visualization

    positions_over_time_cpu_gpu = []  # To store particle positions for animation

    # Initialize time-dimensioned arrays on GPU
    positions_over_time_gpu = cuda.device_array((num_steps, num_particles, 3), dtype=np.float32)

    if(simulation_type  == 1):                                                          # type = 1: GPU
        # # START GPU SIMULATION TIME
        # start_time_gpu = time.time()  # Start time

        # Kernel launch configuration
        threads_per_block = 256
        blocks_per_grid = (particle_positions_gpu.shape[0] + threads_per_block - 1) // threads_per_block

        # Create a host array to store intermediate values before copying to GPU
        positions_over_time_host = np.zeros((num_steps, 2, 3), dtype=np.float32)

        for step in range(5):
            # --------- STORE PARTICLE INFO
            positions_over_time_gpu[step,:,:] = particle_positions_gpu

            # Launch kernel for a single simulation step
            small_test[blocks_per_grid, threads_per_block](
                particle_positions_gpu)

        # Transfer updated host array to the GPU
        # positions_over_time_gpu.copy_to_device(positions_over_time_host)

        positions_over_time_cpu_gpu = positions_over_time_gpu.copy_to_host()  # To store particle positions for animation
    
    if(simulation_type  == 0):                                                          # type = 0: CPU
        # START CPU SIMULATION TIME
        start_time_cpu = time.time()  # Start time

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
                position = position + direction
                position = np.clip(position, 0, np.array(grid_size) - 1)  # Keep within grid bounds

                # --------- UPDATE GRID ARRAY WITH ENERGY
                # Update grid with the energy deposited by the particle
                grid[tuple(position.astype(int))] += energy

                # --------- UPDATE POS
                # Update the particle's position
                particle_positions[i] = position

                # --------- UPDATE EG
                # Simulate energy loss (simple model)
                particle_energies[i] *= 0.98  # Particle loses energy each step

        # END CPU SIMULATION TIME
        end_time_cpu = time.time()  # End time

        # TOTAL TIME
        elapsed_time = end_time_cpu - start_time_cpu
        print(f"CPU simulation took {elapsed_time:.4f} seconds.")  # Print execution time

            
    return positions_over_time, energies_over_time, grids_over_time, positions_over_time_cpu_gpu

# --------------------------------------------------------------------------------------

@cuda.jit
def small_test(particle_positions_gpu):
    idx = cuda.grid(1)
    if idx >= particle_positions_gpu.shape[0]:
        return
    
    # Define a local array (size must match particle_positions_gpu's row size)
    # position = cuda.local.array(shape=(3,), dtype=numba.float32)  # Replace 3 with actual row size

    position = particle_positions_gpu[idx]

    # --------- UPDATE POS AND BOUND THEM
    # Update the local array: position
    position[:] = 5.6
    # for i in range(position.shape[0]):
    #     position[i] = 5.6

    # --------- UPDATE POS
    # Update the particle's position
    # Copy data back to particle_positions_gpu
    for i in range(particle_positions_gpu.shape[1]):
        particle_positions_gpu[idx, i] = position[i]

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

def monte_carlo_kernel(grid, particle_positions, particle_directions, particle_energies, grid_gpu,particle_positions_gpu,particle_directions_gpu,particle_energies_gpu,grid_size_gpu,num_steps_gpu):
    positions_over_time = []  # To store particle positions for animation
    energies_over_time = []   # To store particle energies for visualization
    grids_over_time = []      # To store grid states for visualization

    positions_over_time_cpu_gpu = []  # To store particle positions for animation
    energies_over_time_cpu_gpu = []   # To store particle energies for visualization
    grids_over_time_cpu_gpu = []      # To store grid states for visualization

    # Initialize time-dimensioned arrays on GPU
    positions_over_time_gpu = cuda.device_array((num_steps, num_particles, 3), dtype=np.float32)
    energies_over_time_gpu = cuda.device_array((num_steps, num_particles), dtype=np.float32)
    grids_over_time_gpu = cuda.device_array((num_steps, *grid_size), dtype=np.float32)
    
    if(simulation_type == 0):                                                          # type = 0: CPU
        # START CPU SIMULATION TIME
        start_time_cpu = time.time()  # Start time

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
                position = position + direction
                position = np.clip(position, 0, np.array(grid_size) - 1)  # Keep within grid bounds

                # --------- UPDATE GRID ARRAY WITH ENERGY
                # Update grid with the energy deposited by the particle
                grid[tuple(position.astype(int))] += energy

                # --------- UPDATE POS
                # Update the particle's position
                particle_positions[i] = position

                # --------- UPDATE EG
                # Simulate energy loss (simple model)
                particle_energies[i] *= 0.98  # Particle loses energy each step

        # END CPU SIMULATION TIME
        end_time_cpu = time.time()  # End time

        # TOTAL TIME
        elapsed_time = end_time_cpu - start_time_cpu
        print(f"CPU simulation took {elapsed_time:.4f} seconds.")  # Print execution time

    if(simulation_type == 1):                                                          # type = 1: GPU
        # START GPU SIMULATION TIME
        start_time_gpu = time.time()  # Start time

        # Kernel launch configuration
        threads_per_block = 256
        blocks_per_grid = (particle_positions_gpu.shape[0] + threads_per_block - 1) // threads_per_block

        # Synchronize to ensure accurate timing
        cuda.synchronize()

        for step in range(num_steps):
            # --------- STORE PARTICLE INFO
            # Store current state into time-dimensioned arrays
            positions_over_time_gpu[step, :, :] = particle_positions_gpu
            energies_over_time_gpu[step, :] = particle_energies_gpu
            grids_over_time_gpu[step, :, :, :] = grid_gpu

            # Launch kernel for a single simulation step
            particle_pos_energy_grid_update[blocks_per_grid, threads_per_block](
                grid_gpu, particle_positions_gpu, particle_directions_gpu,particle_energies_gpu,grid_size_gpu,num_steps_gpu)

        positions_over_time_cpu_gpu = positions_over_time_gpu.copy_to_host()  # To store particle positions for animation
        energies_over_time_cpu_gpu = energies_over_time_gpu.copy_to_host()    # To store particle energies for visualization
        grids_over_time_cpu_gpu = grids_over_time_gpu.copy_to_host()          # To store grid states for visualization

        # END GPU SIMULATION TIME
        cuda.synchronize()  # Ensure all GPU tasks are complete
        end_time_gpu = time.time()  # End time

        # TOTAL TIME
        elapsed_time = end_time_gpu - start_time_gpu
        print(f"GPU simulation took {elapsed_time:.4f} seconds.")  # Print execution time
            
    return positions_over_time, energies_over_time, grids_over_time, positions_over_time_cpu_gpu, energies_over_time_cpu_gpu, grids_over_time_cpu_gpu


@cuda.jit
def particle_pos_energy_grid_update(grid_gpu, particle_positions_gpu, particle_directions_gpu, particle_energies_gpu, grid_size_gpu, num_steps_gpu):
    idx = cuda.grid(1)
    if idx >= particle_positions_gpu.shape[0]:
        return
    
    # --------- LOCAL VARIABLE TO STORE POS, DIR, EG    
    # Load particle state
    position = particle_positions_gpu[idx]
    direction = particle_directions_gpu[idx]
    energy = particle_energies_gpu[idx]

    # --------- UPDATE POS AND BOUND THEM
    # Update position
    for i in range(position.shape[0]):
        position[i] += direction[i]
    for i in range(3):  # Handle boundaries (clipping)
        if position[i] < 0:
            position[i] = 0
        elif position[i] >= grid_size_gpu[i]:
            position[i] = grid_size_gpu[i] - 1

    # --------- UPDATE GRID ARRAY WITH ENERGY
    # Deposit energy into the grid at the current particle position
    cuda.atomic.add(grid_gpu, (int(position[0]), int(position[1]), int(position[2])), energy * 0.01)

    # --------- UPDATE POS
    # Update the particle's position
    for i in range(particle_positions_gpu.shape[1]):
        particle_positions_gpu[idx, i] = position[i]

    # --------- UPDATE EG
    # Simulate energy loss (simple model)
    energy *= 0.98
    particle_energies_gpu[i] *= 0.98  # Particle loses energy each step




 