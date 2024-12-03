import sys
import gc
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from allocation import allocate_data  # Import the allocation function
from mc_simulation import monte_carlo_kernel_cpu
from mc_simulation import monte_carlo_kernel
from mc_simulation import monte_carlo_kernel_test
from mc_simulation import run_simulation
from matplotlib.animation import FuncAnimation
from config import *  # Import configuration settings
from numba import cuda

# -----------------------------------------------------------------------

# Add the 'input' directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'input'))
# Now import from the 'input' directory
from input import *  # Import simulation parameters from input.py

# -----------------------------------------------------------------------

# Ensure the output directory exists
output_dir = "../output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ------------------------------------------------------------------------

# Set up the figure for two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.4})
ax1.set_title("Monte Carlo Simulation Results")
ax1.set_xlabel("State Value")
ax1.set_ylabel("Frequency")
ax2.set_title("Simulation Time vs. Steps")
ax2.set_xlabel("Number of Steps")
ax2.set_ylabel("Time (seconds)")

# Run the simulation
update = run_simulation(total_samples, num_steps, final_steps, s_size, interval, fig, ax1, ax2)

# Create the animation
ani = FuncAnimation(fig, update, frames=int((final_steps - num_steps) / s_size), interval=interval)

# Absolute path for the output file
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'state_file_linear.mp4')

# Debugging output path
print(f"Saving the animation to: {output_path}")

# Save the animation as an MP4 file
ani.save(output_path, writer="ffmpeg", fps=20)

# Explicitly close the plot and finish
plt.close(fig)
print(f"Animation saved at: {output_path}")

# -----------------------------------------------------------------------------

# ------------------------- GPU PRE-REQUISITIES -------------------------------

grid, particle_positions, particle_directions, particle_energies = \
    allocate_data(grid_size, num_particles)

grid_gpu = cuda.to_device(grid)
particle_positions_gpu = cuda.to_device(particle_positions)
particle_directions_gpu = cuda.to_device(particle_directions)
particle_energies_gpu = cuda.to_device(particle_energies)
grid_size_gpu = cuda.to_device(grid_size)
num_steps_gpu = cuda.to_device(num_steps)

# Run the Monte Carlo kernel on the CPU
positions_over_time, energies_over_time, grids_over_time, positions_over_time_cpu_gpu, energies_over_time_cpu_gpu, grids_over_time_cpu_gpu = monte_carlo_kernel(
    grid, particle_positions, particle_directions, particle_energies, grid_gpu,particle_positions_gpu,particle_directions_gpu,particle_energies_gpu,grid_size_gpu,num_steps_gpu
)

if simulation_type==1:
    positions_over_time = positions_over_time_cpu_gpu
    energies_over_time = energies_over_time_cpu_gpu
    grids_over_time = grids_over_time_cpu_gpu


# ---------------------- SMALL TEST (NOT PART OF MAIN CODE) ---------------
# positions_over_time, energies_over_time, grids_over_time, positions_over_time_cpu_gpu = monte_carlo_kernel_test(
#     grid, particle_positions, particle_directions, particle_energies, particle_positions_gpu
# )
# print("positions_over_time_cpu_gpu: PRINTING")
# print(positions_over_time_cpu_gpu[0:4,1,0:3])
# ---------------------- SMALL TEST (NOT PART OF MAIN CODE) ---------------

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

# Particle Movement Visualization
fig1, ax1 = plt.subplots(figsize=(10, 8))

# Setting axis limits with a bit more padding
ax1.set_xlim(0, grid_size[0] + 0)  # Adding 10 units padding for better visibility
ax1.set_ylim(0, grid_size[1] + 0)

# Create a scatter plot object with more visually appealing points and features
scat = ax1.scatter([], [], s=30, c='cyan', edgecolor='black', label="Particles", alpha=0.8)

# Add a legend for clarity
ax1.legend(loc='upper right', fontsize=12, frameon=False)

# Set a bold and larger title
ax1.set_title("Particle Movement Over Time", fontsize=18, fontweight='bold', color='darkgreen')

# Set axis labels with larger font size
ax1.set_xlabel('X Position (mm)', fontsize=14, color='darkblue')
ax1.set_ylabel('Y Position (mm)', fontsize=14, color='darkblue')

# Add gridlines with slight transparency and dashed style for clarity
ax1.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

# Update function for animation
def update_particle(frame):
    # Update particle scatter plot positions
    scat.set_offsets(positions_over_time[frame][:, :2])
    return scat,

# Create the animation with improved interval and smoother transitions
ani1 = FuncAnimation(fig1, update_particle, frames=num_steps, interval=50, blit=True)

# Adjust layout to prevent overlapping elements
# plt.tight_layout()

# Display the plot
# plt.show()

# -----------------------------------------------------------------------------

# Visualization of energy deposition with customization
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Start with the frame showing energy deposition
energy_img = ax2.imshow(grids_over_time[min(1, num_steps-1)][:, :, grid_size[2] // 2],
                         cmap='inferno',  # Change color map to 'inferno' for better contrast
                         origin='lower', 
                         interpolation='bicubic',  # Smoother interpolation
                         alpha=0.9)  # Increased opacity for better visibility

# Adding color bar for better interpretation of energy levels
cbar = plt.colorbar(energy_img, ax=ax2, orientation='vertical', shrink=0.8)
cbar.set_label('Energy Deposition (arbitrary units)', fontsize=12)

# Setting title and labels with improved font size and style
ax2.set_title("Energy Deposition Over Time", fontsize=16, fontweight='bold', color='darkblue')
ax2.set_xlabel('X Position (mm)', fontsize=12, color='darkred')
ax2.set_ylabel('Y Position (mm)', fontsize=12, color='darkred')

# Adding a grid for better readability of positions
ax2.grid(True, color='gray', linestyle='--', linewidth=0.5)

# Adjusting the aspect ratio and limits to ensure the image is displayed correctly
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(0, grid_size[0])
ax2.set_ylim(0, grid_size[1])

# Updating the image for each frame
def update_energy(frame):
    energy_img.set_array(grids_over_time[frame][:, :, grid_size[2] // 2])
    return energy_img,

# Animation
ani2 = FuncAnimation(fig2, update_energy, frames=num_steps, interval=50, blit=True)

# -----------------------------------------------------------------------------

plt.tight_layout()  # Adjust the layout to avoid overlapping labels
plt.show()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Absolute path for the output file
output_path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'pos_mc_simul.mp4')
output_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'eg_mc_simul.mp4')

# Debugging output path
print(f"Saving the animation to: {output_path1}")

# Save the animation as an MP4 file
ani1.save(output_path1, writer="ffmpeg", fps=20)
ani2.save(output_path2, writer="ffmpeg", fps=20)

# Explicitly close the plot and finish
plt.close(fig1)
plt.close(fig2)
print(f"Animation saved at: {output_path1}")