import sys
import gc
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import *  # Import configuration settings
from simulation import run_simulation  # Import the simulation running function

# Add the 'input' directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'input'))

# Now import from the 'input' directory
from input import *  # Import simulation parameters from input.py

# Ensure the output directory exists
output_dir = "../output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Manually trigger garbage collection
gc.collect()

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
