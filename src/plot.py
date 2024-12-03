import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation

# Sample data (replace these with your actual results)
grid_sizes = [50, 100, 150, 200, 250]  # Grid sizes in the X direction
cpu_times = [3.9975, 4.0048, 4.0549, 4.9314, 5.8057]  # Time taken by CPU in seconds
gpu_times = [1.0525, 1.1039, 1.4423, 1.9987, 4.2523]  # Time taken by GPU in seconds

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(40, 260)
ax.set_ylim(0, 10)
ax.set_xlabel('Grid Size (Cubic Shaped)', fontsize=14, weight='bold', color='darkgreen')
ax.set_ylabel('Time Taken (seconds)', fontsize=14, weight='bold', color='darkgreen')
ax.set_title('Accelerating Simulations: A CPU vs GPU Perspective', fontsize=16, weight='bold', color='darkblue')
ax.grid(which='both', linestyle='--', linewidth=0.7, alpha=0.7)
ax.legend(loc='upper left', fontsize=12, fancybox=True, shadow=True)

# Initialize the lines
cpu_line, = ax.plot([], [], marker='o', linestyle='-', color='red', label='CPU Time', linewidth=2, markersize=8)
gpu_line, = ax.plot([], [], marker='s', linestyle='--', color='blue', label='GPU Time', linewidth=2, markersize=8)

# Add legend
ax.legend(loc='upper left', fontsize=12, fancybox=True, shadow=True)

# Animation initialization
def init():
    cpu_line.set_data([], [])
    gpu_line.set_data([], [])
    return cpu_line, gpu_line

# Animation update function
def update(frame):
    # Update CPU and GPU lines incrementally
    cpu_line.set_data(grid_sizes[:frame + 1], cpu_times[:frame + 1])
    gpu_line.set_data(grid_sizes[:frame + 1], gpu_times[:frame + 1])
    return cpu_line, gpu_line

# Create animation
ani = FuncAnimation(fig, update, frames=len(grid_sizes), init_func=init, blit=True, interval=500)

# Ensure the output directory exists
output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

# Save the animation as a GIF
gif_path = os.path.join(output_dir, 'animation.gif')
ani.save(gif_path, writer='imagemagick', fps=2)
print(f"Animation saved to {gif_path}")

# Save the final static figure as a PNG
png_path = os.path.join(output_dir, 'figure.png')
fig.tight_layout()
fig.savefig(png_path, dpi=300)
print(f"Static plot saved to {png_path}")

# Show the animation
plt.show()
