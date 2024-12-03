# Input file for simulation parameters

# -----------------------------------------------------------------------

# Parameters for test optimization

total_samples = 1000000  # Total number of samples in the simulation
num_steps = 1            # Initial number of steps
final_steps = 500        # Total steps for the simulation
s_size = 1               # Step size increment
interval = 500           # Interval between frames (ms)

# -----------------------------------------------------------------------

# Parameters for Stochastic Process

grid_size = (250, 250, 250) # 3D grid dimensions
num_particles = 3000     # Number of particles
num_steps = 100          # Number of simulation steps
simulation_type = 0      # 0: For CPU Simulation
                         # 1: For GPU Acceleration
                         
# -----------------------------------------------------------------------
