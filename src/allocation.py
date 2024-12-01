# allocation.py
import numpy as np

def allocate_data(grid_size, num_particles):
    # Allocate grid and particle data
    grid = np.zeros(grid_size, dtype=np.float32)
    
    # Random positions for particles within the grid dimensions
    particle_positions = np.random.rand(num_particles, 3) * np.array(grid_size)
    
    # Random directions for particles, normalized to unit vectors
    particle_directions = np.random.randn(num_particles, 3)
    particle_directions /= np.linalg.norm(particle_directions, axis=1, keepdims=True)
    
    # Initial energy for each particle
    particle_energies = np.ones(num_particles, dtype=np.float32) * 10.0
    
    return grid, particle_positions, particle_directions, particle_energies
