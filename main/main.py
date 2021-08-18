import numpy as np
import cupy as cp
import grid as g
import basis as b
import pyvista as pv
import elliptic as ell

import matplotlib.pyplot as plt

# Parameters
order = 8
res_x, res_y, res_z = 25, 25, 25

# Build basis
orders = np.array([order, order, order])
print('Initializing basis...')
basis = b.Basis3D(orders)

# Initialize grids
L = 2.0 * np.pi
lows, highs = np.array([-L/2.0, -L/2.0, -L/2.0]), np.array([L/2.0, L/2.0, L/2.0])
resolutions, resolutions_ghosts = np.array([res_x, res_y, res_z]), np.array([res_x+2, res_y+2, res_z+2])
print('\nInitializing grids...')
grids = g.Grid3D(basis=basis, lows=lows, highs=highs, resolutions=resolutions)

# Time info
final_time, write_time = 1.0, 0.1

# Initialize variable
source = g.Scalar(resolutions=resolutions_ghosts, orders=orders)
source.initialize(grids=grids)

# Visualize




