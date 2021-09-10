import numpy as np
import cupy as cp
import grid as g
import basis as b
import fluxes as fx
import timestep as ts
import plotter as my_plt
import elliptic as ell

# import matplotlib.pyplot as plt

# Parameters
order = 8
res_x, res_y, res_z = 32, 32, 32
final_time, write_time = 4.0e-1, 1.0e-2
nu = 1.0e-2
plot_ic = True
lobatto = False

# Build basis
orders = np.array([order, order, order])
print('\nInitializing basis')
basis = b.Basis3D(orders, lobatto=lobatto)

# Initialize grids
print('\nInitializing grids, flux, and time-stepper')
L = 2.0 * np.pi
lows, highs = np.array([-L/2.0, -L/2.0, -L/2.0]), np.array([L/2.0, L/2.0, L/2.0])
resolutions, resolutions_ghosts = np.array([res_x, res_y, res_z]), np.array([res_x+2, res_y+2, res_z+2])
grids = g.Grid3D(basis=basis, lows=lows, highs=highs, resolutions=resolutions)

# Initialize flux and stepper
# dg_flux = fx.DGFlux(resolutions=resolutions_ghosts, orders=orders)
dg_flux = fx.Spectral(nu=nu)  # resolutions=resolutions_ghosts, orders=orders)
stepper = ts.Stepper(time_order=3, space_order=order, write_time=write_time, final_time=final_time)

# Time info
final_time, write_time = 1.0, 0.1

# Initialize vector
print('\nInitializing velocity and pressure fields')
velocity = g.Vector(resolutions=resolutions_ghosts, orders=orders)
velocity.initialize(grids=grids)

# Compute Poisson problem
# source = g.Scalar(resolutions=resolutions_ghosts, orders=orders)
# source.initialize(grids=grids)
poisson = ell.Elliptic(grids=grids)
poisson.pressure_solve(velocity=velocity, grids=grids)
# max_p = cp.amax(poisson.pressure.arr)
# plotter = my_plt.Plotter3D(grids=grids)
# plotter.scalar_contours3d(scalar=source, contours=[-0.75 * max_p, 0.75 * max_p])
# quit()

# Visualize
if plot_ic:
    print('\nVisualizing initial velocity field')
    plotter = my_plt.Plotter3D(grids=grids)
    max_p = cp.amax(poisson.pressure.arr)
    plotter.scalar_contours3d(scalar=poisson.pressure, contours=[-0.75 * max_p, 0.75 * max_p])
    plotter.velocity_magnitude_contours3d(velocity=velocity, contours=[0.1, 0.5, 0.8, 1.5, 2.0, 3.0, 5.0, 8.0])
    # plotter.vector_contours3d(vector=velocity, contours=[-0.25, 0.25], component=0)
    # plotter.vector_contours3d(vector=velocity, contours=[-0.25, 0.25], component=1)
    plotter.streamlines3d(vector=velocity)

# Begin main loop
stepper.main_loop(vector=velocity, basis=basis, elliptic=poisson, grids=grids, dg_flux=dg_flux)

if plot_ic:
    print('\nVisualizing final state')
    plotter = my_plt.Plotter3D(grids=grids)
    max_p = cp.amax(poisson.pressure.arr)
    min_p = cp.amin(poisson.pressure.arr)
    # print(str(min_p) + ' ' + str(max_p))
    plotter.scalar_contours3d(scalar=poisson.pressure, contours=[0.75 * min_p, 0, 0.75 * max_p])
    plotter.velocity_magnitude_contours3d(velocity=velocity, contours=[0.1, 0.5, 0.8, 1.5, 2.0, 3.0, 5.0, 8.0])
    # plotter.vector_contours3d(vector=velocity, contours=[-0.25, 0.25], component=0)
    # plotter.vector_contours3d(vector=velocity, contours=[-0.25, 0.25], component=1)
    plotter.streamlines3d(vector=velocity)
