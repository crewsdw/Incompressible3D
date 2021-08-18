import pyvista as pv
import cupy as cp
import numpy as np

pv.set_plot_theme("document")


class Plotter3D:
    """
    Plots objects on 3D piecewise (as in DG) grid
    """

    def __init__(self, grids):
        # Build structured grid
        (ix, iy, iz) = (cp.ones(grids.x.res * grids.x.order),
                        cp.ones(grids.y.res * grids.y.order),
                        cp.ones(grids.z.res * grids.z.order))
        (x3, y3, z3) = (outer3(a=grids.x.arr_cp[1:-1, :].flatten(), b=iy, c=iz),
                        outer3(a=ix, b=grids.y.arr_cp[1:-1, :].flatten(), c=iz),
                        outer3(a=ix, b=iy, c=grids.z.arr_cp[1:-1, :].flatten()))
        self.grid = pv.StructuredGrid(x3, y3, z3)

    def scalar_contours3d(self, scalar, contours):
        """
        plot contours of a scalar function f=f(x,y,z) on Plotter3D's grid
        """
        self.grid['.'] = scalar.grid_flatten_gpu_no_ghosts().get().transpose().flatten()
        plot_contours = self.grid.contour(contours)

        # Create plot
        p = pv.Plotter()
        p.add_mesh(plot_contours, cmap='summer', show_scalar_bar=True)
        p.show_grid()
        p.show(auto_close=False)

    def vector_contours3d(self, vector, component, contours):
        self.grid['.'] = vector.grid_flatten_arr_no_ghost()[component, :, :, :].get().transpose().flatten()
        plot_contours = self.grid.contour(contours)

        # Create plot
        p = pv.Plotter()
        p.add_mesh(plot_contours, cmap='summer', show_scalar_bar=True)
        p.show_grid()
        p.show(auto_close=False)

    def streamlines3d(self, vector):
        # set active vectors
        self.grid['vectors'] = np.column_stack(
            tuple(vector.grid_flatten_arr_no_ghost()[idx, :, :, :].get().transpose().flatten()
                  for idx in range(3))
        )
        self.grid.set_active_vectors('vectors')
        streamlines = self.grid.streamlines(source_radius=np.sqrt(2) * np.pi, n_points=500)

        p = pv.Plotter()
        p.add_mesh(streamlines.tube(radius=0.05))
        p.show_grid()
        p.show(auto_close=False)


def outer3(a, b, c):
    """
    Compute outer tensor product of vectors a, b, and c
    :param a: vector a_i
    :param b: vector b_j
    :param c: vector c_k
    :return: tensor a_i b_j c_k as numpy array
    """
    return cp.tensordot(a, cp.tensordot(b, c, axes=0), axes=0).get()
