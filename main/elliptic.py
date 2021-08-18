import numpy as np
import cupy as cp
import grid as g


class Elliptic:
    """
    Class to compute pressure and pressure gradient from pressure Poisson equation.
    Important member objects are Elliptic.pressure (Scalar) and Elliptic.pressure_gradient (Vector) attributes
    """
    def __init__(self, grids):
        # Resolutions
        self.ord = np.array(grids.orders)
        self.res = np.array(grids.res_ghosts)
        grid_tuple = (self.res[0], self.ord[0], self.res[1], self.ord[1], self.res[2], self.ord[2])

        # Pressure Scalar object
        self.pressure = g.Scalar(resolutions=self.res, orders=self.ord)
        self.pressure.arr = cp.zeros(grid_tuple)

        # Pressure gradient Vector object
        self.pressure_gradient = g.Vector(resolutions=self.res, orders=self.ord)
        self.pressure_gradient.arr = cp.zeros((3,) + grid_tuple)

        # spectral 3-grid
        (ix, iy, iz) = (cp.ones_like(grids.x.d_wave_numbers),
                        cp.ones_like(grids.y.d_wave_numbers),
                        cp.ones_like(grids.z.d_wave_numbers))
        self.kr_sq = (outer3(a=grids.x.d_wave_numbers, b=iy, c=iz) ** 2.0 +
                      outer3(a=ix, b=grids.y.d_wave_numbers, c=iz) ** 2.0 +
                      outer3(a=ix, b=iy, c=grids.z.d_wave_numbers) ** 2.0)

    def pressure_solve(self, velocity, grids):
        """
        Solve the pressure Poisson equation given a velocity vector-field, then set pressure and pressure gradient
        :param velocity: Vector object
        :param grids: Grid3D object
        """
        # Compute velocity gradient tensor and its transpose double contraction
        velocity.gradient_tensor(grids=grids)
        velocity.poisson_source()

        # Fourier transform the pressure poisson source
        spectrum = grids.fourier_transform(function=velocity.pressure_source)

        # Determine Poisson solution spectrum, p_tilde = - s_tilde / k**2.0
        poisson_spectrum = -1.0 * cp.nan_to_num(cp.divide(spectrum, self.kr_sq))

        # Inverse transform for the pressure solution and components of pressure gradient
        self.pressure.arr[1:-1, :, 1:-1, :, 1:-1, :] = grids.inverse_transform(spectrum=poisson_spectrum)
        self.pressure_gradient.arr[0, 1:-1, :, 1:-1, :, 1:-1, :] = (
            grids.inverse_transform(spectrum=cp.multiply(1j * grids.x.d_wave_numbers[:, None, None], poisson_spectrum))
        )
        self.pressure_gradient.arr[1, 1:-1, :, 1:-1, :, 1:-1, :] = (
            grids.inverse_transform(spectrum=cp.multiply(1j * grids.y.d_wave_numbers[None, :, None], poisson_spectrum))
        )
        self.pressure_gradient.arr[2, 1:-1, :, 1:-1, :, 1:-1, :] = (
            grids.inverse_transform(spectrum=cp.multiply(1j * grids.z.d_wave_numbers[None, None, :], poisson_spectrum))
        )


def outer3(a, b, c):
    """
    Compute outer tensor product of vectors a, b, and c
    :param a: vector a_i
    :param b: vector b_j
    :param c: vector c_k
    :return: tensor a_i b_j c_k
    """
    return cp.tensordot(a, cp.tensordot(b, c, axes=0), axes=0)
