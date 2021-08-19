import numpy as np
import cupy as cp


# noinspection PyTypeChecker
class Grid1D:
    def __init__(self, low, high, res, basis, spectrum=False, fine=False, linspace=False):
        self.low = low
        self.high = high
        self.res = int(res)  # somehow gets non-int...
        self.res_ghosts = int(res + 2)  # resolution including ghosts
        self.order = basis.order

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.res

        # element Jacobian
        self.J = 2.0 / self.dx

        # The grid does not have a basis but does have quad weights
        self.quad_weights = cp.tensordot(cp.ones(self.res), cp.asarray(basis.weights), axes=0)
        # arrays
        self.arr = np.zeros((self.res_ghosts, self.order))
        self.create_grid(basis.nodes)
        self.arr_cp = cp.asarray(self.arr)
        self.midpoints = np.array([(self.arr[i, -1] + self.arr[i, 0]) / 2.0 for i in range(1, self.res_ghosts - 1)])
        self.arr_max = np.amax(abs(self.arr))

        # velocity axis gets a positive/negative indexing slice
        self.one_negatives = cp.where(condition=self.arr_cp < 0, x=1, y=0)
        self.one_positives = cp.where(condition=self.arr_cp >= 0, x=1, y=0)

        # fine array
        if fine:
            fine_num = 25  # 200 for 1D poisson study
            self.arr_fine = np.array([np.linspace(self.arr[i, 0], self.arr[i, -1], num=fine_num)
                                      for i in range(self.res_ghosts)])

        if linspace:
            lin_num = 150
            self.arr_lin = np.linspace(self.low, self.high, num=lin_num)

        # spectral coefficients
        if spectrum:
            self.nyquist_number = 2.0 * self.length // self.dx  # 2.5 *  # mode number of nyquist frequency
            # print(self.nyquist_number)
            self.k1 = 2.0 * np.pi / self.length  # fundamental mode
            self.wave_numbers = self.k1 * np.arange(1 - self.nyquist_number, self.nyquist_number)
            self.d_wave_numbers = cp.asarray(self.wave_numbers)
            self.grid_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr[1:-1, :], axes=0)))

            if linspace:
                self.lin_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr_lin, axes=0)))

            # Spectral matrices
            self.spectral_transform = basis.fourier_transform_array(self.midpoints, self.J, self.wave_numbers)
            self.inverse_transform = basis.inverse_transform_array(self.midpoints, self.J, self.wave_numbers)

    def create_grid(self, nodes):
        """
        Initialize array of global coordinates (including ghost elements).
        """
        # shift to include ghost cells
        min_gs = self.low - self.dx
        max_gs = self.high  # + self.dx
        # nodes (iso-parametric)
        nodes = (np.array(nodes) + 1) / 2

        # element left boundaries (including ghost elements)
        xl = np.linspace(min_gs, max_gs, num=self.res_ghosts)

        # construct coordinates
        for i in range(self.res_ghosts):
            self.arr[i, :] = xl[i] + self.dx * nodes

    def grid2cp(self):
        self.arr = cp.asarray(self.arr)

    def grid2np(self):
        self.arr = self.arr.get()

    def fourier_basis(self, function, idx):
        """
        On GPU, compute Fourier coefficients on the LGL grid of the given grid function
        """
        # print(function.shape)
        # print(self.spectral_transform.shape)
        # quit()
        return cp.tensordot(function, self.spectral_transform, axes=(idx, [0, 1])) * self.dx / self.length

    def sum_fourier(self, coefficients, idx):
        """
        On GPU, re-sum Fourier coefficients up to pre-set cutoff
        """
        return cp.tensordot(coefficients, self.grid_phases, axes=(idx, [0]))

    def sum_fourier_to_linspace(self, coefficients, idx):
        return cp.tensordot(coefficients, self.lin_phases, axes=(idx, [0]))


class Grid3D:
    def __init__(self, basis, lows, highs, resolutions, fine_all=False, linspace=False):
        # Grids
        self.x = Grid1D(low=lows[0], high=highs[0], res=resolutions[0],
                        basis=basis.basis_x, spectrum=True, fine=fine_all, linspace=linspace)
        self.y = Grid1D(low=lows[1], high=highs[0], res=resolutions[0],
                        basis=basis.basis_x, spectrum=True, fine=fine_all, linspace=linspace)
        self.z = Grid1D(low=lows[2], high=highs[0], res=resolutions[0],
                        basis=basis.basis_x, spectrum=True, fine=fine_all, linspace=linspace)

        # list of grids (all 1D, not too big)
        # self.grid = [self.x, self.y, self.z]

        # resolutions
        self.res_ghosts = [self.x.res_ghosts, self.y.res_ghosts, self.z.res_ghosts]
        self.orders = [self.x.order, self.y.order, self.z.order]

    def fourier_transform(self, function):
        # Transform function on a 3D grid
        x_transform = cp.transpose(self.x.fourier_basis(function=function, idx=[0, 1]),
                                   axes=(4, 0, 1, 2, 3))
        xy_transform = cp.transpose(self.y.fourier_basis(function=x_transform, idx=[1, 2]),
                                    axes=(0, 3, 1, 2))
        xyz_transform = self.z.fourier_basis(function=xy_transform, idx=[2, 3])
        return xyz_transform

    def inverse_transform(self, spectrum):
        # Transform back to piecewise grid from spectrum
        z_transform = self.z.sum_fourier(coefficients=spectrum, idx=[2])
        yz_transform = self.y.sum_fourier(coefficients=z_transform, idx=[1])
        xyz_transform = self.x.sum_fourier(coefficients=yz_transform, idx=[0])
        return cp.real(xyz_transform)


class Scalar:
    def __init__(self, resolutions, orders, perturbation=True):
        # if perturbation
        self.perturbation = perturbation

        # resolutions (no ghosts)
        self.x_res, self.y_res, self.z_res = resolutions[0], resolutions[1], resolutions[2]

        # orders
        self.x_ord, self.y_ord, self.z_ord = int(orders[0]), int(orders[1]), int(orders[2])

        # array
        self.arr = None

        # sizes (slices including ghost (g) cells)
        slice0 = slice(resolutions[0] + 2)
        slice1 = slice(resolutions[1] + 2)
        slice2 = slice(resolutions[2] + 2)

        self.boundary_slices = [
            # x-directed face slices [(left), (right)]
            [(slice0, 0, slice1, slice(self.y_ord), slice2, slice(self.z_ord)),
             (slice0, -1, slice1, slice(self.y_ord), slice2, slice(self.z_ord))],
            [(slice0, slice(self.x_ord), slice1, 0, slice2, slice(self.z_ord)),
             (slice0, slice(self.x_ord), slice1, -1, slice2, slice(self.z_ord))],
            [(slice0, slice(self.x_ord), slice1, slice(self.y_ord), slice2, 0),
             (slice0, slice(self.x_ord), slice1, slice(self.y_ord), slice2, -1)]
        ]

        # Grid and sub-element axes
        self.grid_axis = np.array([0, 2, 4])
        self.sub_element_axis = np.array([1, 3, 5])

    def initialize(self, grids):
        # Just sine product...
        (ix, iy, iz) = (cp.ones((grids.x.res + 2, grids.x.order)),
                        cp.ones((grids.y.res + 2, grids.y.order)),
                        cp.ones((grids.z.res + 2, grids.z.order)))
        (x3, y3, z3) = (outer3(a=grids.x.arr_cp, b=iy, c=iz),
                        outer3(a=ix, b=grids.y.arr_cp, c=iz),
                        outer3(a=ix, b=iy, c=grids.z.arr_cp))
        # random function
        self.arr = cp.sin(x3) * cp.sin(y3) * cp.sin(z3)

    def grid_flatten_gpu(self):
        return self.arr.reshape((self.x_res * self.x_ord, self.y_res * self.y_ord, self.z_res * self.z_ord))

    def grid_flatten_gpu_no_ghosts(self):
        return self.arr[1:-1, :, 1:-1, :, 1:-1, :].reshape(((self.x_res - 2) * self.x_ord,
                                                            (self.y_res - 2) * self.y_ord,
                                                            (self.z_res - 2) * self.z_ord))


class Vector:
    def __init__(self, resolutions, orders, perturbation=True):
        self.perturbation = perturbation
        # resolution and orders
        self.res = resolutions
        self.ord = orders

        # arrays
        self.arr, self.arr_stages, self.grad, self.pressure_source = None, None, None, None

        # no ghost slices of vector on grid
        self.no_ghost_slice = (slice(2),
                               slice(1, self.res[0] - 1), slice(self.ord[0]),
                               slice(1, self.res[1] - 1), slice(self.ord[1]),
                               slice(1, self.res[2] - 1), slice(self.ord[2]))

    def initialize(self, grids):
        # Just sine product...
        (ix, iy, iz) = (cp.ones((grids.x.res + 2, grids.x.order)),
                        cp.ones((grids.y.res + 2, grids.y.order)),
                        cp.ones((grids.z.res + 2, grids.z.order)))
        (x3, y3, z3) = (outer3(a=grids.x.arr_cp, b=iy, c=iz),
                        outer3(a=ix, b=grids.y.arr_cp, c=iz),
                        outer3(a=ix, b=iy, c=grids.z.arr_cp))

        arr_x, arr_y, arr_z = abc(x3=x3, y3=y3, z3=z3, amps=(1, 1, 1), mode=1, phase=0)

        self.arr = cp.array([arr_x, arr_y, arr_z])

    def gradient_tensor(self, grids):
        """
        Compute gradient tensor of vector quantity using Fourier spectrum
        :param grids: 3D grids object
        sets self.grad as gradient tensor
        """
        # Spectrum
        spectra = [grids.fourier_transform(function=self.arr[idx, 1:-1, :, 1:-1, :, 1:-1, :]) for idx in range(3)]

        # Gradient in spectral space
        spectral_derivatives = [[cp.multiply(1j * grids.x.d_wave_numbers[:, None, None], spectra[dim]),
                                 cp.multiply(1j * grids.y.d_wave_numbers[None, :, None], spectra[dim]),
                                 cp.multiply(1j * grids.z.d_wave_numbers[None, None, :], spectra[dim])]
                                for dim in range(3)]

        # Inverse transform
        self.grad = cp.array([[grids.inverse_transform(spectral_derivatives[i][j])
                               for j in range(3)]
                              for i in range(3)])

    def poisson_source(self):
        """
        Compute double contraction of velocity-gradient tensor
        """
        self.pressure_source = -1.0 * cp.einsum('ijklmnop,jiklmnop->klmnop', self.grad, self.grad)

    def grid_flatten_arr(self):
        return self.arr.reshape((3, self.res[0] * self.ord[0], self.res[1] * self.ord[1], self.res[2] * self.ord[2]))

    def grid_flatten_arr_no_ghost(self):
        return self.arr[:, 1:-1, :, 1:-1, :, 1:-1, :].reshape((3, (self.res[0] - 2) * self.ord[0],
                                                                  (self.res[1] - 2) * self.ord[1],
                                                                  (self.res[2] - 2) * self.ord[2]))

    def grid_flatten_grad(self):
        return self.grad.reshape((3, 3, (self.res[0] - 2) * self.ord[0],
                                        (self.res[1] - 2) * self.ord[1],
                                        (self.res[2] - 2) * self.ord[2]))

    def grid_flatten_source(self):
        return self.pressure_source.reshape((self.res[0] - 2) * self.ord[0],
                                            (self.res[1] - 2) * self.ord[1],
                                            (self.res[2] - 2) * self.ord[2])

    def ghost_sync(self):
        self.arr[:, 0, :, :, :, :, :] = self.arr[:, -2, :, :, :, :, :]
        self.arr[:, -1, :, :, :, :, :] = self.arr[:, 1, :, :, :, :, :]
        self.arr[:, :, :, 0, :, :, :] = self.arr[:, :, :, -2, :, :, :]
        self.arr[:, :, :, -1, :, :, :] = self.arr[:, :, :, 1, :, :, :]
        self.arr[:, :, :, :, :, 0, :] = self.arr[:, :, :, :, :, -2, :]
        self.arr[:, :, :, :, :, -1, :] = self.arr[:, :, :, :, :, 1, :]

    def filter(self, grids):
        # Compute spectrum
        spectrum_x = grids.fourier_transform(function=self.arr[0, 1:-1, :, 1:-1, :, 1:-1, :])
        spectrum_y = grids.fourier_transform(function=self.arr[1, 1:-1, :, 1:-1, :, 1:-1, :])
        spectrum_z = grids.fourier_transform(function=self.arr[2, 1:-1, :, 1:-1, :, 1:-1, :])
        # Inverse transform
        self.arr[0, 1:-1, :, 1:-1, :, 1:-1, :] = grids.inverse_transform(spectrum=spectrum_x)
        self.arr[1, 1:-1, :, 1:-1, :, 1:-1, :] = grids.inverse_transform(spectrum=spectrum_y)
        self.arr[2, 1:-1, :, 1:-1, :, 1:-1, :] = grids.inverse_transform(spectrum=spectrum_z)


def outer3(a, b, c):
    """
    Compute outer tensor product of vectors a, b, and c
    :param a: vector a_i
    :param b: vector b_j
    :param c: vector c_k
    :return: tensor a_i b_j c_k as numpy array
    """
    return cp.tensordot(a, cp.tensordot(b, c, axes=0), axes=0)


def abc(x3, y3, z3, amps, mode, phase):
    """
    Returns ABC flow vector field with amplitudes amp=(a,b,c), mode number m = mode, and angle phi=phase
    """
    a, b, c = amps
    return (a * cp.sin(mode * z3 + phase) + c * cp.cos(mode * y3 + phase),
            b * cp.sin(mode * x3 + phase) + a * cp.cos(mode * z3 + phase),
            c * cp.sin(mode * y3 + phase) + b * cp.cos(mode * x3 + phase))

# Bin
# dfx_x_k = cp.multiply(1j * grids.x.d_wave_numbers[:, None, None], spectrum_x)
# dfx_y_k = cp.multiply(1j * grids.y.d_wave_numbers[None, :, None], spectrum_x)
# dfx_z_k = cp.multiply(1j * grids.z.d_wave_numbers[None, None, :], spectrum_x)
#
# dfy_x_k = cp.multiply(1j * grids.x.d_wave_numbers[:, None, None], spectrum_y)
# dfy_y_k = cp.multiply(1j * grids.y.d_wave_numbers[None, :, None], spectrum_y)
# dfy_z_k = cp.multiply(1j * grids.z.d_wave_numbers[None, None, :], spectrum_y)
#
# dfy_x_k = cp.multiply(1j * grids.x.d_wave_numbers[:, None, None], spectrum_y)
# dfy_y_k = cp.multiply(1j * grids.y.d_wave_numbers[None, :, None], spectrum_y)
# dfy_z_k = cp.multiply(1j * grids.z.d_wave_numbers[None, None, :], spectrum_y)
# self.grad = cp.array([[grids.inverse_transform(spectrum=spectral_derivative(wave_numbers=))]])


# Static functions
# def spectral_derivatives(grids, spectra):
#     return
