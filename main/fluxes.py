import numpy as np
import cupy as cp


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr, axes=([axis, [1]])),
                        axes=permutation)


class DGFlux:
    """
    DGFlux object computes the DG projection RHS in dy/dt = F(y)
    """
    def __init__(self, resolutions, orders):
        self.resolutions = resolutions
        self.orders = orders
        self.permutations = [(0, 1, 6, 2, 3, 4, 5),
                             (0, 1, 2, 3, 6, 4, 5),
                             (0, 1, 2, 3, 4, 5, 6)]
        self.boundary_slices = [
            # x-directed face slices [(comps), (left), (right)]
            [(slice(3),
              slice(resolutions[0]), 0,
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), slice(orders[2])),
             (slice(3),
              slice(resolutions[0]), -1,
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), slice(orders[2]))],
            # y-directed face slices [(left), (right)]
            [(slice(3),
              slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), 0,
              slice(resolutions[2]), slice(orders[2])),
             (slice(3),
              slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), -1,
              slice(resolutions[2]), slice(orders[2]))],
            # z-directed face slices [(left), (right)]
            [(slice(3),
              slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), 0),
             (slice(3),
              slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), -1)]
        ]
        # flux slices [(comps), (left), (right)]
        self.flux_slices = [
            # x-directed face slices [(comps), (left), (right)]
            [(slice(resolutions[0]), 0,
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), slice(orders[2])),
             (slice(resolutions[0]), -1,
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), slice(orders[2]))],
            # y-directed face slices [(left), (right)]
            [(slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), 0,
              slice(resolutions[2]), slice(orders[2])),
             (slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), -1,
              slice(resolutions[2]), slice(orders[2]))],
            # z-directed face slices [(left), (right)]
            [(slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), 0),
             (slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), -1)]
        ]

        # grid and sub-element axes
        self.grid_axis = np.array([1, 3, 5])
        self.sub_element_axis = np.array([2, 4, 6])
        # numerical flux allocation size arrays
        self.num_flux_sizes = [(3, resolutions[0], 2, resolutions[1], orders[1], resolutions[2], orders[2]),
                               (3, resolutions[0], orders[0], resolutions[1], 2, resolutions[2], orders[2]),
                               (3, resolutions[0], orders[0], resolutions[1], orders[1], resolutions[2], 2)]

    def semi_discrete_rhs(self, vector, elliptic, basis, grids):
        """
        Calculate the semi-discrete equation's right-hand side given RK stage vector components
        """
        return ((self.flux(vector=vector, basis=basis.basis_x, dim=0) * grids.x.J) +
                (self.flux(vector=vector, basis=basis.basis_y, dim=1) * grids.y.J) +
                (self.flux(vector=vector, basis=basis.basis_z, dim=2) * grids.z.J) +
                (-1.0 * elliptic.pressure_gradient.arr))

    def flux(self, vector, basis, dim):
        """
        Compute x-directed flux of system
        """
        # Advection: flux is the tensor v_i * v_j
        flux = vector.arr[dim, :, :, :, :, :, :] * vector.arr[:, :, :, :, :, :]
        # compute internal (1st term) and numerical (2nd term) fluxes
        return (basis_product(flux=flux, basis_arr=basis.up,
                              axis=self.sub_element_axis[dim],
                              permutation=self.permutations[dim]) -
                self.numerical_flux(flux=flux, speed=vector.arr, basis=basis, dim=dim))

    def numerical_flux(self, flux, speed, basis, dim):
        # allocate the numerical flux
        num_flux = cp.zeros(self.num_flux_sizes[dim])
        # measure upwind directions
        speed_neg = cp.where(condition=speed[dim, :, :, :, :, :, :] < 0, x=1, y=0)
        speed_pos = cp.where(condition=speed[dim, :, :, :, :, :, :] >= 0, x=1, y=0)

        # upwind flux, first left then right faces
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.multiply(cp.roll(flux[self.boundary_slices[dim][1]],
                                                                             shift=1, axis=self.grid_axis[dim]),
                                                                     speed_pos[self.flux_slices[dim][0]]) +
                                                         cp.multiply(flux[self.boundary_slices[dim][0]],
                                                                     speed_neg[self.flux_slices[dim][0]]))
        num_flux[self.boundary_slices[dim][1]] = (cp.multiply(flux[self.boundary_slices[dim][1]],
                                                              speed_pos[self.flux_slices[dim][1]]) +
                                                  cp.multiply(cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
                                                                      axis=self.grid_axis[dim]),
                                                              speed_neg[self.flux_slices[dim][1]]))

        return basis_product(flux=num_flux, basis_arr=basis.xi,
                             axis=self.sub_element_axis[dim],
                             permutation=self.permutations[dim])
