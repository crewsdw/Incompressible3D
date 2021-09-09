import numpy as np
import cupy as cp
import grid as g
import time as timer

# Courant numbers for RK-DG stability from Cockburn and Shu 2001, [time_order][space_order-1]
courant_numbers = {
    1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    2: [1.0, 0.333],
    3: [1.256, 0.409, 0.209, 0.130, 0.089, 0.066, 0.051, 0.040, 0.033],
    4: [1.392, 0.464, 0.235, 0.145, 0.100, 0.073, 0.056, 0.045, 0.037],
    5: [1.608, 0.534, 0.271, 0.167, 0.115, 0.085, 0.065, 0.052, 0.042],
    6: [1.776, 0.592, 0.300, 0.185, 0.127, 0.093, 0.072, 0.057, 0.047],
    7: [1.977, 0.659, 0.333, 0.206, 0.142, 0.104, 0.080, 0.064, 0.052],
    8: [2.156, 0.718, 0.364, 0.225, 0.154, 0.114, 0.087, 0.070, 0.057]
}

# Shu-Osher SSPRK stage coefficients
nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    """
    Class to time-step the discretized PDE, or the "main loop"
    """

    def __init__(self, time_order, space_order, write_time, final_time):
        # CLass info
        self.time_order, self.space_order = time_order, space_order
        self.coefficients = self.get_nonlinear_coefficients()
        self.courant = self.get_courant_number()

        # Stepping parameters and init
        self.write_time, self.final_time = write_time, final_time
        self.time, self.dt = 0, None
        self.steps_counter, self.write_counter = 0, 1

        # Save initializations
        self.time_array, self.field_energy = np.array([0]), np.array([])
        self.save_array = []

    def get_nonlinear_coefficients(self):
        return np.array(nonlinear_ssp_rk_switch.get(self.time_order, "nothing"))

    def get_courant_number(self):
        return courant_numbers.get(self.time_order)[self.space_order - 1]

    def main_loop(self, vector, basis, elliptic, grids, dg_flux):
        """
        Loop while time is less than final time
        """
        print('\nInitializing main loop and time-step...')
        t0 = timer.time()
        self.adapt_time_step(vector=vector, dx=grids.dx)
        # self.save_array += [vector.arr.get()]
        print('Initial dt is {:0.3e}'.format(self.dt.get()))

        while self.time < self.final_time:
            print('Step taken and dt is {:0.3e}'.format(self.dt.get()))
            self.nonlinear_ssp_rk(vector=vector, basis=basis, elliptic=elliptic,
                                  grids=grids, dg_flux=dg_flux)
            self.time += self.dt.get()
            self.steps_counter += 1
            # Get field energy and time
            self.time_array = np.append(self.time_array, self.time)

            # Do periodic write-out
            if self.time > self.write_counter * self.write_time:
                print('\nI made it through step ' + str(self.steps_counter))
                print('The simulation time is {:0.3e}'.format(self.time))
                print('The time-step is {:0.3e}'.format(self.dt.get()))
                print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')
                self.write_counter += 1
            if cp.isnan(vector.arr).any():
                print('\nThere is nan... quitting now.')
                print(self.steps_counter)
                quit()

    def nonlinear_ssp_rk(self, vector, basis, elliptic, grids, dg_flux):
        # Sync ghost cells
        vector.ghost_sync()
        # Set and init up stage vectors
        stages = 3 * [g.Vector(resolutions=grids.res_ghosts, orders=grids.orders)]
        for i in range(3):
            stages[i].arr = cp.zeros_like(vector.arr)

        # Zeroth stage: solve pressure, adapt time-step, accumulate
        t_0 = timer.time()
        elliptic.pressure_solve(velocity=vector, grids=grids)
        t_1 = timer.time()
        self.adapt_time_step(vector=vector, dx=grids.dx)
        t_2 = timer.time()
        stages[0].arr[vector.no_ghost_slice] = (vector.arr[vector.no_ghost_slice] +
                                                self.dt * dg_flux.semi_discrete_rhs(vector=vector,
                                                                                    elliptic=elliptic,
                                                                                    basis=basis,
                                                                                    grids=grids)[vector.no_ghost_slice]
                                                )
        t_3 = timer.time()
        stages[0].ghost_sync()

        # Additional stages: solve pressure, accumulate
        for i in range(2):
            elliptic.pressure_solve(velocity=stages[i], grids=grids)
            stages[i + 1].arr[
                vector.no_ghost_slice
            ] = (self.coefficients[i, 0] * vector.arr[vector.no_ghost_slice] +
                 self.coefficients[i, 1] * stages[i].arr[vector.no_ghost_slice] +
                 self.coefficients[i, 2] * self.dt * (
                     dg_flux.semi_discrete_rhs(vector=stages[i],
                                               elliptic=elliptic,
                                               basis=basis,
                                               grids=grids)[vector.no_ghost_slice]
                 ))
            stages[i + 1].ghost_sync()

        # Update distribution
        vector.arr[vector.no_ghost_slice] = stages[2].arr[vector.no_ghost_slice]

    def adapt_time_step(self, vector, dx):
        """
        Adapt time-step according to hyperbolic stability limit
        """
        max_speeds = cp.array([cp.amax(cp.absolute(
            vector.arr[i, :, :, :, :, :, :]))
            for i in range(3)])
        om = sum([max_speeds[i] / dx[i] for i in range(3)])
        self.dt = self.courant / om
