class BaseIntegrator:
    def __init__(self):
        pass


class rk1(BaseIntegrator):
    name = "rk1"

    def __init__(self):
        super().__init__()

        self.nbanks = 2

    def step(self, system, dt):

        system.stage = 1
        system.RHS(0, 1)
        system.u0 += dt * system.u1

        system.t += dt
        system.niter += 1


class rk3(BaseIntegrator):
    name = "rk3"

    def __init__(self):
        super().__init__()

        self.nbanks = 3

    def step(self, system, dt):
        # stage 1
        system.stage = 1
        system.RHS(0, 2)
        system.u1 = system.u0 + dt * system.u2

        # stage 2
        system.stage = 2
        system.RHS(1, 2)
        system.u1 = 0.75 * system.u0 + 0.25 * system.u1 + 0.25 * dt * system.u2

        # stage 3
        system.stage = 3
        system.RHS(1, 2)
        system.u0 = (
            1.0 / 3.0 * system.u0 + 2.0 / 3.0 * system.u1 + 2.0 / 3.0 * dt * system.u2
        )

        # Post Process final stage solution
        system.upoly.compute_coeff(system.ua, system.u0, system.invuvdm)
        system.entropy_filter(0)
        system.t += dt
        system.niter += 1
