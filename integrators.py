class BaseIntegrator:
    def __init__(self):
        pass


class rk1(BaseIntegrator):
    name = "rk1"

    def __init__(self):
        super().__init__()

        self.nbanks = 1

    def step(self, system, dt):

        system.RHS(0)
        system.u0 += dt * system.negdivconf

        system.t += dt
        system.niter += 1


class rk3(BaseIntegrator):
    name = "rk3"

    def __init__(self):
        super().__init__()

        self.nbanks = 2

    def step(self, system, dt):
        # store first stage (u0 already processed)
        system.u1[:] = system.u0[:]
        # stage 1
        system.RHS(0)
        system.u0 += dt * system.negdivconf

        # stage 2
        system.RHS(0)
        system.u0 = 0.75 * system.u1 + 0.25 * system.u0 + 0.25 * system.negdivconf * dt

        # stage 3
        system.RHS(0)
        system.u0 = (system.u1 + 2.0 * system.u0 + 2.0 * system.negdivconf * dt) / 3.0

        system.t += dt
        system.niter += 1
