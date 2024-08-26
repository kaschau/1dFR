class BaseIntegrator:
    def __init__(self):
        pass


class rk1(BaseIntegrator):
    name = "rk1"

    def __init__(self):
        super().__init__()

        self.nbanks = 1

    def step(system, dt):
        system.RHS()

        system.u += dt*system.negdivconf