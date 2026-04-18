import numpy as np
print("Program started")
class Ising2D:
    def __init__(self, L: int, J: float = 1.0, seed: int | None = None):
        self.L = L
        self.J = J
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.choice([-1, 1], size=(L, L))
        
        self.M = float(np.sum(self.spins))
        self.E = self.total_energy()
        
    def total_energy(self) -> float:
        s = self.spins
        J = self.J
        # count each bond once (right + down)
        E = -J * np.sum(s * (np.roll(s, -1, axis=0) + np.roll(s, -1, axis=1)))
        return float(E)
    
    def initialize_all_up(self) -> None:
        self.spins[:, :] = 1
        self.M = float(self.L * self.L)
        self.E = self.total_energy()
    
    def metropolis_sweep(self, T: float) -> None:
        L = self.L
        s = self.spins
        rng = self.rng
        J = self.J
        
        # compute only the positive Boltzmann factors used to accept the spin flip or not 
        w4 = np.exp(-4.0 * J / T)
        w8 = np.exp(-8.0 * J / T)
        
        for _ in range(L * L):
            i = rng.integers(0, L)
            j = rng.integers(0, L)

            S = s[i, j]
            nn = (
                s[(i + 1) % L, j] + s[(i - 1) % L, j] +
                s[i, (j + 1) % L] + s[i, (j - 1) % L]
            )
            dE = 2.0 * J * S * nn

            accept = False
            if dE <= 0.0:
                accept = True
            elif dE == 4.0 * J:
                accept = (rng.random() < w4)
            elif dE == 8.0 * J:
                accept = (rng.random() < w8)

            if accept:
                s[i, j] = -S

                # Update stored total magnetization and total energy
                self.M += -2.0 * S
                self.E += dE

    def magnetization(self) -> float:
        return self.M

    def energy(self) -> float:
        return self.E
