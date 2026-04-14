import numpy as np
print("Program started")
class Ising2D:
    def __init__(self, L: int, J: float = 1.0, seed: int | None = None):
        self.L = L
        self.J = J
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.choice([-1, 1], size=(L, L))

    def metropolis_sweep(self, T: float) -> None:
        L = self.L
        s = self.spins
        rng = self.rng
        J = self.J

        for _ in range(L * L):
            i = rng.integers(0, L)
            j = rng.integers(0, L)

            S = s[i, j]
            nn = (
                s[(i + 1) % L, j] + s[(i - 1) % L, j] +
                s[i, (j + 1) % L] + s[i, (j - 1) % L]
            )
            dE = 2.0 * J * S * nn

            if dE <= 0.0 or rng.random() < np.exp(-dE / T):
                s[i, j] = -S

    def magnetization(self) -> float:
        return float(np.sum(self.spins))

    def energy(self) -> float:
        s = self.spins
        J = self.J
        # count each bond once (right + down)
        E = -J * np.sum(s * (np.roll(s, -1, axis=0) + np.roll(s, -1, axis=1)))
        return float(E)
