import numpy as np


class FastIsing2D:
    def __init__(
        self,
        L: int,
        J: float = 1.0,
        seed: int | None = None,
        init_state: str = "random",
    ):
        self.L = L
        self.J = J
        self.rng = np.random.default_rng(seed)

        if init_state == "random":
            self.spins = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
        elif init_state == "ordered":
            self.spins = np.ones((L, L), dtype=np.int8)
        else:
            raise ValueError("init_state must be 'random' or 'ordered'")

        x, y = np.indices((L, L))

        red = np.logical_and((x + y) % 2 == 0, np.logical_and(x < L - 1, y < L - 1))
        red[L - 1, L - 1] = True

        blue = np.logical_and((x + y) % 2 == 1, np.logical_and(x < L - 1, y < L - 1))

        green = np.logical_and((x + y) % 2 == 0, np.logical_or(x == L - 1, y == L - 1))
        green[L - 1, L - 1] = False

        yellow = np.logical_and((x + y) % 2 == 1, np.logical_or(x == L - 1, y == L - 1))

        self.masks = {
            "red": red,
            "blue": blue,
            "green": green,
            "yellow": yellow,
        }

    def _neighbor_sum(self) -> np.ndarray:
        s = self.spins
        return (
            np.roll(s, 1, axis=0)
            + np.roll(s, -1, axis=0)
            + np.roll(s, 1, axis=1)
            + np.roll(s, -1, axis=1)
        )

    def _heat_bath_update_mask(self, mask: np.ndarray, T: float) -> None:
        beta = 1.0 / T
        s = self.spins

        nn = self._neighbor_sum()
        local_field = self.J * nn[mask]

        # Probability that selected spins become +1
        p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))

        r = self.rng.random(np.count_nonzero(mask))
        s[mask] = np.where(r < p_up, 1, -1).astype(np.int8)

    def heat_bath_sweep(self, T: float) -> None:
        # Update one independent color set at a time
        self._heat_bath_update_mask(self.masks["red"], T)
        self._heat_bath_update_mask(self.masks["blue"], T)
        self._heat_bath_update_mask(self.masks["green"], T)
        self._heat_bath_update_mask(self.masks["yellow"], T)

    def magnetization(self) -> float:
        return float(np.sum(self.spins))

    def energy(self) -> float:
        s = self.spins
        E = -self.J * np.sum(s * (np.roll(s, -1, axis=0) + np.roll(s, -1, axis=1)))
        return float(E)