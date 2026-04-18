from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..ising.fast_model import FastIsing2D


def run_fast_temperature_scan():
    Path("figures").mkdir(exist_ok=True)

    L = [16, 32,64]
    n_eq = 1500
    n_mc = 2500
    temps = np.linspace(1.5, 3.5, 20)

    seed = np.random.randint(0, 1_000_000)
    print(f"Initial seed = {seed}")

    model = FastIsing2D(L=L, J=1.0, seed=seed, init_state="ordered")

    mags = []
    ens = []
    chis = []

    for i, T in enumerate(temps):
        print(f"T {i+1}/{len(temps)} : {T:.3f}")

        # Equilibrate
        for _ in range(n_eq):
            model.heat_bath_sweep(T)

        M_sum = 0.0
        M2_sum = 0.0
        E_sum = 0.0

        # Measure
        for _ in range(n_mc):
            model.heat_bath_sweep(T)

            M = model.magnetization()
            E = model.energy()

            M_sum += abs(M)
            M2_sum += M**2
            E_sum += E

        N = L * L

        m_avg = M_sum / n_mc
        m2_avg = M2_sum / n_mc
        e_avg = E_sum / n_mc

        m_per_spin = m_avg / N
        e_per_spin = e_avg / N
        chi = (m2_avg - m_avg**2) / (N * T)

        mags.append(m_per_spin)
        ens.append(e_per_spin)
        chis.append(chi)

        print(
            f"   m = {m_per_spin:.6f}, "
            f"e = {e_per_spin:.6f}, "
            f"chi = {chi:.6f}"
        )

    Tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))

    plt.figure()
    plt.plot(temps, mags, "o-")
    plt.axvline(Tc_exact, linestyle="--")
    plt.xlabel("Temperature T")
    plt.ylabel(r"$\langle |m| \rangle$ per spin")
    plt.title(f"Fast 2D Ising Magnetization (L={L})")
    plt.tight_layout()
    plt.savefig("figures/fast_magnetization.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(temps, ens, "o-")
    plt.axvline(Tc_exact, linestyle="--")
    plt.xlabel("Temperature T")
    plt.ylabel(r"$\langle e \rangle$ per spin")
    plt.title(f"Fast 2D Ising Energy (L={L})")
    plt.tight_layout()
    plt.savefig("figures/fast_energy.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(temps, chis, "o-")
    plt.axvline(Tc_exact, linestyle="--")
    plt.xlabel("Temperature T")
    plt.ylabel(r"$\chi$")
    plt.title(f"Fast 2D Ising Susceptibility (L={L})")
    plt.tight_layout()
    plt.savefig("figures/fast_susceptibility.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_fast_temperature_scan()
