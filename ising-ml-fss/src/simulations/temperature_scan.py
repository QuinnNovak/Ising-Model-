from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..ising.model import Ising2D


def run_single_scan(L, temps, n_eq, n_mc, seed):
    """
    Run one full temperature scan for one lattice size and one random seed.
    Returns arrays for m, e, chi, and C.
    """
    model = Ising2D(L=L, J=1.0, seed=seed)
    model.initialize_all_up()

    mags = []
    ens = [] 
    chis = []
    Cs = []

    for i, T in enumerate(temps): # loops over temps
        print(f"    T {i+1}/{len(temps)} : {T:.3f}")

        # Equilibrate
        for _ in range(n_eq):
            model.metropolis_sweep(T)

        M_sum = 0.0
        M2_sum = 0.0
        E_sum = 0.0
        E2_sum = 0.0

        # Measure
        for _ in range(n_mc):
            model.metropolis_sweep(T)

            M = model.magnetization()
            E = model.energy()

            M_sum += abs(M)
            M2_sum += M**2
            E_sum += E
            E2_sum += E**2

        N = L * L

        m_avg = M_sum / n_mc
        m2_avg = M2_sum / n_mc
        e_avg = E_sum / n_mc
        e2_avg = E2_sum / n_mc

        m_per_spin = m_avg / N
        e_per_spin = e_avg / N
        chi = (m2_avg - m_avg**2) / (N * T)
        C = (e2_avg - e_avg**2) / (N * T**2)

        mags.append(m_per_spin)
        ens.append(e_per_spin)
        chis.append(chi)
        Cs.append(C)

        print(
            f"       m = {m_per_spin:.6f}, "
            f"e = {e_per_spin:.6f}, "
            f"chi = {chi:.6f}, "
            f"C = {C:.6f}"
        )

    return {
        "mags": np.array(mags),
        "ens": np.array(ens),
        "chis": np.array(chis),
        "Cs": np.array(Cs),
    }


def run_temperature_scan():
    Path("figures").mkdir(exist_ok=True)

    # Lattice sizes and number of runs per lattice size
    Ls = [16,32]
    n_runs = 1

    # Temps
    temps = np.linspace(2.0, 2.6, 25)

    # Monte Carlo settings
    n_eq = 500 #1500
    n_mc = 1000 #2500

    Tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0)) # Critical point 

    # Final averaged results by lattice size
    results = {}

    for L in Ls: # loops over lattice sizes 
        print(f"\n{'='*50}")
        print(f"Running finite-size scan for L = {L}")
        print(f"{'='*50}")

        # These will store one full "curve" per run
        mags_runs = []
        ens_runs = []
        chis_runs = []
        Cs_runs = []

        for run_idx in range(n_runs): #loops over runs 
            seed = np.random.randint(0, 1_000_000)
            print(f"\n  Run {run_idx+1}/{n_runs} for L={L}, seed={seed}")

            run_data = run_single_scan(
                L=L,
                temps=temps,
                n_eq=n_eq,
                n_mc=n_mc,
                seed=seed,
            )

            mags_runs.append(run_data["mags"])
            ens_runs.append(run_data["ens"])
            chis_runs.append(run_data["chis"])
            Cs_runs.append(run_data["Cs"])

        # Convert to arrays of shape (n_runs, n_temps)
        mags_runs = np.array(mags_runs)
        ens_runs = np.array(ens_runs)
        chis_runs = np.array(chis_runs)
        Cs_runs = np.array(Cs_runs)

        # Average across runs at each temperature
        mags_mean = np.mean(mags_runs, axis=0)
        ens_mean = np.mean(ens_runs, axis=0)
        chis_mean = np.mean(chis_runs, axis=0)
        Cs_mean = np.mean(Cs_runs, axis=0)

        # standard deviation across runs
        mags_std = np.std(mags_runs, axis=0, ddof=1)
        ens_std = np.std(ens_runs, axis=0, ddof=1)
        chis_std = np.std(chis_runs, axis=0, ddof=1)
        Cs_std = np.std(Cs_runs, axis=0, ddof=1)

        results[L] = {
            "temps": temps.copy(),
            "mags_mean": mags_mean,
            "ens_mean": ens_mean,
            "chis_mean": chis_mean,
            "Cs_mean": Cs_mean,
            "mags_std": mags_std,
            "ens_std": ens_std,
            "chis_std": chis_std,
            "Cs_std": Cs_std,
            "raw_mags": mags_runs,
            "raw_ens": ens_runs,
            "raw_chis": chis_runs,
            "raw_Cs": Cs_runs,
        }

        print(f"\nFinished averaging {n_runs} runs for L={L}")
        print(f"Stored averaged data for {len(temps)} temperatures.")

        # Save individual averaged plots for this L
        plt.figure()
        plt.plot(temps, mags_mean, "o-")
        plt.axvline(Tc_exact, linestyle="--")
        plt.xlabel("Temperature T")
        plt.ylabel(r"$\langle |m| \rangle$ per spin")
        plt.title(f"Averaged Magnetization (L={L}, {n_runs} runs)")
        plt.tight_layout()
        plt.savefig(f"figures/magnetization_avg_L{L}.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(temps, ens_mean, "o-")
        plt.axvline(Tc_exact, linestyle="--")
        plt.xlabel("Temperature T")
        plt.ylabel(r"$\langle e \rangle$ per spin")
        plt.title(f"Averaged Energy (L={L}, {n_runs} runs)")
        plt.tight_layout()
        plt.savefig(f"figures/energy_avg_L{L}.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(temps, chis_mean, "o-")
        plt.axvline(Tc_exact, linestyle="--")
        plt.xlabel("Temperature T")
        plt.ylabel(r"$\chi$")
        plt.title(f"Averaged Susceptibility (L={L}, {n_runs} runs)")
        plt.tight_layout()
        plt.savefig(f"figures/susceptibility_avg_L{L}.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(temps, Cs_mean, "o-")
        plt.axvline(Tc_exact, linestyle="--")
        plt.xlabel("Temperature T")
        plt.ylabel("C")
        plt.title(f"Averaged Specific Heat (L={L}, {n_runs} runs)")
        plt.tight_layout()
        plt.savefig(f"figures/specific_heat_avg_L{L}.png", dpi=300)
        plt.close()
    
    
        # Average chi across lattice sizes L=16,32,64
    chi_all_L = np.array([results[L]["chis_mean"] for L in Ls])   # shape = (3, n_temps)
    chi_avg_all_L = np.mean(chi_all_L, axis=0)
    chi_std_all_L = np.std(chi_all_L, axis=0, ddof=1)

    # Save the averaged chi data
    results["chi_avg_all_L"] = {
        "temps": temps.copy(),
        "chi_mean": chi_avg_all_L,
        "chi_std": chi_std_all_L,
    }

    # Plot one combined averaged chi curve
    plt.figure()
    plt.plot(temps, chi_avg_all_L, "o-", label="Average of L=16,32,64")
    plt.axvline(Tc_exact, linestyle="--", label=r"$T_c$")
    plt.xlabel("Temperature T")
    plt.ylabel(r"$\chi$")
    plt.title("Average Susceptibility Across L = 16, 32, 64")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/chi_avg_all_lattice_sizes.png", dpi=300)
    plt.close()
    
    
    # Combined FSS plot: susceptibility
    plt.figure()
    for L in Ls:
        plt.plot(
            results[L]["temps"],
            results[L]["chis_mean"],
            "o-",
            label=f"L={L}"
        )
    plt.axvline(Tc_exact, linestyle="--")
    plt.xlabel("Temperature T")
    plt.ylabel(r"$\chi$")
    plt.title("Susceptibility vs Temperature (FSS, averaged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/fss_susceptibility_avg.png", dpi=300)
    plt.close()

    # Combined FSS plot: specific heat
    plt.figure()
    for L in Ls:
        plt.plot(
            results[L]["temps"],
            results[L]["Cs_mean"],
            "o-",
            label=f"L={L}"
        )
    plt.axvline(Tc_exact, linestyle="--")
    plt.xlabel("Temperature T")
    plt.ylabel("C")
    plt.title("Specific Heat vs Temperature (FSS, averaged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/fss_specific_heat_avg.png", dpi=300)
    plt.close()

    # Print pseudo-critical temperatures from averaged susceptibility peaks
    print(f"\n{'='*50}")
    print("Pseudo-critical temperatures from averaged susceptibility peaks")
    print(f"{'='*50}")
    for L in Ls:
        temps_L = results[L]["temps"]
        chis_L = results[L]["chis_mean"]
        Tc_L = temps_L[np.argmax(chis_L)]
        chi_max = np.max(chis_L)
        print(f"L = {L:2d} : Tc(L) ≈ {Tc_L:.3f}, chi_max = {chi_max:.6f}")