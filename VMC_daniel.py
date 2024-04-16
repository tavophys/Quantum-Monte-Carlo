import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, Bounds
from pathlib import Path
from scipy.optimize import basinhopping



def save_vals(input_path, val_str_list, val_list):                                                                                                                                                               
    for (val_str, val) in zip(val_str_list, val_list):                                                                                                                                                           
        print(f"{val_str}: {val}")                                                                                                                                                                               
        with open(Path(input_path, val_str + ".txt"), "a") as f:                                                                                                                                                 
            f.write(f"{val}\n")


#Trial wave function
def wave_function(position, alpha, a):
    r1, r2 = np.linalg.norm(position, axis=1)
    r12 = np.linalg.norm(np.diff(position, axis=0), axis=1)

    if r12 <= a:
        return 0.0
    else:
        return np.exp(-alpha*(r1**2+r2**2))*(1-a/r12)

#Local energy  
def local_energy(position, alpha, a): 
    r1, r2 = np.linalg.norm(position, axis=1)
    r12 = np.linalg.norm(np.diff(position, axis=0), axis=1)

    if r12 > a:
        return 0.5*(r1**2+r2**2)+2*alpha*(3-alpha*(r1**2+r2**2))+2*alpha*a/(r12-a)
    else:
        return 0.0

def energy_jac(r1, r2, r12, alpha, a): 
    if r12 > a:
        return 2*(3-2 * alpha*(r1**2+r2**2))+ 2*a/(r12-a)
    else:
        return 0.0


# The Monte Carlo sampling with the Metropolis algo
def monte_carlo_sampling(alpha_values, a_values, number_mc_cycles=100000, mc_measure=100,
                         step_size=1.0, return_variance=True):
    # positions
    energies = np.zeros((alpha_values.size,a_values.size))
    variances_sqrt = np.zeros((alpha_values.size,a_values.size))
    Position = np.zeros((number_particles,dimension), np.double)
    PositionTest = np.zeros((number_particles,dimension), np.double)
    ia=0

    for alpha in alpha_values:
        jb=0
        print(f"alpha: {alpha}")
        for a in a_values:
            print(f"a: {a}")
            energy, variance_sqrt = monte_carlo(alpha, a,
                                                number_mc_cycles=number_mc_cycles,
                                                mc_measure=mc_measure,
                                                step_size=step_size,
                                                retun_variance=return_variance)
            energies[ia,jb] = energy
            variances_sqrt[ia,jb] = variance_sqrt
            jb+=1
        ia += 1
    return energies, variances_sqrt

def monte_carlo(alpha, a, number_mc_cycles=100000, mc_measure=100, step_size=1.0, retun_variance=False):
    delta_E = 0.0
    energy = energy2 = 0.0

    # Initial state
    # uses the same wf every monte_carlo
    np.random.seed(0)
    wf = 0.0
    while wf == 0.0:
        position = step_size * (np.random.rand(number_particles, dimension) - 0.5)
        wf = wave_function(position, alpha, a)

    # Loop over Monte Carlo cycles
    for i in range(number_mc_cycles):
        np.random.seed(None)
        # Trial position
        position_next = position + step_size * (np.random.rand(number_particles, dimension) - 0.5)
        wf_next = wave_function(position_next, alpha, a)
        
        # Metropolis algorithm test to see whether we accept the move or not
        if min(np.random.rand(), 1.0) < wf_next**2 / wf**2:
            position = np.copy(position_next)
            wf = np.copy(wf_next)

        if i % mc_measure == 0:
            delta_E = local_energy(position, alpha, a)
            energy += delta_E
            energy2 += delta_E ** 2

    number_of_measure = (number_mc_cycles / mc_measure)
    # Calculation of the energy and the variance
    energy /= number_of_measure
    energy2 /= number_of_measure
    
    
    if retun_variance:
        variance = energy2 - energy**2
        variance_sqrt = np.sqrt(np.abs(variance / number_of_measure))
        print(f"a, alpha, energy, variance_sqrt: {a}, {alpha}, {energy}, {variance_sqrt}")

        return energy, variance_sqrt
    else:
        print(f"a, alpha, energy: {a}, {alpha}, {energy}")

        return energy


if __name__ == "__main__":
    ## input parameters
    input_path = Path(".", "vmc_results")
    number_particles = 2
    dimension = 3
    number_mc_cycles = 100000
    # number_mc_cycles = 80000
    # number_mc_cycles = 800000
    # mc_measures = [1, 10, 20, 30, 40, 50]
    # mc_measures = [1, 2, 3, 4, 5, 7, 8]
    mc_measures = [10]
    step_size = 1.0

    # calc_minimized = True
    calc_minimized = False
    
    # use basinhopping or minimize
    basin = False

    # alpha_min = 0.40
    # alpha_max = 0.70
    # alpha_min = 0.7
    # alpha_max = 0.7
    alpha_min = 0.10
    alpha_max = 1.10
    # for grid
    alpha_num = 11

    with cProfile.Profile(subcalls=False) as prof:
        print(f"number_mc_cycles: {number_mc_cycles}")
        if calc_minimized:
            alpha_init = 0.7
            a_init = 0.0

            if basin:
                result = basinhopping(monte_carlo, alpha_init,
                                      minimizer_kwargs={"args": (a_init, number_mc_cycles, mc_measures[0], step_size)},
                                      stepsize=0.3,
                                      niter_success=3,
                                      niter=100,
                                      disp=True,
                                      )
            else:
                result = minimize(monte_carlo,
                                  alpha_init,
                                  args=(a_init, number_mc_cycles, mc_measures[0], step_size),
                                  method="L-BFGS-B",
                                  bounds=Bounds(alpha_min, alpha_max),
                                  tol=10**-3,
                                  options={"disp": True},
                                  )

            print(f"{result}")
        else:
            ## scanning parameters
            alpha_values = np.linspace(alpha_min, alpha_max, alpha_num)
            # alpha_values = np.array([0.55] * alpha_num)
            a_values = np.array([0.0])
            # a_values = np.linspace(0.0, 0.5, 5)
            
            #### INPUT END

            # create result dir, if non existent
            if not input_path.exists():
                input_path.mkdir(parents=True)

            for mc_measure in mc_measures:
                # MC
                energies, variances_sqrt = monte_carlo_sampling(alpha_values, a_values,
                                                                number_mc_cycles=number_mc_cycles,
                                                                mc_measure=mc_measure,
                                                                step_size=step_size)
                
            # prepare results for plotting
            E_min_a_fixed_index = np.argmin(energies, axis=0)
            Alpha_E_min_a_fixed = alpha_values[E_min_a_fixed_index]
            # energy along the line of best alphaValues (where E is smallest for a fixed a)
            E_min_a_fixed = np.array([energies[E_min_index, i] for i, E_min_index in enumerate(E_min_a_fixed_index)])

            # save results in txt
            save_vals(input_path,
                      val_str_list=["AlphaValues", "aValues", "Energies", "Variances", "E_min_a_fixed_index", "Alpha_E_min_a_fixed", "E_min_a_fixed"],
                      val_list=[alpha_values, a_values, energies, variances_sqrt, E_min_a_fixed_index, Alpha_E_min_a_fixed, E_min_a_fixed])

            # plot
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 10))
            for i, ax in enumerate(axes):
                if i == 0:
                    # Prepare for plots
                    axes[i].set_xlabel("a")
                    axes[i].set_ylabel("Alpha_E_min_a_fixed")
                    axes[i].plot(a_values, Alpha_E_min_a_fixed, "x-")
                    axes[i].grid()
                elif i == 1:
                    axes[i].set_xlabel("a")
                    axes[i].set_ylabel("E_min_a")
                    axes[i].plot(a_values, E_min_a_fixed, "x-")
                    axes[i].grid()
                elif i == 2:
                    axes[i].set_title("E")
                    axes[i].set_xlabel(r"$\alpha$")
                    axes[i].set_ylabel("a")
                    X, Y = np.meshgrid(alpha_values, a_values, indexing="ij")
                    im = ax.pcolormesh(X, Y, energies, shading="auto")
                    fig.colorbar(im, ax=axes[i])
                elif i == 3:
                    axes[i].set_xlabel(r"$\alpha$")
                    axes[i].set_ylabel("E")
                    for j, a in enumerate(a_values):
                        axes[i].errorbar(alpha_values, energies[:,j], variances_sqrt[:, j], fmt="o-",
                                         label=f"a={a}",
                                         )
                    axes[i].grid()
                    axes[i].legend()
                else:
                    print("Not implemented")
            fig.savefig(Path(input_path,"VMC_result.png"))

        ps = pstats.Stats(prof).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats("VMC.py")
            