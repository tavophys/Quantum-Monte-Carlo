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
def wave_function(r1, r2, r12, alpha, a):
    if r12 <= a:
        return 0.0
    else:
        return np.exp(-alpha*(r1**2+r2**2))*(1-a/r12)

#Local energy  
def local_energy(r1, r2, r12, alpha, a): 
    if r12 > a:
        return 0.5*(r1**2+r2**2)+2*alpha*(3-alpha*(r1**2+r2**2))+2*alpha*a/(r12-a)
    else:
        return 0.0

# The Monte Carlo sampling with the Metropolis algo
def monte_carlo_sampling(alpha_values, a_values, number_mc_cycles=100000, step_size=1.0, return_error=True):
    # positions
    Energies = np.zeros((alpha_values.size,a_values.size))
    Variances = np.zeros((alpha_values.size,a_values.size))
    Position = np.zeros((number_particles,dimension), np.double)
    PositionTest = np.zeros((number_particles,dimension), np.double)
    ia=0

    for alpha in alpha_values:
        jb=0
        for a in a_values:
            energy, error = monte_carlo(alpha, a,
                               number_mc_cycles=number_mc_cycles,
                               step_size=step_size,
                               return_error=return_error)
            Energies[ia,jb] = energy
            Variances[ia,jb] = error
            jb+=1
        ia += 1
    return Energies, Variances

# def MC(alpha):
# def MC(alpha, a=0, NumberMCcycles=100000, StepSize=1.0, returnError=False):
def monte_carlo(alpha, a, number_mc_cycles=100000, step_size=1.0, return_error=False):
    # a=0
    # NumberMCcycles=100000
    # StepSize=1.0
    # returnError=False

    delta_E = 0.0
    energy = energy2 = 0.0

    # Initial state
    np.random.seed(0)
    wf = 0.0
    while wf == 0.0:
        position = step_size * (np.random.rand(number_particles, dimension) - 0.5)
        r1, r2 = np.linalg.norm(position, axis=1)
        r12 = np.linalg.norm(np.diff(position, axis=0), axis=1)
        wf = wave_function(r1, r2, r12, alpha, a)

    # Loop over Monte Carlo cycles
    mc_cycle_counter = 0
    while mc_cycle_counter < number_mc_cycles:
        np.random.seed(None)
        # Trial position
        position_next = position + step_size * (np.random.rand(number_particles, dimension) - 0.5)
        r1_next, r2_next = np.linalg.norm(position_next, axis=1)
        r12_next = np.linalg.norm(np.diff(position_next, axis=0), axis=1)
        wf_next = wave_function(r1_next, r2_next, r12_next, alpha, a)
        
        # Metropolis algorithm test to see whether we accept the move or not
        if min(np.random.rand(), 1.0) < wf_next**2 / wf**2:
            delta_E = local_energy(r1_next, r2_next, r12_next, alpha, a)
            energy += delta_E
            energy2 += delta_E ** 2

            position = np.copy(position_next)
            wf = np.copy(wf_next)

            mc_cycle_counter += 1

    # Calculation of the energy and the variance
    energy /= number_mc_cycles
    energy2 /= number_mc_cycles
    
    if return_error:
        variance = energy2 - energy**2
        error = np.sqrt(np.abs(variance/number_mc_cycles))
        return energy, error
    else:
        return energy
    

def f(x, offset=5):
    return -np.sinc(x - offset)


if __name__ == "__main__":
    ## input parameters
    input_path = Path(".", "vmc_results")
    number_particles = 2
    dimension = 3
    number_mc_cycles = 100000
    step_size = 1.0

    calc_minimized = True
    # calc_minimized = False
    alpha_min = 0.45
    alpha_max = 0.55
    with cProfile.Profile(subcalls=False) as prof:
        if calc_minimized:
            alpha_init = 0.5
            a_init = 0.0

            # result = basinhopping(monte_carlo, alpha_init,
            #                       stepsize=0.3, disp=True, niter_success=3, niter=10)

            # result = minimize(fun,
            result = minimize(monte_carlo,
                              alpha_init,
                              args=(a_init, number_mc_cycles, step_size),
                              method="L-BFGS-B",
                              bounds=Bounds(alpha_min, alpha_max),
                              options={"disp": True},
                              )

            print(f"{result}")
            """           
            results for a_init=0, Alpha_min=0.3, Alpha_max=0.6
            fun: 3.133083265596988
            hess_inv: <1x1 LbfgsInvHessProduct with dtype=float64>
            jac: array([381251.31410628])
            message: 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
            nfev: 32
            nit: 1
            njev: 16
            status: 0
            success: True
            x: array([0.36999921])
            """
        else:
            ## scanning parameters
            alpha_values = np.linspace(alpha_min, alpha_max, 10)
            a_values = np.array([0.0])
            # alpha_values = np.linspace(0.4, 0.6, 20)
            # a_values = np.linspace(0.0, 0.5, 5)
            
            #### INPUT END

            # create result dir, if non existent
            if not input_path.exists():
                input_path.mkdir(parents=True)

            # MC
            Energies, Variances = monte_carlo_sampling(alpha_values, a_values,
                                                       number_mc_cycles=number_mc_cycles,
                                                       step_size=step_size)
            
            # prepare results for plotting
            E_min_a_fixed_index = np.argmin(Energies, axis=0)
            Alpha_E_min_a_fixed = alpha_values[E_min_a_fixed_index]
            # energy along the line of best alphaValues (where E is smallest for a fixed a)
            E_min_a_fixed = np.array([Energies[E_min_index, i] for i, E_min_index in enumerate(E_min_a_fixed_index)])

            # save results in txt
            save_vals(input_path,
                      val_str_list=["AlphaValues", "aValues", "Energies", "Variances", "E_min_a_fixed_index", "Alpha_E_min_a_fixed", "E_min_a_fixed"],
                      val_list=[alpha_values, a_values, Energies, Variances, E_min_a_fixed_index, Alpha_E_min_a_fixed, E_min_a_fixed])

            # plot
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
            for i, ax in enumerate(axes):
                if i == 0:
                    # Prepare for plots
                    axes[i].set_xlabel("aValues")
                    axes[i].set_ylabel("Alpha_E_min_a_fixed")
                    axes[i].plot(a_values, Alpha_E_min_a_fixed)
                    axes[i].grid()
                elif i == 1:
                    axes[i].set_xlabel("aValues")
                    axes[i].set_ylabel("E_min_a")
                    axes[i].plot(a_values, E_min_a_fixed)
                    axes[i].grid()
                elif i == 2:
                    axes[i].set_title("E")
                    axes[i].set_xlabel("aValues")
                    axes[i].set_ylabel("AlphaValues")
                    X, Y = np.meshgrid(alpha_values, a_values, indexing="ij")
                    im = ax.pcolormesh(X, Y, Energies, shading="auto")
                    fig.colorbar(im, ax=axes[i])
                else:
                    print("Not implemented")
            fig.savefig(Path(input_path,"VMC_result.png"))
            
            """
            Results:

            AlphaValues: [0.3 0.4 0.5 0.6]
            aValues: [0.]
            Energies: [[3.40829594] [3.09173381] [3.        ] [3.04907309]]
            Variances: [[0.00292591] [0.00126307] [0.        ] [0.00099151]]
            E_min_a_fixed_index: [2]
            Alpha_E_min_a_fixed: [0.5]
            E_min_a_fixed: [3.]
            """

        ps = pstats.Stats(prof).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats("VMC.py")
            