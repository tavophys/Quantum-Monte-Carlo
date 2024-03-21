import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Trial wave function
def WaveFunction(r,alpha,a):
    r1,r2 = np.linalg.norm(r,axis=1)
    r12 = np.linalg.norm(np.diff(r,axis=0),axis=1)
    if r12<a:
        return 0.0
    else:
        return np.exp(-alpha*(r1**2+r2**2))*(1-a/r12)

#Local energy  
def LocalEnergy(r,alpha,a): 
    r1,r2 = np.linalg.norm(r,axis=1)
    r12 = np.linalg.norm(np.diff(r,axis=0),axis=1)
    if r12>a:
        return 0.5*(r1**2+r2**2)+2*alpha*(3-alpha*(r1**2+r2**2))+2*alpha*a/(r12-a)
    else:
        return 0.0

# The Monte Carlo sampling with the Metropolis algo
def MonteCarloSampling(AlphaValues,aValues):
    NumberMCcycles= 100000
    StepSize = 1.0
    # positions
    Energies = np.zeros((AlphaValues.size,aValues.size))
    Variances = np.zeros((AlphaValues.size,aValues.size))
    Position = np.zeros((NumberParticles,Dimension), np.double)
    PositionTest = np.zeros((NumberParticles,Dimension), np.double)
    ia=0
    for alpha in AlphaValues:
        jb=0
        for a in aValues:
            energy = energy2 = 0.0
            DeltaE = 0.0
            #Initial state
            wf = 0.0
            while wf==0.0:
                Position = StepSize * (np.random.rand(NumberParticles,Dimension) - 0.5)
                wf = WaveFunction(Position,alpha,a)
            #Loop over Monte Carlo cycles
            for MCcycle in range(NumberMCcycles):
                #Trial position
                PositionTest = Position + StepSize * (np.random.rand(NumberParticles,Dimension) - 0.5)
                wftest = WaveFunction(PositionTest,alpha,a)
                
                #Metropolis algorithm test to see whether we accept the move or not
                if min(np.random.rand(),1.0) < wftest**2 / wf**2:
                    Position = PositionTest+0.0
                    DeltaE = LocalEnergy(Position,alpha,a)
                    wf = wftest+0.0

                energy += DeltaE
                energy2 += DeltaE**2

            #Calculation of the energy and the variance
            energy /= NumberMCcycles
            energy2 /= NumberMCcycles
            variance = energy2 - energy**2
            error = np.sqrt(np.abs(variance/NumberMCcycles))
            Energies[ia,jb] = energy
            Variances[ia,jb] = error
            jb+=1
        ia += 1
    return Energies, Variances


#Main program
NumberParticles = 2
Dimension = 3

AlphaValues = np.linspace(0.4,0.6,20)
aValues = np.linspace(0.0,0.5,5)
Energies,Variances = MonteCarloSampling(AlphaValues,aValues)

# Prepare for plots
plt.plot(aValues,AlphaValues[np.argmin(Energies,axis=0)])
plt.show()
