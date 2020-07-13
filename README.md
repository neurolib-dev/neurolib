<p align="center">
  <a href="https://travis-ci.org/neurolib-dev/neurolib">
  	<img alt="Header image of neurolib - A Python simulation framework foreasy whole-brain neural mass modeling." src="https://github.com/neurolib-dev/neurolib/raw/master/resources/readme_header.png" ></a>
</p> 
<p align="center">
  <a href="https://travis-ci.org/neurolib-dev/neurolib">
  	<img alt="Build" src="https://travis-ci.org/neurolib-dev/neurolib.svg?branch=master"></a>
  
  <a href="https://zenodo.org/badge/latestdoi/236208651">
  	<img alt="DOI" src="https://zenodo.org/badge/236208651.svg"></a>
    
   
  <a href="https://github.com/neurolib-dev/neurolib/releases">
  	<img alt="Release" src="https://img.shields.io/github/v/release/neurolib-dev/neurolib"></a>
  
  <br>
  
  <a href="https://codecov.io/gh/neurolib-dev/neurolib">
  	<img alt="codecov" src="https://codecov.io/gh/neurolib-dev/neurolib/branch/master/graph/badge.svg"></a>
  
  <a href="https://pepy.tech/project/neurolib">
  	<img alt="Downloads" src="https://pepy.tech/badge/neurolib"></a>
  
  <a href="https://github.com/psf/black">
  	<img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

  <a href="https://mybinder.org/v2/gh/neurolib-dev/neurolib.git/master?filepath=examples">
  	<img alt="Launch in binder" src="https://mybinder.org/badge_logo.svg"></a>
  	 
</p>


# What is neurolib?

Please read the [gentle introduction](https://caglorithm.github.io/notebooks/neurolib-intro/) to `neurolib` for an overview of the basic functionality and some background information on the science behind whole-brain simulations.

`neurolib` allows you to build, simulate, and optimize your own state-of-the-art whole-brain models. To simulate the neural activity of each brain area, the main implementation provides an advanced neural mass mean-field model of spiking adaptive exponential integrate-and-fire neurons (AdEx) called `aln`. Each brain area is represented by two populations of excitatory and inhibitory neurons. An extensive analysis and validation of the `aln` model can be found in our [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007822) and its associated [github page](https://github.com/caglarcakan/stimulus_neural_populations).

`neurolib` provides a simulation and optimization framework which allows you to easily implement your own neural mass model, simulate fMRI BOLD activity, analyse the results and fit your model to empirical data.

Please reference the following paper if you use `neurolib` for your own research:

**Reference:** Cakan, C., Obermayer, K. (2020). Biophysically grounded mean-field models of neural populations under electrical stimulation. PLOS Computational Biology ([Link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007822)).

The figure below shows a schematic of how a brain network can be constructed:

<p align="center">
  <img src="https://github.com/neurolib-dev/neurolib/raw/master/resources/pipeline.png" width="700">
</p>

<p align="center">
Examples:
<a href="#single-node">Single node simulation</a> ·
<a href="#whole-brain-network">Whole-brain network</a> ·
<a href="#parameter-exploration">Parameter exploration</a> ·
<a href="#evolutionary-optimization">Evolutionary optimization</a>
<br><br>    
    
</p>

## Whole-brain modeling

Typically, in whole-brain modeling, diffusion tensor imaging (DTI) is used to infer the structural connectivity (the connection strength) between different brain areas. In a DTI scan, the direction of the diffusion of molecules is measured across the whole brain. Using [tractography](https://en.wikipedia.org/wiki/Tractography), this information can yield the distribution of axonal fibers in the brain that connect distant brain areas, called the connectome. Together with an atlas that divides the brain into distinct areas, a matrix can be computed that encodes how many fibers go from one area to another, the so-called structural connectivity (SC) matrix. This matrix defines the coupling strengths between brain areas and acts as an adjacency matrix of the brain network. The fiber length determines the signal transmission delay between all brain areas. Combining the structural data with a computational model of the neuronal activity of each brain area, we can create a dynamical model of the whole brain.

The resulting whole-brain model consists of interconnected brain areas, with each brain area having their internal neural dynamics. The neural activity can also be used to simulate hemodynamic [BOLD](https://en.wikipedia.org/wiki/Blood-oxygen-level-dependent_imaging) activity using the Balloon-Windkessel model, which can be compared to empirical fMRI data. Often, BOLD activity is used to compute correlations of activity between brain areas, the so called [resting state functional connectivity](https://en.wikipedia.org/wiki/Resting_state_fMRI#Functional), resulting in a matrix with correlations between each brain area. This matrix can then be fitted to empirical fMRI recordings of the resting-state activity of the brain.


Below is an animation of the neuronal activity of a whole-brain model plotted on a brain.

<p align="center">
  <img src="https://github.com/neurolib-dev/neurolib/raw/master/resources/brain_slow_waves_small.gif">
</p>

# Installation
The easiest way to get going is to install the pypi package using `pip`:

```
pip install neurolib
```
Alternatively, you can also clone this repository and install all dependencies with

```
git clone https://github.com/neurolib-dev/neurolib.git
cd neurolib/
pip install -r requirements.txt
pip install .
```

# Examples
Example [IPython Notebooks](examples/) on how to use the library can be found in the `./examples/` directory, don't forget to check them out! You can run the examples in your browser using Binder by clicking [here](https://mybinder.org/v2/gh/neurolib-dev/neurolib.git/master?filepath=examples) or one of the following links:

- [Example 0.0](https://mybinder.org/v2/gh/neurolib-dev/neurolib/master?filepath=examples%2Fexample-0-aln-minimal.ipynb) - Basic use of the `aln` model
- [Example 0.3](https://mybinder.org/v2/gh/neurolib-dev/neurolib/master?filepath=examples%2Fexample-0.3-fhn-minimal.ipynb) - Fitz-Hugh Nagumo model `fhn` on a brain network
- [Example 1.2](https://mybinder.org/v2/gh/neurolib-dev/neurolib/master?filepath=examples%2Fexample-1.2-brain-network-exploration.ipynb) - Parameter exploration of a brain network and fitting to BOLD data
- [Example 2.0](https://mybinder.org/v2/gh/neurolib-dev/neurolib/master?filepath=examples%2Fexample-2-evolutionary-optimization-minimal.ipynb) - A simple example of the evolutionary optimization framework 

A basic overview of the functionality of `neurolib` is also given in the following. 

## Single node

This example is available in detail as a [IPython Notebook](examples/example-0-aln-minimal.ipynb). 

To create a single `aln` model with the default parameters, simply run

```python
from neurolib.models.aln import ALNModel

aln = ALNModel()
aln.params['sigma_ou'] = 0.1 # add some noise

aln.run()
```

The results from this small simulation can be plotted easily:

```python
import matplotlib.pyplot as plt
plt.plot(aln.t, aln.rates_exc.T)

```
<p align="left">
  <img src="https://github.com/neurolib-dev/neurolib/raw/master/resources/single_timeseries.png">
</p>

## Whole-brain network

A detailed example is available as a [IPython Notebook](examples/example-0-aln-minimal.ipynb). 

To simulate a whole-brain network model, first we need to load a DTI and a resting-state fMRI dataset. `neurolib` already provides some example data for you:

```python
from neurolib.utils.loadData import Dataset

ds = Dataset("gw")
```
The dataset that we just loaded, looks like this:

<p align="center">
  <img src="https://github.com/neurolib-dev/neurolib/raw/master/resources/gw_data.png">
</p>

We initialize a model with the dataset and run it:

```python
aln = ALNModel(Cmat = ds.Cmat, Dmat = ds.Dmat)
aln.params['duration'] = 5*60*1000 # in ms, simulates for 5 minutes

aln.run(bold=True)
```
This can take several minutes to compute, since we are simulating 80 brain regions for 5 minutes realtime. Note that we specified `bold=True` which simulates the BOLD model in parallel to the neuronal model. The resulting firing rates and BOLD functional connectivity looks like this:
<p align="center">
  <img src="https://github.com/neurolib-dev/neurolib/raw/master/resources/gw_simulated.png">
</p>

The quality of the fit of this simulation can be computed by correlating the simulated functional connectivity matrix above to the empirical resting-state functional connectivity for each subject of the dataset. This gives us an estimate of how well the model reproduces inter-areal BOLD correlations. As a rule of thumb, a value above 0.5 is considered good. 

We can compute the quality of the fit of the simulated data using `func.fc()` which calculates a functional connectivity matrix of `N` (`N` = number of brain regions) time series. We use `func.matrix_correlation()` to compare this matrix to empirical data.

```python
scores = [func.matrix_correlation(func.fc(aln.BOLD.BOLD[:, 5:]), fcemp) for fcemp in ds.FCs]

print("Correlation per subject:", [f"{s:.2}" for s in scores])
print(f"Mean FC/FC correlation: {np.mean(scores):.2}")
```
```
Correlation per subject: ['0.34', '0.61', '0.54', '0.7', '0.54', '0.64', '0.69', '0.47', '0.59', '0.72', '0.58']
Mean FC/FC correlation: 0.58
```
## Parameter exploration
A detailed example of a single-node exploration is available as a [IPython Notebook](examples/example-1-aln-parameter-exploration.ipynb). For an example of a brain network exploration, see [this Notebook](examples/example-1.2-brain-network-exploration.ipynb).

Whenever you work with a model, it is of great importance to know what kind of dynamics it exhibits given a certain set of parameters. It is often useful to get an overview of the state space of a given model of interest. For example in the case of `aln`, the dynamics depends a lot on the mean inputs to the excitatory and the inhibitory population. `neurolib` makes it very easy to quickly explore parameter spaces of a given model:

```python
# create model
aln = ALNModel()
# define the parameter space to explore
parameters = ParameterSpace({"mue_ext_mean": np.linspace(0, 3, 21),  # input to E
		"mui_ext_mean": np.linspace(0, 3, 21)}) # input to I

# define exploration              
search = BoxSearch(aln, parameters)

search.run()                
```
That's it!. You can now use the builtin functions to load the simulation results from disk and perform your analysis:

```python
search.loadResults()

# calculate maximum firing rate for each parameter
for i in search.dfResults.index:
    search.dfResults.loc[i, 'max_r'] = np.max(search.results[i]['rates_exc'][:, -int(1000/aln.params['dt']):])
```
We can plot the results to get something close to a bifurcation diagram!

<p align="center">
  <img src="https://github.com/neurolib-dev/neurolib/raw/master/resources/exploration_aln.png">
</p>

## Evolutionary optimization

A detailed example is available as a [IPython Notebook](examples/example-2-evolutionary-optimization-minimal.ipynb). 

`neurolib` also implements evolutionary parameter optimization, which works particularly well with brain networks. In an evolutionary algorithm, each simulation is represented as an individual and the parameters of the simulation, for example coupling strengths or noise level values, are represented as the genes of each individual. An individual is a part of a population. In each generation, individuals are evaluated and ranked according to a fitness criterion. For whole-brain network simulations, this could be the fit of the simulated activity to empirical data. Then, individuals with a high fitness value are `selected` as parents and `mate` to create offspring. These offspring undergo random `mutations` of their genes. After all offspring are evaluated, the best individuals of the population are selected to transition into the next generation. This process goes on for a given amount generations until a stopping criterion is reached. This could be a predefined maximum number of generations or when a large enough population with high fitness values is found. 

An example genealogy tree is shown below. You can see the evolution starting at the top and individuals reproducing generation by generation. The color indicates the fitness.

<p align="center">
  <img src="https://github.com/neurolib-dev/neurolib/raw/master/resources/evolution_tree.png", width="600">
</p>

`neurolib` makes it very easy to set up your own evolutionary optimization and everything else is handled under the hood. You can chose between two implemented evolutionary algorithms: `adaptive` is a gaussian mutation and rank selection algorithm with adaptive step size that ensures convergence (a schematic is shown in the image below). `nsga2` is an implementation of the popular multi-objective optimization algorithm by Deb et al. 2002. 

<p align="center">
  <img src="https://github.com/neurolib-dev/neurolib/raw/master/resources/evolutionary-algorithm.png", width="600">
</p>

Of course, if you like, you can dig deeper, define your own selection, mutation and mating operators. In the following demonstration, we will simply evaluate the fitness of each individual as the distance to the unit circle. After a couple of generations of mating, mutating and selecting, only individuals who are close to the circle should survive:

```python
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.evolution import Evolution

def optimize_me(traj):
    ind = evolution.getIndividualFromTraj(traj)
    
    # let's make a circle
    fitness_result = abs((ind.x**2 + ind.y**2) - 1)
    
    # gather results
    fitness_tuple = (fitness_result ,)
    result_dict = {"result" : [fitness_result]}
    
    return fitness_tuple, result_dict
    
# we define a parameter space and its boundaries
pars = ParameterSpace(['x', 'y'], [[-5.0, 5.0], [-5.0, 5.0]])

# initialize the evolution and go
evolution = Evolution(optimize_me, pars, weightList = [-1.0], POP_INIT_SIZE= 100, POP_SIZE = 50, NGEN=10)
evolution.run()    
```

That's it! Now we can check the results:

```python
evolution.loadResults()
evolution.info(plot=True)
```

This will gives us a summary of the last generation and plots a distribution of the individuals (and their parameters). Below is an animation of 10 generations of the evolutionary process. Ass you can see, after a couple of generations, all remaining individuals lie very close to the unit circle.

<p align="center">
  <img src="https://github.com/neurolib-dev/neurolib/raw/master/resources/evolution_animated.gif">
</p>

## More information

### Built With

`neurolib` is built on other amazing open source projects:

* [pypet](https://github.com/SmokinCaterpillar/pypet) - Python parameter exploration toolbox
* [deap](https://github.com/DEAP/deap) - Distributed Evolutionary Algorithms in Python
* [numpy](https://github.com/numpy/numpy) - The fundamental package for scientific computing with Python
* [numba](https://github.com/numba/numba) - NumPy aware dynamic Python compiler using LLVM
* [Jupyter](https://github.com/jupyter/notebook) - Jupyter Interactive Notebook

### Get in touch

Caglar Cakan (cakan@ni.tu-berlin.de)  
Department of Software Engineering and Theoretical Computer Science, Technische Universität Berlin, Germany  
Bernstein Center for Computational Neuroscience Berlin, Germany  

### Acknowledgments
This work was supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) with the project number 327654276 (SFB 1315) and the Research Training Group GRK1589/2.