<p align="center">
  	<img alt="Header image of neurolib - A Python simulation framework foreasy whole-brain neural mass modeling." src="https://github.com/neurolib-dev/neurolib/raw/master/resources/readme_header.png" >
</p> 

## Getting started
* To browse the source code of `neurolib` visit out [GitHub repository](https://github.com/neurolib-dev/neurolib).
* Read the [gentle introduction](https://caglorithm.github.io/notebooks/neurolib-intro/) to `neurolib` for an overview of the basic functionality and some background information on the science behind whole-brain simulations.

## Installation
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

## Project layout


    neurolib/					# Main module
    	models/					# Neural mass models
    		model.py			# Base model class
    		/.../				# Implemented neural models
    	optimize/				# Optimization submodule
    		evolution/			# Evolutionary optimization
    			evolution.py
    			...
    		exploration/		# Parameter exploration
    			exploration.py
    			...
    	data/					# Empirical datasets (structural, functional)
    		...
    	utils/					# Utility belt
    		atlases.py			# Atlases (Region names, coordinates)
    		collections.py		# Custom data types
    		functions.py		# Useful functions
    		loadData.py			# Dataset loader
    		parameterSpace.py	# Parameter space
			saver.py			# Save simulation outputs
			signal.py			# Signal processing functions
			stimulus.py			# Stimulus construction
    examples/					# Example Jupyter notebooks
    docs/						# Documentation 			
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
