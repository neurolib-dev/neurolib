<!--include-in-documentation-->
## Compute optimal control signals within neurolib

This package enables to compute optimal control signals for single-node or whole-brain network models available in neurolib. Optimal control is the most efficient stimulus that can be applied to a system such that it, e.g., follows a particular trajectory. This framework enables to:
- define individual targets
- compute optimal control signals for any desired simulation setup (duration, network structure, dynamical model)

It is based on the neurolib library, a simulation and optimization framework for whole-brain modeling. More information can be found in the project overview.

### Installation

By cloning this repository, you will automatically install the latest version of neurolib. You can also clone this repository and install all dependencies with

```
git clone https://github.com/lenasal/neurolib.git
cd neurolib/
pip install -r requirements.txt
pip install .
```
It is recommended to clone or fork the entire repository since it will also include all examples and tests.

### Project layout

This project is an extenstion to the neurolib framework. The optimal control package is structured as follows:

```
neurolib/	 				# Main module
├── optimal_control/ 				# Optimal control package
	├── examples 				# Example Jupyter notebooks
	├── oc_fhn 				# Methods for optimal control of the fhn model
	├── tests 				# Automated tests
	├── cost_functions.py 			# Implementation of cost functions and gradients
```

### Examples

Example [IPython Notebooks](examples/) on how to use the library can be found in the `./examples/` directory.

### More information

#### Get in touch

Lena Salfenmoser (lena.salfenmoser@tu-berlin.de)
Department of Software Engineering and Theoretical Computer Science, Neural Information Processing Group, Technische Universität Berlin, Germany

Collaborators: Martin Krück (martink@bccn-berlin.de)
Bernstein Center for Computational Neuroscience Berlin, Germany

#### Acknowledgments
This work is supported by the DFG (German Research Foundation) via the CRC 910 (Project number
538 163436311).

<!--end-include-in-documentation-->
