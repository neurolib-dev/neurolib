![](https://travis-ci.org/neurolib-dev/neurolib.svg?branch=master) ![](https://img.shields.io/github/v/release/neurolib-dev/neurolib) ![PyPI](https://img.shields.io/pypi/v/neurolib) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Neurolib
*Easy whole-brain neural mass modeling* 👩‍🔬💻🧠

Neurolib allows you to easily create your own state-of-the-art whole-brain models. The main implementation is a neural mass firing rate model of spiking adaptive exponential integrate-and-fire neurons (AdEx) called `aln` which consists of two populations of excitatory and a inhibitory neurons. An extensive analysis of the model can be found in the paper *Biophysically grounded mean-field models of neural populations under electrical stimulation*, Cakan et al. 2020 ([ArXiv](https://arxiv.org/abs/1906.00676)), and its associated [github page](https://github.com/caglarcakan/stimulus_neural_populations).

### Whole-brain modeling

In combination with structural brain data, for example from diffusion tensor imaging (DTI) [tractography](https://en.wikipedia.org/wiki/Tractography), and resting state [BOLD](https://en.wikipedia.org/wiki/Blood-oxygen-level-dependent_imaging) data from magnetic resonance imaging (rs-fMRI), a network model of a whole brain can be created. Structural connectivity matrices from DTI tractography define 1) the connection strengths between areas, represented for example by the number of axonal fibers between each two brain areas and 2) the signal transmission delays measured from the length of the axonal fibers. 

The resulting whole-brain model consists of interconnected brain areas, with each brain area having their internal neural dynamics. The neural activity is used to simulate BOLD activity using the Balloon-Windkessel model. The resulting simulated [resting state functional connectivity](https://en.wikipedia.org/wiki/Resting_state_fMRI#Functional) can then be used to fit the model to empirical functional brain data. 

## Installation
The easiest way to get going is to install the pypi release of `neurolib` using

```
pip install neurolib
```
Alternatively, you can also clone this repository and install all dependencies with

```
git clone https://github.com/neurolib-dev/neurolib.git
cd neurolib/
pip install -r requirements.txt
```

## Usage
Example iPython notebooks on how to use the library can be found in the `./examples/` directory. A basic overview is given here. 

### Single node
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

### Whole-brain network

To simulate a whole-brain network model, first we need to load a DTI and a resting-state fMRI dataset (an example dataset called `gw` is provided in the `neurolib/data/datasets/` directory).

```python
from neurolib.utils.loadData import Dataset

ds = Dataset("gw")
```

We can now initialise the model with the dataset:

```python
aln = ALNModel(Cmat = ds.Cmat, Dmat = ds.Dmat, simulateBOLD=True)
aln.params['duration'] = 5*60*1000 # in ms, simulates for 5 minutes

aln.run()
```
This can take several minutes to compute, since we are simulating 90 nodes for 5 minutes realtime. Here, we have created a network model in which each brain area is an `aln` node. Note that we specified `simulateBOLD=True`, which simulates the BOLD model in parallel to the firing rate model.

The quality of the fit of this simulation to the functional empirical data can now be computed per subject or for the whole group on average:

```python
scores = []
for i in range(len(ds.FCs)):
    fc_score = func.matrix_correlation(func.fc(alnModel.BOLD[:, 5:]), ds.FCs[i]) 
    scores.append(fc_score)
    print("Subject {}: {:.2f}". format(i, fc_score))
print("Mean simulated FC to empirical FC correlation: {:.2f}".format(np.mean(scores)))
```
```
Subject 0: 0.71
Subject 1: 0.70
Subject 2: 0.52
Subject 3: 0.56
Subject 4: 0.51
Subject 5: 0.60
Subject 6: 0.64
Subject 7: 0.65
Subject 8: 0.36
Subject 9: 0.54
Subject 10: 0.49
Mean simulated FC to empirical FC correlation: 0.57
```

