# Contributing to neurolib

Thank you for your interest in contributing to `neurolib`. We welcome bug reports through the issues tab and pull requests for fixes or improvements. You are warlmy invited to join our development efforts and make brain network modeling easier and more useful for all researchers.

## Pull requests

To propose a change to `neurolib`'s code, you should first clone the repository to your own Github account. 
Then, create a branch and make some changes. You can then send a pull request to neurolib's own repository 
and we will review and discuss your proposed changes. 


More information on how to make pull requests can be found in the 
[Github help](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) pages.

### Maintaining code

Please be aware that we have a conservative policy for implementing new functionality. All new features need to be maintained, sometimes forever. We are a small team of developers and can only maintain a limited amount of code. Therefore, ideally, you should also feel responsible for the changes you have proposed and maintain it after it becomes part of `neurolib`. 

## Code style

We are using the [black](https://github.com/psf/black) code formatter with the additional argument `--line-length=120`. 
It's called the "uncompromising formatter" because it is completely deterministic and you have literally no control over how your code will look like. 
We like that! We recommend using black directly in your IDE, 
for example in [VSCode](https://marcobelo.medium.com/setting-up-python-black-on-visual-studio-code-5318eba4cd00).

### Commenting Code

We are using the [sphinx format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) for commenting code. Comments are incredibly important to us since `neurolib` is supposed to be a library of user-facing code. It's encouraged to read the code, change it and build something on top of it. Our users are coders. Please write as many comments as you can, including a description of each function and method and its arguments but also single-line comments for the code itself. 

## Implementing a neural mass model

You are very welcome to implement your favorite neural mass model and contribute it to `neurolib`. 

* The easiest way of implementing a model is to copy a model directory and adapt the relevant parts of it to your own model. Please have a look of how other models are implemented. We recommend having a look at the `HopfModel` which is a fairly simple model.
* All models inherit from the `Model` base class which can be found in `neurolib/models/model.py`.
* You can also check out the [model implementation example](https://neurolib-dev.github.io/examples/example-0.6-custom-model/) to find out how a model is implemented.
* All models need to pass tests. Tests are located in the `tests/` directory of the project. A model should be added to the test files `tests/test_models.py` and `tests/test_autochunk.py`. However, you should also make sure that your model supports as many `neurolib` features as possible, such as exploration and optimization. If you did everything right, this should be the case.
* As of now, models consist of three parts:
  * The `model.py` file which contains the class of the model. Here the model specifies attributes like its name, its state variables, its initial value parameters. Additionally, in the constructor (the `__init__()` method), the model loads its default parameters.
  * The `loadDefaultParams.py` file contains a function (`loadDefaultParams()`) which has the arguments `Cmat` for the structural connectivity matrix, `Dmat` for the delay matrix and `seed` for the seed of the random number generator. This function returns a dictionary (or `dotdict`, see `neurolib/utils/collections.py`) with all parrameters inside.
  * The `timeIntegration.py` file which contains a `timeIntegration()` function which has the argument `params` coming from the previous step. Here, we need to prepare the numerical integration. We load all relevant parameters from the `params` dictionary and pass it to the main integration loop. The integration loop is written such that it can be accelerated by `numba` ([numba's page](https://numba.pydata.org/)) which speeds up the integration by a factor of around 1000. 

## Contributing examples 

We very much welcome example contributions since they help new users to learn how to make use of `neurolib`. They can include basic usage examples or tutorials of `neurolib`'s features, or a demonstration of how to solve a specific scientific task using neural mass models or whole-brain networks.

* Examples are provided as Jupyter Notebooks in the `/examples/` directory of the project repository.
* Notebooks should have a brief description of what they are trying to accomplish at the beginning.
* It is recommended to change the working directory to the root directory at the very beginning of the notebook (`os.chdir('..')`).
* Notebooks should be structured with different subheadings (Markdown style). Please also describe in words what you are doing in code. 


## Contributing brain data

We have a few small datasets already in neurolib so everyone can start simulating right away. If you'd like to contribute more data to the project, please feel invited to do so. We're looking for more structural connectivity matrices and fiber length matrices in the MATLAB matrix `.mat` format (which can be loaded by `scipy.loadmat`). We also appreciate BOLD data, EEG data, or MEG data. Other modalities could be useful as well. Please be aware that the data has to be in a parcellated form, i.e., the brain areas need to be organized according to an atlas like the AAL2 atlas (or others).
