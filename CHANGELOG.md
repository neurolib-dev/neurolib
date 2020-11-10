**v0.5.10**

- models now have the parameter `sampling_dt` which will downsample the output to a specified step size (in ms)
- loadData: add subject-wise length matrices `ds.Dmats`

**v0.5.9**

- `ALN` model added to the multimodel framework
- `ThalamicMassModel` now works with autochunk for very long simulations with minimal RAM usage!

**v0.5.8**

- Hotfix: include `pypet_logging.ini` in pypi package
- Evolution: new method `getIndividualFromHistory()`

**v0.5.7**

- `example-0.5`: Demonstrating the use of external stimuli on brain networks
- `example-1.3`: 2D bifurcation diagrams using `pypet`
- `bold`: BOLD numerical overflow bug fixed
- `evolution`: dfEvolution and dfPop fix
- `exploration`: fix seed for random initial conditions
- various minor bugfixes

**v0.5.5**

- Hotfix for RNG seed in exploration: Seed `None` is now converted to `"None"` for for `pypet` compatibility only when saving the `model.params` to the trajectory. 
- Fix: `dfEvolution` drops duplicate entries from the `evolution.history`.

**v0.5.4**

- New function `func.construct_stimulus()` 
- New example of stimulus usage: `examples/example-0.5-aln-external-stimulus.ipynb`
- Fixed RNG seed bug, where the seed value None was converted to 0 (because of pypet) and lead to a predictable random number generator

**v0.5.3**

- `ALNModel` now records adaptation currents! Accessible via model.outputs.IA

**v0.5.1**

*Evolution:*

- NSGA-2 algorithm implemented (Deb et al. 2002)
- Preselect complete algorithms (using `algorithm="adaptive"` or `"nsga2"`)
- Implement custom operators for all evolutionary operations
- Keep track of the evolution history using `evolution.history`
- Genealogy `evolution.tree` available from `evolution.buildEvolutionTree()` that is `networkx` compatible [1]
- Continue working: `saveEvolution()` and `loadEvolution()` can load an evolution from another session [2]
- Overview dataframe `evolution.dfPop` now has all fitness values as well
- Get scores using `getScores()`
- Plot evolution progress with `evolutionaryUtils.plotProgress()`

*Exploration:*

- Use `loadResults(all=True)` to load all simulated results from disk to memory (available as `.results`) or use `all=False` to load runs individually from hdf. Both options populate `dfResults`.
- `loadResults()` has memory cap to avoid filling up RAM
- `loadDfResults()` creates the parameter table from a previous simulation
- `explorationUtils.plotExplorationResults()` for plotting 2D slices of the explored results with some advanced functions like alpha maps and contours for predefined regions.

*devUtils*

- A module that we are using for development and research with some nice features. Please do not rely on this file since there might be breaking changes in the future.
   - `plot_outputs()` like a true numerical simlord
   - `model_fit()` to compute the model's FC and FCD fit to the dataset, could be usefull for everyone
   - `getPowerSpectrum()` does what is says
   - `getMeanPowerSpectrum()` same
   -  a very neat `rolling_window()` from a `numpy` PR that never got accepted

*Other:*

- Data loading:
    - `Dataset` can load different SC matrix normalizations: `"max", "waytotal", "nvoxel"`
    - Can precompute FCD matrices to avoid having to do it later (`fcd=True`)
- `neurolib/utils/atlas.py` added with aal2 region names (thanks @jajcayn) and coordinates of centers of regions (from scans of @caglorithm's  brain 🤯)
- `ParameterSpace` has `.lowerBound` and `.upperBound`.
- `pypet` finally doesn't create a billion log files anymore due to a custom log config

**v0.5.0**

- **New model**: Thalamus model `ThalamicMassModel` (thanks to @jajcayn)
  - Model by Costa et al. 2016, PLOS Computational Biology
- New tools for parameter exploration: `explorationUtils.py` aka `eu`
  - Postprocessing of exploration results using `eu.processExplorationResults()`
  - Find parameters of explored simulations using `eu.findCloseResults()`
  - Plot exploration results via `eu.plotExplorationResults()` (see example image below)
- Custom transformation of the inputs to the `BOLDModel`. 
  - This is particularly handy for phenomenological models (such as `FHNModel`, `HopfModel` and `WCModel`) which do not produce firing rate outputs with units in `Hz`.
- Improvements
  - Models can now generate random initial conditions using `model.randomICs()`
  - `model.params['bold'] = True` forces BOLD simulation
  - `BoxSearch` class: `search.run()` passes arguments to `model.run()`
  - BOLD output time array renamed to `t_BOLD`

**v0.4.1**
  
- **New model:** Wilson-Cowan neural mass model implemented (thanks to @ChristophMetzner )
- Simulations now start their output at `t=dt` (as opposed to `t=0` before). Everything before is now considered an initial condition.
- Fix: Running a simulation chunkwise (using `model.run(chunkwise=True)`) and normally (using `model.run()`) produces output of the same length
- Fix: `aln` network coupling, which apparent when simulating chunkwise with `model.run(chunkwise=True, chunksize=1)`
- Fix: Correct use of seed for RNG
- Fix: Matrices are not normalized to max-1 anymore before each run.
- Fix: Kolmogorov distance of FCD matrices and timeseries
