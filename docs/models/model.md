# Models

Models are the core of `neurolib`. The `Model` superclass will help you to load, simulate, and analyse models. It also makes it very easy to implement your own neural mass model (see [Example 0.6 custom model](/examples/example-0.6-custom-model/)).

## Loading a model
To load a model, we need to import the submodule of a model and instantiate it. This example shows how to load a single node of the `ALNModel`. See [Example 0 aln minimal](/examples/example-0-aln-minimal/) on how to simulate a whole-brain network using this model.


```
from neurolib.models.aln import ALNModel # Import the model
model = ALNModel() # Create an instance
model.run() # Run it
```

## Model base class methods

::: neurolib.models.model.Model