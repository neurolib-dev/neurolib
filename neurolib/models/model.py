
import logging 

class Model:
    # I/O
    inputNames = []
    inputs = []
    nInputs = 0

    outputNames = []
    outputs = []
    nOutputs = 0
    def __init__(self, name, description = None):
        assert isinstance(name, str), f"name {name} is not a string"
        self.name = name

        logging.info(f"Model {name} created")
    
    def addOutputs(self, outputs, outputNames = None):
        # if no names are provided, make up names
        # if outputs is a list
        if outputNames == None and isinstance(outputs, list):
            outputNames = [self.name + "-output-" + str(self.nOutputs + i) for i in range(len(outputs))]
        elif outputNames == None:
            outputNames = [self.name + "-output-" + self.nOutputs]

        # sanity check
        if len(outputs) == len(outputNames):
            self.nOutputs += len(outputs)
            # add outputs
            self.outputs = self.outputs + outputs
            self.outputNames = self.outputNames + outputNames
        logging.info(f"{len(outputs)}/{self.nOutputs} outputs added: {outputNames}")

    def addInputs(self, inputs, inputNames = None):
        # if no names are provided, make up names
        # if inputs is a list
        if inputNames == None and isinstance(inputs, list):
            inputNames = [self.name + "-input-" + str(i) for i in range(len(inputs))]
        elif inputNames == None:
            inputNames = self.name + "-input"

        # sanity check
        if len(inputs) == len(inputNames):
            # add inputs
            self.inputs = self.inputs + inputs
            self.inputNames = self.inputNames + inputNames


