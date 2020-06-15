from neurolib.models.aln import ALNModel
from matplotlib import pyplot as plt
import numpy as np
import h5py as h5
import sys
from neurolib.models.aln import loadDefaultParams
from neurolib.utils.loadData import Dataset

ds = Dataset("hcp")
inp = sys.argv[1]


if inp == '0':
    
    aln = ALNModel(Cmat = ds.Cmat, Dmat = ds.Dmat)
    aln.params.duration = 2.0*1000 #3.0*60 * 1000
    aln.params.mue_ext_mean = 1.57
    aln.params.mui_ext_mean = 1.6
    aln.params.sigma_ou = 0.09
    aln.params.b = 5.0
    aln.run()
    
    fig = plt.figure()
    plt.imshow(aln.params.lengthMat, cmap='hot', interpolation = 'nearest')
    plt.show()

elif inp == 'a':
    aln = ALNModel()
    
    aln.params.duration = 2.0 * 1000
    
    max_out = []
    min_out = []
    
    inputs = np.linspace(0,2,50)
    
    for mue in inputs:
        aln.params.mue_ext_mean = mue
    
        aln.run()
        
        max_out.append(np.max(aln.output[0, -int(1000/aln.params.dt):]))
        min_out.append(np.min(aln.output[0, -int(1000/aln.params.dt):]))
    
    #print(max_out)
    #print(min_out)
    
    fig = plt.figure()
    plt.plot(inputs, max_out)
    plt.plot(inputs, min_out)
    plt.show()
elif inp == 'b':
    fil = input('enter file')
    hdf = h5.File(fil,'r')
    print(hdf[0])

elif inp == 'c':

    stuff = loadDefaultParams.loadDefaultParams()


    print(stuff)
