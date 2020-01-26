import time
print("> Running tests ...")
testingStart = time.time()

# ----------------------------------------
print("\t > ALN: Testing single node ...")
start = time.time()
from neurolib.models import aln

alnModel = aln.ALNModel()
alnModel.params['duration'] = 2.0*1000
alnModel.params['sigma_ou'] = 0.1 # add some noise

alnModel.run()
end = time.time()
print("\t > Done in {:.2f} s".format(end - start))

# ----------------------------------------
print("\t > ALN: Testing brain network ...")
start = time.time()
from neurolib.utils.loadData import Dataset

ds = Dataset("gw")

alnModel = aln.ALNModel(Cmat = ds.Cmat, Dmat = ds.Dmat, simulateBOLD=True)
alnModel.params['duration'] = 10*1000 # in ms, simulates for 5 minutes

alnModel.run()
end = time.time()
print("\t > Done in {:.2f} s".format(end - start))

# ----------------------------------------
print("\t > Hopf: Testing single node ...")
start = time.time()
from neurolib.models import hopf

hopfModel = hopf.HopfModel()
hopfModel.params['duration'] = 2.0*1000
hopfModel.params['sigma_ou'] = 0.03

hopfModel.run()
end = time.time()
print("\t > Done in {:.2f} s".format(end - start))

# ----------------------------------------
print("\t > Hopf: Testing brain network ...")
start = time.time()
hopfModel = hopf.HopfModel(Cmat = ds.Cmat, Dmat = ds.Dmat, simulateChunkwise=False)
hopfModel.params['w'] = 1.0
hopfModel.params['signalV'] = 0
hopfModel.params['duration'] = 10 * 1000 
hopfModel.params['sigma_ou'] = 0.14
hopfModel.params['K_gl'] = 0.6

hopfModel.run()
end = time.time()
print("\t > Done in {:.2f} s".format(end - start))
# ----------------------------------------

end = time.time()
print("> All tests passed in {:.2f} s!".format(end - testingStart))