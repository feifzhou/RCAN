import numpy as np
from random import seed
from random import random
from datetime import datetime


def variance(x):
    return 0.1*x+0.1


def addLine(en,spec,xp,yp,var):
    g = yp/np.sqrt(2.0*np.pi*var)*np.exp(-0.5*(en-xp)**2/var)
    spec = spec + g
    return spec

def printSpec(en,spec,color):
    for (e,s) in zip(en,spec):
        print(e,s,color)

# seed random number generator
seed(datetime.now())
def generate_1spectrum(n, enMin, enMax, varMax, intMin, intMa,
                      N, nPeaks, varMin, energy, mat):
    spec=np.zeros_like(en)
#    # seed random number generator
#    seed(datetime.now())
    # generate some random numbers

    for i in range(nPeaks):
        xp = random()*(enMax-enMin)+enMin
        yp = random()*(intMax-intMin)+intMin
        var = random()*(varMax-varMin)+varMin
        spec=addLine(en,spec,xp,yp,var)
    specHR = spec.astype(np.float32)
    specLR = np.dot(mat[:n],specHR)

    return specHR[None,:], specLR



enMin = 10.0
enMax = 100.0
varMin = 0.1
varMax = 3.0
intMin = 0.1
intMax = 1.0
N=1000
nPeaks = 20

dE=(enMax-enMin)/(N-1)
en=np.arange(enMin,enMax+dE,dE).astype(np.float32)

#mat=np.zeros((N,N))
#for i, e in enumerate(en):
#    var = variance(e)
#    g = 1.0/np.sqrt(2.0*np.pi*var)*np.exp(-0.5*(en-e)**2/var)
#    mat[:,i] = g
var=variance(en[None,:])
mat = 1.0/np.sqrt(2.0*np.pi*var*N)*np.exp(-0.5*(en[:,None]-en[None,:])**2/var)
mat=mat.astype(np.float32)


def variance2(x):
    return 11.0+np.sqrt(x*0.4)

var=variance2(en[None,:])
mat2 = 1.0/np.sqrt(2.0*np.pi*var*N)*np.exp(-0.5*(en[:,None]-en[None,:])**2/var)
mat2=mat2.astype(np.float32)

matN= np.array([mat, mat2])

#s1,s2 = generate_1spectrum(enMin, enMax, varMax, intMin, intMax,
#                           N, nPeaks, varMin, en, mat)
#printSpec(en,s1,1)
#print()
#printSpec(en,s2,2)

def generate(n=1):
    return generate_1spectrum(n, enMin, enMax, varMax, intMin, intMax,
                           N, nPeaks, varMin, en, matN)

