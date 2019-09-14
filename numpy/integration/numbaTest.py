import numpy as np
import math
from numba import vectorize,njit
@vectorize('float64(float64)',target='parallel')
def computeRiemanNumbaVec(dx):
    x = np.arange(dx/2,np.pi/2-dx/2,dx)
    S = np.sum(np.sin(x))*dx
    return S
@njit(parallel = True)
def computeRiemanNumbaJit(dx):
    x = np.arange(dx/2,np.pi/2-dx/2,dx)
    S = np.sum(np.sin(x))*dx
    return S
def computeRieman(dx):
    x = np.arange(dx/2,np.pi/2-dx/2,dx)
    S = np.sum(np.sin(x))*dx
    return S
def func1():
    for dx in np.geomspace(0.001,1,200):
        computeRieman(dx)
def func2():
    for dx in np.geomspace(0.001,1,200):
        computeRiemanNumbaJit(dx)
def func3():
   dx = np.geomspace(0.001,1,200)
   computeRiemanNumbaVec(dx)
computeRiemanNumbaJit(1.0)
func1()
func2()
func3()