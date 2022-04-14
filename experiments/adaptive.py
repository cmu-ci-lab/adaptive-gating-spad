import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.interpolate import RectBivariateSpline, bisplrep, bisplev

import time
from scipy import stats
import time
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import pyvista as pv
import cv2
from numba import jit
from scipy import optimize

@jit(nopython=False)
def ForwardModel_PulseStart_pdf(Lambda):
  CurrTime = 0
  Hist = np.zeros(Lambda.size)
  csum = np.cumsum(Lambda)   
  Hist = np.exp(-np.roll(csum, 1)) * (1-np.exp(-Lambda))
  Hist[0] = (1-np.exp(-Lambda[0]))
  return Hist

def ForwardModel_PulseStart_mat(Lambdas):
  CurrTime = 0
  Hist = np.zeros(Lambdas.size)
  csum = np.cumsum(Lambdas, axis = 1)   
  Hist = np.exp(-np.roll(csum, 1, axis = 1)) * (1-np.exp(-Lambdas))
  Hist[:,0] = (1-np.exp(-Lambdas[:,0]))
  return Hist

@jit(nopython=False)
def square_error(est, truth, l):
  if np.abs(est - truth) > l/2:
    return (l - np.abs(est - truth))**2
  else:
    return (est - truth)**2

def gen_trans(depth, phi_bkg, phi_sig, c = 500, mode = "constant"):
  if mode == "constant":
    phi_sig = phi_sig
  else:
    phi_sig = phi_sig * ((1/((depth/c) + 1))**2)
  ret = np.zeros(500)
  ret[depth] = 1
  ret = gaussian_filter(ret, sigma = 1.8, mode = 'wrap')
  ret = (phi_sig - phi_bkg)*(ret/np.max(ret))
  ret = ret + phi_bkg
  return ret


def gen_Ms(phi_bkg, phi_sig):
  M = np.zeros((500,50,500+1))

  tmp = np.zeros((500, 500))
  for i in range(500):
    tmp[i] = gen_trans(i, phi_bkg, phi_sig)
    
  for j in range(20):
    curr_mat = np.roll(tmp, -j*25, axis = 1)
    curr_mat = ForwardModel_PulseStart_mat(curr_mat)
    M[:,j,1:] = curr_mat
    M[:,j,0] = 1 - np.sum(curr_mat, axis = 1)

  M = np.where(M != 0, np.log(M), 0)
  return M


@jit(nopython=False)
def sample_transient(trans):
  sample = np.random.uniform(0, 1)
  if sample > np.sum(trans):
    return -1
  else:
    return np.argmax(np.cumsum(trans) >= sample)

@jit(nopython=False)
def isample(pdf):
  if np.sum(pdf) == 0:
    pdf = np.ones(500)/500
  pdf = np.cumsum(pdf)
  ran = np.random.uniform(0, 1)
  ret = np.argmax(pdf >= ran)
  return ret

@jit(nopython=False)
def adapt_shift(prior, M, N, Ms, dead_time = 480, mode = 0):
  numer = np.zeros(500)
  denom = np.zeros(500)
  
  shifts= np.zeros(500)
  pdf = prior
  log_pdf = np.log(pdf)
  shift = isample(pdf)
  entropy = -np.sum(np.log(pdf)*pdf)
  gate = 0
  count = 0
  scount = 0
  elapsed = 0
  ent_list = []
  detects = np.zeros(500)

  while elapsed < N:
    curr_trans = M[shift//25]
    sample = sample_transient(curr_trans)
    start = 25*(shift//25) 
    
    if sample != -1:
      if sample > start:
        denom[start:sample+1] += 1
        numer[sample] += 1
      else:
        denom[start:] += 1
        denom[:sample+1] += 1
        numer[sample] += 1
    
    
    if sample != -1:
      log_pdf += Ms[:,start//25,((sample+500-start)%500)+1]
    
    log_pdf = log_pdf - np.min(log_pdf)

    pdf = np.exp(log_pdf)
    pdf = pdf/np.sum(pdf)
    
    elapsed += dead_time + (sample+500-start)%500
    
    gate = isample(pdf/np.sum(pdf))
    shift = (gate+490)%500
    
    if mode == 1:
      shift = elapsed%500
    else:
      elapsed += (shift + 500 - (elapsed%500))%500
      scount += 1
    count += 1
    prev_entropy = entropy
    entropy = -np.sum(np.log(pdf)*pdf)
    ent_list.append(entropy)
  return log_pdf, elapsed, numer, denom

def adapt_exposure(prior, M, N, Ms, dead_time = 480, mode = 0):
  numer = np.zeros(500)
  denom = np.zeros(500)
  
  shifts= np.zeros(500)
  pdf = prior
  log_pdf = np.log(pdf)
  shift = isample(pdf)
  entropy = -np.sum(np.log(pdf)*pdf)
  gate = 0
  count = 0
  scount = 0
  elapsed = 0
  ent_list = []
  detects = np.zeros(500)

  while entropy > N:
    curr_trans = M[shift//25]
    sample = sample_transient(curr_trans)
    start = 25*(shift//25) 
    
    if sample != -1:
      if sample > start:
        denom[start:sample+1] += 1
        numer[sample] += 1
      else:
        denom[start:] += 1
        denom[:sample+1] += 1
        numer[sample] += 1
    
    
    if sample != -1:
      log_pdf += Ms[:,start//25,((sample+500-start)%500)+1]
    
    log_pdf = log_pdf - np.min(log_pdf)

    pdf = np.exp(log_pdf)
    pdf = pdf/np.sum(pdf)
    
    elapsed += dead_time + (sample+500-start)%500
    
    gate = isample(pdf/np.sum(pdf))
    shift = (gate+490)%500
    
    if mode == 1:
      shift = elapsed%500
    else:
      elapsed += (shift + 500 - (elapsed%500))%500
      scount += 1
    count += 1
    prev_entropy = entropy
    entropy = -np.sum(np.log(pdf)*pdf)
    ent_list.append(entropy)
  return log_pdf, elapsed, numer, denom

@jit(nopython=False)
def get_trans(stack):
    bkg = 0
    denom = 0
    for i in range(20):
        denom += np.sum(stack[i])
        bkg += stack[i,i*25]
    bkg = bkg/denom
    bkg = np.log(1/ (1-bkg))
    summed = np.sum(stack, axis = 0)
    sig = bkg*np.max(summed)/np.mean(summed)
    td = np.argmax(summed)
    return bkg, sig, td

def gen_front(true_depth, bkg, sig):
    front = np.zeros((100,500))
    for i in range(100):
      front[i] = 500000*np.roll(ForwardModel_PulseStart_pdf(gen_trans((true_depth + 500 - i*5)%500, bkg, sig, mode = "constant")), i*5)
    return front

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)