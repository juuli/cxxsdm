# -*- coding: utf-8 -*-

################################################################################
# This file is a part of the CxxSDM spatial decomposition
# library. It is released under the MIT License. You should have
# received a copy of the MIT License along with CxxSDM.  If not, see
# http://www.opensource.org/licenses/mit-license.php
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# For details, see the LICENSE file
#
# (C) 2018 Jukka Saarelma
################################################################################

import numpy as np
import scipy.io as sio
import time
import matplotlib.pyplot as plt
import ctypes as ct
from numpy.ctypeslib import ndpointer

import plotFunctions as pf

def analyzeIRs(irs, mic_locs, fs, frame_len, c=344.0):
  # It is assumed that the dynamic library libSDM locates in the same folder
  # as this script, so here we grab the folder path
  import os
  from sys import platform
  folder =  os.path.dirname(os.path.abspath(__file__))

  # Load sdm extern "C" functions from the library
  if(os.name == 'nt'):
    sdm = ct.cdll.LoadLibrary(folder + "/SDM.dll")
  else:
    if(platform=='darwin'):
      sdm = ct.cdll.LoadLibrary(folder + "/libSDM.dylib")
    else:
      sdm = ct.cdll.LoadLibrary(folder + "/libSDM.so")

  # Init input and output argument types
  sdm.initialize.restype = ct.c_void_p

  sdm.processIRs.argtypes = [ct.c_void_p,
                             ndpointer(ct.c_float, flags="F_CONTIGUOUS"),
                             ndpointer(ct.c_float, flags="F_CONTIGUOUS"),
                             ndpointer(ct.c_float, flags="F_CONTIGUOUS")]

  # mic location coordinates are handled row-major in the library,
  # IRs and return vectors are column-major
  mic_locs = mic_locs.astype(np.float32)
  num_mics = np.shape(mic_locs)[0]

  irs = irs.astype(np.float32)
  resp_len = np.shape(irs)[0]

  # Force column major memory order for the impulse responses
  if(not np.isfortran(irs)):
    irs = np.asfortranarray(irs)

  # Initialize return values
  ret_az = np.zeros((resp_len, 1), dtype=np.float32, order='F')
  ret_el = np.zeros((resp_len, 1), dtype=np.float32, order='F')

  ## Init the sdm block
  # If many responses are analyzed, it may be beneficial to separate
  # the initialization and actual processing to improve performance
  sd_ptr = sdm.initialize(ct.c_uint64(fs), ct.c_uint64(frame_len),
                          ct.c_uint64(resp_len),
                          mic_locs.ctypes.data_as(ct.POINTER(ct.c_float)),
                          ct.c_uint64(num_mics))

  # Run analysis
  sdm.processIRs(ct.c_void_p(sd_ptr),
                 irs,
                 ret_az,
                 ret_el)

  sdm.destroy(ct.c_void_p(sd_ptr))

  loc = np.zeros((resp_len, 3), dtype = np.float32)
  loc[:,0] = irs[:,0]
  loc[:,1] = ret_az[:,0]
  loc[:,2] = ret_el[:,0]

  loc[np.isnan(loc[:,1]), 1] = 0.0
  loc[np.isnan(loc[:,2]), 2] = 0.0
  return loc


# Load a array response
data = sio.loadmat("IR_living_room.mat")
irs = data['ir_left'].astype(np.float32)

# irs column-major
resp_len = np.shape(irs)[0]

# Setup parameters
fs = 192000
frame_len = 36
m_dim = 2.5*1e-2

# mic_locs row major and single precision
mic_locs = np.array([[m_dim, 0.0, 0.0], [-m_dim, 0.0, 0.0],
                    [0.0, m_dim, 0.0], [0.0, -m_dim, 0.0],
                    [0.0, 0.0, m_dim], [0.0, 0.0, -m_dim]],
                    dtype=np.float32)


# Run SDM analysis
start = time.time()

locs = analyzeIRs(irs, mic_locs, fs, frame_len)

print "Processing done, time elapsed %f s"%(time.time()-start)

# Load reference data calculated with Matlab
ref_data = sio.loadmat('living_room_ref.mat')
ref_az = ref_data['az']/np.pi*180.0
ref_el = ref_data['el']/np.pi*180.0

ref_az[np.isnan(ref_az)] = 0.0
ref_el[np.isnan(ref_el)] = 0.0

# Get pressure values in log scale and normalize to 0 dB
p_dB = 10.0*np.log10(locs[:,0]**2)
p_dB -= np.max(p_dB)

# Get pressure vector above decibel limits
dB_60 = p_dB.copy()+60
dB_30 = p_dB.copy()+30
dB_60[dB_60<0] = 0.0
dB_30[dB_30<0] = 0.0

# Print mean error for pressure values above -30 and -60 dB from max
# Small error should occur especially for small pressure values as
# the backend differs significantly
print "Mean localization difference for pressures above -30 dB: %f deg" % np.mean(np.sqrt((ref_az[:,0]-locs[:,1])**2.0)*np.sign(dB_30))
print "Mean localization difference for pressures above -60 dB: %f deg" % np.mean(np.sqrt((ref_az[:,0]-locs[:,1])**2.0)*np.sign(dB_60))


ang, p11 = pf.getAngularBins(locs[:,1], locs[:,0], 0.011*fs, db_min = -30)
ang, p20 = pf.getAngularBins(locs[:,1], locs[:,0], 0.02*fs, db_min = -30)
ang, p50 = pf.getAngularBins(locs[:,1], locs[:,0], 0.05*fs, db_min = -30)
ang, p1000 = pf.getAngularBins(locs[:,1], locs[:,0], 1.0*fs, db_min = -30)

colors = pf.colorsAtRange(0,4 ,4)

rlim = np.linspace(np.min(p1000), np.max(p1000)+5, 5)
pf.polarPlot(ang, p1000, color=colors[0], fill_color=colors[0], rlim=rlim)
pf.polarPlot(ang, p50, color=colors[1], fill_color=colors[1], rlim=rlim)
pf.polarPlot(ang, p20, color=colors[2], fill_color=colors[2], rlim=rlim)
pf.polarPlot(ang, p11, color=colors[3], fill_color=colors[3], rlim=rlim)