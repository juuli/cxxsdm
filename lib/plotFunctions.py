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
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

def getAngularBins(angles, data, t_end_idx, db_min=-150,
                   ang_vec = np.linspace(-180.0, 181.0, 360)):
  t_end_idx = int(t_end_idx)
  digitized = np.digitize(angles[:t_end_idx], ang_vec)
  cur_data = data[:t_end_idx]
  bin_means = np.array([10.0*np.log10((cur_data[digitized==i]**2).sum()) for i in range(len(ang_vec))])
  bin_means[np.isnan(bin_means)] = db_min
  bin_means[np.isinf(bin_means)] = db_min
  bin_means[bin_means<db_min] = db_min
  return ang_vec, bin_means

def polarPlot(angle, data, fig=None, subplot=(1,1,1), rlim=None,
              lw=1, ls='-', color='k', fill_color=None,
              line_alpha=0.8, fill_alpha=0.8):
  if fig == None:
    plt.figure(0)
  else:
    plt.figure(fig)

  if np.any(rlim) == None:
    rlim = np.linspace(np.min(data), np.max(data)+5, 5.0)

  ax = plt.subplot(subplot[0],subplot[1], subplot[2], polar=True)
  ax.plot(angle/180.0*np.pi, data, ls=ls, lw=lw, color=color, alpha=line_alpha)
  if(fill_color is not None):
    ax.fill(angle/180.0*np.pi, data, color=fill_color, alpha=fill_alpha)

  ax.set_yticks(rlim)
  plt.grid(True, which='both', linestyle='-', color='#CCCCCC')
  ax.set_rlim((rlim[0], rlim[-1]))
  ax.set_axisbelow(True)

def getScalarColormapForRange(min_val, max_val, cmap=plt.get_cmap('viridis')):
  """
   A function to extract scalar colormap from matplotlib colormap

  Arguments
  ---------
  min_val : float/int
    The minimum value to which the color map should span
  max_val : float/int
    The maximum value to which the color map should span
  cmap : matplotlib colormap
    The colormap to be use

  Returns
  -------
  scalar_colormap : matplotlib.cm.ScalarMappable
    An instance of matplotlib scalar colormap
  """
  cNorm  = colors.Normalize(vmin=min_val, vmax=max_val)
  return cmx.ScalarMappable(norm=cNorm, cmap=cmap)


def colorsAtRange(min_val, max_val, N, cmap=plt.get_cmap('viridis')):
  """
  A function to extract discrete color values from a matplotlib colormap

  Arguments
  ---------

  min_val : float/int
    The minimum value to which the color map should span

  max_val : float/int
    The maximum value to which the color map should span

  N : int
    The number of discrete colors between [min_val, max_val] (linear)

  cmap : matplotlib colormap
    The colormap to be used

  Returns
  -------
  clrs : list
    A python list containing the color values
  """

  sm = getScalarColormapForRange(min_val, max_val, cmap)
  rang = np.linspace(min_val, max_val, N)
  clrs = []
  for r in rang:
    clrs.append(sm.to_rgba(r))
  return clrs
