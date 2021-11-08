#!/usr/bin/env python3.8

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from os.path import splitext
import argparse
from scipy.stats import scoreatpercentile
from matplotlib.gridspec import GridSpec

figsize = (19, 7.5)
plt.rc('text', usetex=False)

parser = argparse.ArgumentParser(description='make a triangle plot and plot chain history, and plot linear, parabola, ... etc. models on top of the provided data', epilog='parameters are assumed to be a + bX + cX^2 + ..., where each line of the text file is:  a  b  c  ...\nIf x0 is given this model is used instead: a + b(X-X0) + c(X-X0)^2 + ...')
parser.add_argument('chain_name', metavar='MCMC_CHAIN.out', help='text file with one chain link per line')
parser.add_argument('--ncols', '-n', type=int, help='how many numbers/parameters to read from each line', default=0)
parser.add_argument('--data', '-d', help='data file:  x  y  [dy]')
parser.add_argument('--x0', type=float, default=0.0, help='')
args = parser.parse_args()

filename = args.chain_name
root, ext = splitext(filename)
filename_out = root + '.png'

usecols = None
if args.ncols > 0: usecols = range(args.ncols)
chain = np.loadtxt(filename, usecols=usecols)
chain_len, ncols = chain.shape

plot_data = False
data_range = (0.0, 1.0)
if args.data is not None:
    try:
      try:
        X, Y, dY = np.loadtxt(args.data, usecols=(0, 1, 2), dtype=float, unpack=True)
      except IndexError as e:
        X, Y = np.loadtxt(args.data, usecols=(0, 1), dtype=float, unpack=True)
        dY = None
    except IOError as e:
      print(e)
      print('Cannot read file: {}'.format(args.data))
      sys.exit(1)
    print('Read {} containing {} data points in {} columns'.format(args.data, len(X), 2 if dY is None else 3))
    plot_data = True
    data_range = (X.min(), X.max())

param_name = ['p{}'.format(i + 1) for i in range(ncols)]

#normalize chain params
#chain.T[0] += chain.T[1] * chain.T[2]
#chain.T[2] = 0.0
#model = lambda x, params: params[0] + params[1] * x + params[2] * x**2


fig = plt.figure(figsize=figsize) #, constrained_layout=True)
#plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99, wspace=0.3, hspace=0.3)
#width_ratios = 
assert figsize[0] > figsize[1]
aspect = figsize[1] / float(figsize[0])
division = 1.0 - (figsize[0] - figsize[1]) / figsize[0]
margin_y1, margin_y2, margin_ymid = 0.09, 0.01, 0.09
margin_x1, margin_x2, margin_xmid = margin_y1 * aspect, margin_y2 * aspect, margin_ymid * aspect
gs_tri   = GridSpec(ncols, ncols, left=margin_x1, right=division, bottom=margin_y1, top=1.0 - margin_y2, wspace=0, hspace=0)
gs_chain = GridSpec(ncols, 2, left=division + margin_x1, right=1.0 - margin_x2, bottom=margin_y1, top=1.0 - margin_y2, wspace=margin_xmid / (1 - division) *2, hspace=0)
#gs_chain = GridSpec(1, 1, left=division + margin_x1, right=1.0 - margin_x2, bottom=margin_y1, top=1.0 - margin_y2, wspace=0, hspace=0)

plt.figtext(division - margin_x2, 1.0 - margin_y2, filename, va='top', ha='right')

############################ triangle plot
for i in range(ncols):
  for j in range(ncols):
      
      if i > j: continue
      ax = fig.add_subplot(gs_tri[j, i])
      #plt.subplot(10, 10, 10 * j + i + 6)
      #plt.axes([0.05, 0.05, 0.9, 0.9])
      #if j == 0: plt.xlabel('a%d' % i)
      ax.tick_params(axis='both', direction='in', which='both', bottom=True, top=True, left=True, right=True)
      if i == 0: 
        plt.ylabel(param_name[j])
      elif i != j: 
        ax.tick_params(axis='y', labelleft=False)
      if j == ncols - 1: 
        plt.xlabel(param_name[i])
      else:
        ax.tick_params(axis='x', labelbottom=False)
      perc16 = scoreatpercentile(chain.T[i], 50 - 34.1)
      perc84 = scoreatpercentile(chain.T[i], 50 + 34.1)
      med = np.median(chain.T[i])
      if i == j:
        plt.hist(chain.T[i], 100, density=True)
        #ax = plt.gca()
        #perc16, med, perc84 = scoreatpercentile(chain.T[i], np.array([50 - 34.1, 50, 50 + 34.1]))
        plt.axvline(med, color='r', ls='--')
        plt.axvspan(perc16, perc84, color='b', alpha=0.1)
        sigma1, sigma2 = perc84 - med, med - perc16
        plt.text(0.95, 0.92, r'$%s = %.3f_{-%.3f}^{+%.3f}$' % (param_name[j], med, sigma1, sigma2) , va='top', ha='right', transform=ax.transAxes) 
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax * 1.3)
        #plt.yticks([])
        ax.tick_params(axis='y', labelleft=False, left=False, right=False)
        continue
      perc16_2 = scoreatpercentile(chain.T[j], 50 - 34.1)
      perc84_2 = scoreatpercentile(chain.T[j], 50 + 34.1)
      med_2 = np.median(chain.T[j])

      #plt.axis([-0.1, 1.7, 2.4, 3.6])
      #plt.gca().set_xlim(slope_lim)
      #plt.gca().set_ylim(level_lim)
      plt.plot(chain.T[i], chain.T[j], 'k.', ms=1, alpha=0.5)
      plt.axvline(med, color='r', ls='--')
      plt.axvspan(perc16, perc84, color='b', alpha=0.1)
      plt.axhline(med_2, color='r', ls='--')
      plt.axhspan(perc16_2, perc84_2, color='b', alpha=0.1)
      #plt.plot( [np.median(chain.T[1])], [np.median(chain.T[0])], 'r+', ms=12)
      #plt.text(0.5, 0.1, '%.2f +- %.2f' % (np.median(chain.T[1]), np.std(chain.T[1])))
      #plt.text(-1.9, 3, '%.2f +- %.2f' % (np.median(chain.T[0]), np.std(chain.T[0])))


############################ trace plots
for i in range(ncols):
  #plt.subplot(10, 1, i + 6)
  ax = fig.add_subplot(gs_chain[i, 1])
  lbl = param_name[i]
  plt.xlabel('time')
  #plt.gca().set_ylim(level_lim)
  plt.plot(chain.T[i], 'k.', ms=1, alpha=0.5)
  #plt.plot(chain_choice_indx, chain_choice.T[i], 'r.', ms=2)
  plt.ylabel(lbl)

  perc16 = scoreatpercentile(chain.T[i], 50 - 34.1)
  perc84 = scoreatpercentile(chain.T[i], 50 + 34.1)
  med = np.median(chain.T[i])
  plt.axhline(med, color='r', ls='--')
  plt.axhspan(perc16, perc84, color='b', alpha=0.1)


############################
# model
def get_model(x, params):
    _ret = x * 0.0
    _x = _ret + 1.0
    _n = len(params)
    for _i, _p in enumerate(params):
      _ret += _p * _x
      if _i < _n - 1: _x *= x
    return _ret

ax = fig.add_subplot(gs_chain[:, 0])

xspan = np.abs(data_range[1] - data_range[0])
x_model = np.linspace(data_range[0] - 0.1 * xspan, data_range[1] + 0.1 * xspan, 50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.tick_params(axis='both', which='both', direction='in', right=True, top=True)

chain_choice_indx = np.random.choice(range(len(chain)), size=100, replace=False)
chain_choice = chain[chain_choice_indx]
v = x_model - args.x0
for params in chain_choice:
  plt.plot(x_model, get_model(v, params), 'r-', alpha=0.05, lw=4)
#plt.axvline(chain[0][2], ls='--', color='gray', lw=1)
#plt.axhline(np.median(chain.T[0]), ls='--', color='gray', lw=1)
#plt.plot(chain.T[0], 'k.', ms=1, alpha=0.5)
#plt.plot(chain_choice_indx, chain_choice.T[0], 'r.', ms=2)
if plot_data:
  if dY is None:
    ax.plot(X, Y, 'ko')
  else:
    ax.errorbar(X, Y, dY, fmt='ko')

#print(get_model(2, [1, 0, 0, 0]))

#ax.set_xlim()

############################
plt.savefig(filename_out)
