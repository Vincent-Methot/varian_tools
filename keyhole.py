#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to reconstruct keyhole data from a fid + procpar files


print ''
print '#########################################'
print 'Importing libraries'
print '#########################################'
print ''

import numpy as np
# import scipy as sp
from scipy.interpolate import interp1d
import nibabel as nib
from struct import unpack
from nmrglue.fileio.varian import read_procpar
import datetime
import sys
import os
from variantools import *

path = sys.argv[1]

print ''
print '#########################################'
print 'Extract fid data and procpar parameters'
print '#########################################'
print ''

par = parseprocpar(path + '/procpar')
data = loadfid(path + '/fid')

print ''
print '#########################################'
print 'Get PETable and necessary scan infos'
print '#########################################'
print ''

# Get phase encode table from the scan parameters
petable = read_petable(sys.argv[0][:-10] + '/' + par['petable'])
# Count number of zero in petable to know number of frame per acquisition
frames_per_acq = np.sum(petable == 0)
time_frames = frames_per_acq * data.shape[0]
kline_per_frame = par['nv'] / frames_per_acq
ns = par['ns']
print 'Number of slices:', ns
print 'Frames per acquisition:', frames_per_acq
print 'K-Lines per frame:', kline_per_frame

print ''
print '#########################################'
print 'Reorder k-space'
print '#########################################'
print ''

# Prepare kspace array
petable -= petable.min()
kspace = np.empty([time_frames, ns, petable.max() + 1,
                   data.shape[-1]], dtype='complex64')
kspace[...] = np.nan
 
# Optimization possibility (removing for loops)
for b in range(par['arraydim']):
    for f in range(frames_per_acq):
        for k in range(kline_per_frame):
            for s in range(ns):
                petable_index = k + f*kline_per_frame
                time_point = b * frames_per_acq + f
                kline = petable[petable_index]
                kspace[time_point, s, kline, :] = data[b,
                               petable_index*ns + s, :]

print ''
print '#########################################'
print 'Interpolation of kspace'
print '#########################################'
print ''

def interp_keyhole(time_serie):
    """
    Takes a time serie (numpy array) with nans where data has not been
    acquired.
    Array dim: (time point, slice number, phase encode, freq. encode)
    """
    position = -np.isnan(abs(time_serie))
    time_frames = time_serie.shape[0]
    k0 = time_serie[position]
    t0 = np.arange(time_frames)[position]

    if -position[0]:
        k0 = np.append(k0[0], k0)
        t0 = np.append(0, t0)
    if -position[-1]:
        k0 = np.append(k0, k0[-1])
        t0 = np.append(t0, time_frames - 1)
    
    t1 = np.arange(time_serie.shape[0])
    fr = interp1d(t0, np.real(k0),kind='linear')
    fi = interp1d(t0, np.imag(k0),kind='linear')
    k1 = fr(t1) + complex(0,1)*fi(t1)
    return k1

modified_kspace = np.empty(kspace.shape, dtype=np.complex64)
for i in np.ndindex(kspace.shape[1:]):
    if -np.any(np.isnan(kspace[:, i[0], i[1], i[2]])):
        modified_kspace[:, i[0], i[1], i[2]] = kspace[:, i[0], i[1], i[2]]
    else:
        modified_kspace[:, i[0], i[1], i[2]] = interp_keyhole(kspace[:, i[0], i[1], i[2]])
        print 'Interpolating kspace data point:', 'slice', i[0], 'kpoint', i[1], i[2]

print ''
print '#########################################'
print 'Correct for DC offset'
print '#########################################'
print ''

for i in range(5):
    modified_kspace -= modified_kspace.mean()

print ''
print '#########################################'
print '2D Fourier Transform'
print '#########################################'
print ''

image_tmp = np.abs((np.fft.fft2(modified_kspace)))
image_tmp = np.fft.fftshift(image_tmp, axes=(2,3))
image_tmp = image_tmp.transpose([2,3,1,0])


# Image x(phase), y(frequency), z(slice), t(time)

# print ''
# print '#########################################'
# print 'Reorder slices'
# print '#########################################'
# print ''
# 
# image = np.empty(image_tmp.shape)
# interleave_order = range(ns)
# interleave_order = interleave_order[::2] + interleave_order[1::2]
# for z in range(ns):
    # image[:,:,interleave_order[z], :] = image_tmp[:,::-1, z, :]
# 
image = image_tmp


print ''
print '#########################################'
print 'Saving image in "' + sys.argv[2] + '"'
print '#########################################'
print ''


affine = np.eye(4)
dx, dy, dz = 0.250, 0.250, 1.000
affine[np.eye(4) == 1] = [dx, dy, dz, 1]
nifti = nib.Nifti1Image(image, affine)
nib.save(nifti, sys.argv[2])


