#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to reconstruct keyhole data from a fid + procpar files

print ''
print '#########################################'
print 'Importing libraries'
print '#########################################'
print ''

import numpy as np
import nibabel as nib
from variantools import *
import multiprocessing as mp

# import sys
# input_path = sys.argv[1]
# module_path = sys.argv[0][:-10]
# output = sys.argv[2]

input_path = '/home/vincent/Maitrise/Data/metv_20141124_001/data/gems_ep04_01.fid'
module_path = '/home/vincent/Programmation/Python/varian_tools'
output_path = 'test.nii.gz'

print ''
print '#########################################'
print 'Extract fid data and procpar parameters'
print '#########################################'
print ''

par = parseprocpar(input_path + '/procpar')
data = loadfid(input_path + '/fid')

print ''
print '#########################################'
print 'Get PETable and necessary scan infos'
print '#########################################'
print ''

# Get phase encode table from the scan parameters
petable = read_petable(module_path + '/' + par['petable'])
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
    # Position points to non-nan elements (which has been acquired)
    position = -np.isnan(abs(time_serie))
    # For center of k-space, no keyhole interpolation required
    if -np.any(np.isnan(time_serie)):
        return time_serie

    # Interpolation is linear, with numpy function 'interp'
    time_frames = time_serie.shape[0]
    k0 = time_serie[position]
    t0 = np.arange(time_frames)[position]
    t1 = np.arange(time_serie.shape[0])
    fr = np.interp(t1, t0, np.real(k0), np.real(k0[0]), np.real(k0[-1]))
    fi = np.interp(t1, t0, np.imag(k0), np.imag(k0[0]), np.imag(k0[-1]))
    k1 = fr + fi*1j
    return k1

kspace = np.apply_along_axis(interp_keyhole, 0, kspace)

print ''
print '#########################################'
print 'Correct for DC offset'
print '#########################################'
print ''

for i in range(5):
    kspace -= kspace.mean()

print ''
print '#########################################'
print '2D Fourier Transform'
print '#########################################'
print ''

image_tmp = np.abs((np.fft.fft2(kspace)))
image_tmp = np.fft.fftshift(image_tmp, axes=(2,3))
image_tmp = image_tmp.transpose([2,3,1,0])

# Image dimensions are now: x(phase), y(frequency), z(slice), t(time)

print ''
print '#########################################'
print 'Reorder slices'
print '#########################################'
print ''

image = np.empty(image_tmp.shape)
interleave_order = range(ns)
interleave_order = interleave_order[::2] + interleave_order[1::2]
for z in range(ns):
    image[:,:,interleave_order[z], :] = image_tmp[:,::-1, z, :]

print ''
print '#########################################'
print 'Saving image in', output_path
print '#########################################'
print ''

affine = np.eye(4)
dx, dy, dz = 0.250, 0.250, 1.000
affine[np.eye(4) == 1] = [dx, dy, dz, 1]
nifti = nib.Nifti1Image(image, affine)
nib.save(nifti, output_path)
