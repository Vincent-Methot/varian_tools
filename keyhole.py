#!/usr/bin/env python
# -*- coding: utf-8 -*-

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



print ''
print '#########################################'
print 'Get pre-scan, scan and post-scan folders'
print '#########################################'
print ''

folder = '/home/vincent/Bureau/Conversion_Varian/metv_20141107_001/data/'
prescan_folder = folder + 'gems_ep04_01.fid'
scan_folder = folder + 'gems_ep04_02.fid'
postscan_folder = folder + 'gems_ep04_03.fid'

print 'Work folder', folder
print 'Prescan:', 'gems_ep04_01'
print 'Scan:', 'gems_ep04_02'
print 'Postscan:', 'gems_ep04_03'



print ''
print '#########################################'
print 'Extract fid data and procpar parameters'
print '#########################################'
print ''

prescan_par = parseprocpar(prescan_folder + '/procpar')
scan_par = parseprocpar(scan_folder + '/procpar')
postscan_par = parseprocpar(postscan_folder + '/procpar')
prescan_data = loadfid(prescan_folder + '/fid')
scan_data = loadfid(scan_folder + '/fid')
postscan_data = loadfid(postscan_folder + '/fid')




print ''
print '#########################################'
print 'Check if parameters are compatible'
print '(Not done yet)'
print '#########################################'
print ''



print ''
print '#########################################'
print 'Get PETable and necessary scan infos'
print '#########################################'
print ''

# Get phase encode table from the scan parameters
petable = read_petable(scan_par['petable'])
# Count number of zero in petable to know number of frame per acquisition
frames_per_acq = np.sum(petable == 0)
time_frames = frames_per_acq * scan_data.shape[0]
kline_per_frame = scan_par['nv'] / frames_per_acq
ns = scan_par['ns']
print 'Number of slices:', ns
print 'Frames per acquisition:', frames_per_acq
print 'K-Lines per frame:', kline_per_frame



print ''
print '#########################################'
print 'Reorder k-space'
print '#########################################'
print ''

# Prepare kspace array
kspace = np.empty([time_frames + 2, ns, prescan_par['nv'],
                   scan_data.shape[-1]], dtype='complex64')
kspace[...] = np.nan
petable -= petable.min()
 
# Optimization possibility (removing for loops)
for b in range(scan_par['arraydim']):
    for f in range(frames_per_acq):
        for k in range(kline_per_frame):
            for s in range(ns):
                petable_index = k + f*kline_per_frame
                time_point = b * frames_per_acq + f
                kline = petable[petable_index]
                kspace[time_point + 1, s, kline, :] = scan_data[b,
                               petable_index*ns + s, :]

prescan_data = prescan_data.squeeze().reshape([prescan_par['nv'], ns,
                    prescan_par['np']/2]).transpose(1,0,2)
postscan_data = postscan_data.squeeze().reshape([postscan_par['nv'], ns,
                    postscan_par['np']/2]).transpose(1,0,2)
kspace[0] = prescan_data
kspace[-1] = postscan_data

modified_kspace = np.empty(kspace.shape, dtype=np.complex)
# Maybe post/pre-scans are not necessary
# Compare with zero-padding


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
    interp_kind = ['nearest', 'linear', 'cubic'][1]
    position = -np.isnan(abs(time_serie))
        
    k0 = time_serie[position]
    t0 = np.arange(time_serie.shape[0])[position]
    mk = np.mean(k0)

    # If acquisition does not include a pre/post scan
    # if -position[0]:
        # k0 = np.append(mk, k0)
        # t0 = np.append(0, t0)
    # if -position[-1]:
        # k0 = np.append(k0, mk)
        # t0 = np.append(t0, time_frames - 1)
    
    t1 = np.arange(time_serie.shape[0])
    fr = interp1d(t0, np.real(k0),kind=interp_kind, fill_value=np.mean(np.real(k0)))
    fi = interp1d(t0, np.imag(k0),kind=interp_kind, fill_value=np.mean(np.imag(k0)))
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
print '2D Fourier Transform'
print '#########################################'
print ''

image_tmp = np.abs((np.fft.fft2(modified_kspace)))
image_tmp = np.fft.fftshift(image_tmp, axes=(2,3))
image_tmp = image_tmp.transpose([3,2,1,0])
image_tmp = image_tmp[...,1:-1]

# Image x(readout), y(phase), z(slice), t(time)

print ''
print '#########################################'
print 'Reorder slices'
print '#########################################'
print ''

image = np.empty(image_tmp.shape)
interleave_order = range(ns)
interleave_order = interleave_order[::2] + interleave_order[1::2]
for z in range(ns):
    image[:,:,interleave_order[z], :] = image_tmp[:,:, z, :]




print ''
print '#########################################'
print 'Saving image in "keyhole.nii.gz"'
print '#########################################'
print ''


affine = np.eye(4)
dx, dy, dz = 0.250, 0.250, 1.000
affine[np.eye(4) == 1] = [dx, dy, dz, 1]
nifti = nib.Nifti1Image(image, affine)
nib.save(nifti,'keyhole.nii.gz')


#
#
# scan_data = scan_data.reshape([scan_data.shape[0]*frames_per_acq, kline_per_frame, ns, scan_data.shape[2]])
# scan_data = scan_data.transpose(0,2,1,3)
# petable = petable.repeat(scan_data.shape[0]).reshape(frames_per_acq, kline_per_frame, scan_data.shape[0]).transpose(2,0,1)
# petable = petable.reshape([frames_per_acq, kline_per_frame])
#
#
# coordinate = np.empty([data.shape[0]*frames_per_acq, ns, kline_per_frame, data.shape[2]])
#
# for i in range(data.shape[1]):
    # coordinate[:, i%ns, i, :] = petable[i/15]
#
# kspace_zeropad = np.fft.fftshift(kspace_zeropad, 2)


# imshow(log(abs(np.fft.fftshift(kspace_zeropad[0,6,...], axis=0))))
# imshow(log(abs(data[0,:600,:])))
# imshow(log(abs(kspace_zeropad[0,6,...])))


