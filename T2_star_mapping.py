#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import variantools as vt
import numpy as np
import nibabel as nib

inpath = '../data/gems_06.fid'
filename = 'air_medical.nii.gz'

fid = vt.load_fid(inpath +'/fid')
par = vt.load_procpar(inpath + '/procpar')
test = fid.reshape(len(par['te']), par['np']/2, par['ns'], par['nv'])
image = abs(np.fft.fftshift(np.fft.fft2(test, axes=(1,3))))
image = image.transpose(1,3,2,0)
image = vt.reorder_interleave(image)
# image = image[::-1, ...]
vt.save_nifti(filename, image, par)

def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, -K

def cut(image, low=1, high=100, rep=0):
    image[((image < low) | (image > high))] = rep 
    return image

oxygen = nib.load('oxygen.nii.gz')
air = nib.load('air.nii.gz')
affine = oxygen.get_affine()
hdr = oxygen.get_header()

# There is motion in the oxygen dataset
data_oxygen = oxygen.get_data()[..., :8]
data_air = air.get_data()

# There is motion in the oxygen dataset
TE_air = np.arange(3, 29, 2)
TE_oxygen = np.arange(3, 19, 2)
oxygen_fit = np.empty(data_oxygen.shape[:-1])
air_fit = np.empty(data_air.shape[:-1])

for index in np.ndindex(data_oxygen.shape[:-1]):
    A, oxygen_fit[index] = fit_exp_linear(TE_oxygen, data_oxygen[index])

for index in np.ndindex(data_air.shape[:-1]):
    A, air_fit[index] = fit_exp_linear(TE_air, data_air[index])

nib.save(nib.Nifti1Image(oxygen_fit, affine, hdr), 'oxygen_fit.nii.gz')
nib.save(nib.Nifti1Image(air_fit, affine, hdr), 'air_fit.nii.gz')

nib.save(nib.Nifti1Image(diff, affine, hdr), 'diff.nii.gz')

O = 1 / oxygen_fit
D = 1 / air_fit
O = cut(O)
D = cut(D)
TE_opt = O * D * np.log(D / O) / (D - O)
# TE_opt[((TE_opt > 500) | (TE_opt < 0))] = 0
TE_opt[isnan(TE_opt)] = 0


nib.save(nib.Nifti1Image(O, affine, hdr), 'T2_air.nii.gz')
nib.save(nib.Nifti1Image(D, affine, hdr), 'T2_oxygen.nii.gz')
nib.save(nib.Nifti1Image(TE_opt, affine, hdr), 'TE_opt.nii.gz')




for i in range(2):
    for j in range(1):
        for k in range(1):
            position = (31+i, 43+j, 6+k)
            A0, R = fit_exp_linear(TE_air, data_air[position])
            # R est en 1/sec, A est l'amplitude du signal.
            T2 = R  # En ms
            plot(TE_air, data_air[position], 'r*', TE_air, A0*np.exp(-TE_air * R))


for i in range(2):
    for j in range(1):
        for k in range(1):
            position = (31+i, 43+j, 6+k)
            A0, R = fit_exp_linear(TE_oxygen, data_oxygen[position])
            # R est en 1/sec, A est l'amplitude du signal.
            T2 = R  # En ms
            plot(TE_oxygen, data_oxygen[position], 'b*', TE_oxygen, A0*np.exp(-TE_oxygen * R))

