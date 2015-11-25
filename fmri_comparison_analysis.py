#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import nibabel as nib
from scipy import stats

nib.load('fmri_detrend.nii.gz')
ideal = loadtxt('ideal.mat')
fmri = nib.load('fmri_volreg.nii.gz')
data = fmri.get_data()
hdr = fmri.get_header()

ON_data = data[..., ideal>0.5]
OFF_data = data[..., ideal<0.5]

mean_ON = ON_data.mean(-1)
mean_OFF = OFF_data.mean(-1)
diff = mean_ON - mean_OFF

mean_data = data.mean(-1)
masked_data = zeros(mean_data.shape)
masked_data[mean_data>0.05] = mean_data[mean_data>0.05]
imshow(masked_data.mean(-1))
mask = masked_data.astype(bool)

masked_diff = zeros(diff.shape)
masked_diff[mask] = diff[mask]

tscore, pvalue = stats.ttest_ind(ON_data, OFF_data, axis=-1)
tscore[invert(mask)] = 0
tscore[pvalue>0.15] = 0
tscore[tscore<1] = 0

slices = diff.shape[-1]

fig, ax = subplots(2, slices/2)
ax = ax.ravel()

for i in range(slices):
	ax[i].imshow(masked_diff[..., i])

masked_diff = masked_diff / masked_diff.max()
diff_image = nib.Nifti1Image(masked_diff, fmri.get_affine(), hdr)
nib.save(diff_image, 'fmri_diff.nii.gz')

tscore_map = nib.Nifti1Image(tscore, fmri.get_affine(), hdr)
nib.save(tscore_map, 'fmri_tscore.nii.gz')

def norm(voxel):
	voxel = voxel - voxel.min()
	voxel = voxel / voxel.max()
	return voxel

voxel = data[29:31, 16:18, 5, :]

snr = ON_data.mean(-1) / ON_data.std(-1)
snr_map = nib.Nifti1Image(snr, fmri.get_affine(), hdr)
nib.save(snr_map, 'fmri_snr.nii.gz')

cnr = 2 * (ON_data.mean(-1) - OFF_data.mean(-1)) / (ON_data.std() + OFF_data.std())
cnr_map = nib.Nifti1Image(cnr, fmri.get_affine(), hdr)
nib.save(cnr_map, 'fmri_cnr.nii.gz')

figure()
plot(ideal)
plot(norm(voxel))
xlabel('time [25 sec]')
ylabel('signal [au]')
legend(['Measurement', 'Ideal'])

perc_change = (data - data.mean(-1, keepdims=True)) / data.mean(-1, keepdims=True)
fmri_perc_change = nib.Nifti1Image(perc_change, fmri.get_affine(), hdr)
nib.save(fmri_perc_change, 'fmri_perc_change.nii.gz')

def denoise_voxel(ideal, voxel):
	"""Denoising?"""

# Remove the 5 first minutes of the scan
# Motion correction