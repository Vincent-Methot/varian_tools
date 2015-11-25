#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import variantools as vt
import numpy as np
import nibabel as nib
import sys

inpath = '../data/gems_02.fid'
filename = 'fmri.nii.gz'

fid = vt.load_fid(inpath +'/fid')
par = vt.load_procpar(inpath + '/procpar')
fid = fid.reshape(len(par['tr']), par['nv'], par['ns'], par['np']/2)
fid = fid.transpose(1,3,2,0)
image, phase = vt.fourier_transform(fid)
image = vt.reorder_interleave(image)
image = image[::-1, ::1, ...]
vt.save_nifti(filename, image, par)
