#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Takes the first n timepoints of a 4D fMRI dataset

import sys
import nibabel as nib
import numpy as np

inpath = sys.argv[1]
outpath = sys.argv[2]
ending = int(sys.argv[3])

fmri = nib.load(inpath)
data = fmri.get_data()
tpoints = data.shape[-1]

# print 'Time points:', type(tpoints)
# print 'Ending:', type(ending)
# print 'Time points >= ending', tpoints >= ending

if tpoints >= ending:
	mini = data[..., 0:ending]
	fmri_mini = nib.Nifti1Image(mini, fmri.get_affine(), fmri.get_header())
	nib.save(fmri_mini, outpath)
	print 'File save as', outpath
else:
	print 'Cannot cut datasize with', data.shape[-1], 'time points at value', ending
	print 'Leaving without doing anything'
