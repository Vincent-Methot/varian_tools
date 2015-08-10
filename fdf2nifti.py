#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Script to convert fdf into nifti

import sys
import variantools as vt
import numpy as np

inpath = sys.argv[1]
outpath = sys.argv[2]

data = vt.load_fdf(inpath)
par = vt.load_procpar(inpath + '/procpar')



# gems
if par['layout'] == 'gems':
	data = data.transpose(2,1,0,3)
	data = data[::-1,::-1,:,:]
# fsems
if par['layout'] == 'fsems':
	data = data.transpose(2,1,0,3)
	data = data[::-1,::-1,:,:]

## With weird aliasing, uncomment
# test = np.empty_like(data)
# for i in range(4):
# 	test[i::4, ...] = data[i*64:(i+1)*64, ...]


report = vt.print_procpar(par)
print report


# Printing information on a .info file
text_file = open(outpath + '.info', "w")
text_file.write(report)
text_file.close()


vt.save_nifti(outpath + '.nii.gz', data, par)


