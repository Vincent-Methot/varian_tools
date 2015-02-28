#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Script to reconstruct keyhole data from a fid + procpar files

import sys
import variantools as vt

inpath = sys.argv[1]
outpath = sys.argv[2]
fid = vt.load_fid(inpath + '/fid')
par = vt.load_procpar(inpath + '/procpar')
kspace = vt.reconstruct_keyhole(fid, par)
image = vt.fourier_transform(kspace)

image = vt.reorder_interleave(image)
# image = image[:, ::-1, :, :]

vt.save_nifti(outpath, image, par)
