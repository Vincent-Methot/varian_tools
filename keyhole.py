#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to reconstruct keyhole data from a fid + procpar files

import variantools as vt
path = '/home/vincent/Maitrise/Data/metv_20141124_001/data/gems_ep04_01.fid'
fid = vt.load_fid(path + '/fid')
par = vt.load_procpar(path + '/procpar')
kspace = vt.reconstruct_keyhole(fid, par)
image = vt.fourier_transform(kspace)
image = vt.reorder_interleave(image)
vt.save_nifti('test.nii.gz', image, par)
