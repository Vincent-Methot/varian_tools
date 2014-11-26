#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to reconstruct keyhole data from a fid + procpar files

import variantools as vt
path = '/home/vincent/Maitrise/Data/metv_20141124_001/data/gems_ep04_01.fid'
fid = vt.loadfid(path + '/fid')
par = vt.parseprocpar(path + '/procpar')
kspace = vt.keyhole(fid, par)
image = vt.fourier_transform(kspace)
vt.save_nifti('test.nii.gz', image, par)
