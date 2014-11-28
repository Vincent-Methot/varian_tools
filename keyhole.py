#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to reconstruct keyhole data from a fid + procpar files

import variantools as vt
path = '/home/vincent/Maitrise/Data/metv_20141112_002/data/gems_ep04_06.fid'
fid = vt.load_fid(path + '/fid')
par = vt.load_procpar(path + '/procpar')
kspace = vt.reconstruct_keyhole(fid, par)
image = vt.fourier_transform(kspace)
image = vt.reorder_interleave(image)
vt.save_nifti('/home/vincent/Bureau/Resting_state/metv_20141112_002.nii.gz', image, par)
