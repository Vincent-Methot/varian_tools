#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Script to convert fdf into nifti

import sys
import variantools as vt

inpath = sys.argv[1]
outpath = sys.argv[2]

data = vt.load_fdf(inpath)
par = vt.load_procpar(inpath + '/procpar')

data = data.transpose(2,1,0,3)
vt.print_procpar(par)

vt.save_nifti(outpath, data, par)
