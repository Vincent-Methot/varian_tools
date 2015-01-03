#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to reconstruct keyhole data from a fid + procpar files

import sys
import variantools as vt

inpath = sys.argv[1]
par = vt.load_procpar(inpath)
vt.print_procpar(par)
