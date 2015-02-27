#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to print interesting procpar informations

import sys
import variantools as vt

inpath = sys.argv[1]

par = vt.load_procpar(inpath)
vt.print_procpar(par)
