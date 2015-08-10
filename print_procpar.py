#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Script to print interesting procpar informations

import sys
import variantools as vt

inpath = sys.argv[1]

par = vt.load_procpar(inpath)
report = vt.print_procpar(par)
print report
