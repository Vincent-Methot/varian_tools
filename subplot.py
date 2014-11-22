#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib.pylab import *
import numpy as np

def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))



# No FFT
clf();
for i in range(100):
    subplot(10,10,i+1)
    imshow(safe_ln(abs((modified_kspace[i,6,...]))), 'gray')





# With FFT
clf();
for i in range(24):
    subplot(4,6,i+1)
    imshow(abs(image[...,6,i]), 'gray')


