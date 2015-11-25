#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#########################################################################
# Author: Vincent Méthot                                                #
# Creation date: 2014/12/12                                             #
# Make petable for random gaussian k-t blast                            #
#########################################################################

import numpy as np
from pylab import *

res = 256
points = 6000

k = np.random.randn(1e5)
k = k[(k>=-3) & (k<=3)]
k *= res / 6
kt = k[:points].astype(int)

# complete = range(-res/2, res/2)
# half = range(-res/4, res/4)
# quarter = range(-res/8, res/8)
# eight = range(-res/16, res/16)

# sequence = complete + eight + quarter + eight + half + eight +quarter + eight
# kt = 30 * sequence

# Plotting k-t graph
plot(kt, '.')
axis('tight')
ylabel('kline')
xlabel('time')
show()

# perange = range(-resolution/2, resolution/2)
# center = resolution / 4
# border = 3 * resolution / (4 * time_frames)

# pepoints = []
# for t in range(time_frames):
#     start = -resolution / 2 + t * border / 2
#     end = start + border / 2
#     if not(half_kspace):
#         pepoints += range(start, end)
#         pepoints += range(-resolution / 8, -resolution / (8*4))
#     pepoints += range(-resolution / (8*4), resolution / 8)
#     pepoints += range(-end, -start)

# print 'Nombre de points:', len(pepoints)

# string = str(pepoints)[1:-1]

# if half_kspace:
#     f = open('gems_half_' + str(acquisition_per_frame) + '_' + str(time_frames) + '_' + str(resolution), 'w')
# else:
#     f = open('gems_' + str(acquisition_per_frame) + '_' + str(time_frames) + '_' + str(resolution), 'w')
# f.write('t1 =\n')
# f.write(string.replace(',', ''))
# f.close()
