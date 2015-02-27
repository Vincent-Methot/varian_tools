#!/usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################
# Author: Vincent Méthot                                               #
# Creation date: 2014/12/12                                            #
# Make petable for keyhole acquisition                                 #
########################################################################

import sys

acquisition_per_frame = int(sys.argv[1])
time_frames = int(sys.argv[2])
resolution = int(sys.argv[3])

# To acquire 60% of kspace
half_kspace = False

perange = range(-resolution/2, resolution/2)
center = resolution / 4
border = 3 * resolution / (4 * time_frames)

pepoints = []
for t in range(time_frames):
    start = -resolution / 2 + t * border / 2
    end = start + border / 2
    if not(half_kspace):
        pepoints += range(start, end)
        pepoints += range(-resolution / 8, -resolution / (8*4))
    pepoints += range(-resolution / (8*4), resolution / 8)
    pepoints += range(-end, -start)

print 'Nombre de points:', len(pepoints)

string = str(pepoints)[1:-1]

if half_kspace:
    f = open('gems_half_' + str(acquisition_per_frame) + '_' + str(time_frames) + '_' + str(resolution), 'w')
else:
    f = open('gems_' + str(acquisition_per_frame) + '_' + str(time_frames) + '_' + str(resolution), 'w')
    
f.write('t1 =\n')
f.write(string.replace(',', ''))
f.close()
