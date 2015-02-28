#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys



fichier = open(sys.argv[1])
motion = fichier.read()
motion = motion.split('\n')
motion = motion[:-1]
motion = [m.split() for m in motion]
motion = np.array(motion, 'float32')


plt.subplot(2,1,1)
plt.title('Translation [mm]')
for i in range(3):
	plt.plot(motion[:, 0], motion[:, i+1])

plt.subplot(2,1,2)
plt.title('Rotation [rad]')
for i in range(3):
	plt.plot(motion[:, 0], motion[:, i+4])

plt.show()