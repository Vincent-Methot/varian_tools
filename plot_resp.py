#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from pylab import *
import sys


vitals = loadtxt(sys.argv[1], delimiter=',')
resp = vitals[:,1]
# plot(resp)
min_resp = resp.min()
max_resp = resp.max()

duration = 2 * 60
stim_ON = 60 * arange(4, 60, 10)
stim_OFF = stim_ON + duration

for i in stim_ON:
	plot((i, i), (min_resp, max_resp), 'k-')
for i in stim_OFF:
	plot((i, i), (min_resp, max_resp), 'r-')

title('Respiration rate')
xlabel('time [sec]')
ylabel('Respiration rat [1/min]')

def smooth(signal, length):
	smooth = zeros(signal.shape)
	max = len(signal)
	for i in range(max):
		start = i - length/2
		if start < 0: start = 0
		end = i + length/2
		if end > max: end = max
		smooth[i] = signal[start:end].mean()
	return smooth


from scipy.interpolate import interp1d
smooth_resp = smooth(resp, 60)
t = arange(0, len(resp))
f = interp1d(t, resp, kind='linear')
tnew = arange(0,len(resp),len(resp)/70)[:70]
plot(t,resp,'.')
# plot(tnew,f(tnew),'-')
plot(smooth_resp)
axis('tight')
# legend(['data', 'linear', 'cubic'], loc='best')
show()

