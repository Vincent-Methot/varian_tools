#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from pylab import *
import sys

def mm2inch(value):
    return value/25.4

vitals = loadtxt(sys.argv[1], delimiter=',')
ideal = np.loadtxt(sys.argv[2])
resp = vitals[:,1]
# plot(resp)
min_resp = resp.min()
max_resp = resp.max()

duration = 5 * 60
stim_ON = 60 * arange(5, 60, 10)
stim_OFF = stim_ON + duration

figure(figsize=(mm2inch(150), mm2inch(100)))

# plot((debut, i), (position, position), color='red', linewidth=2)

# title('Respiration rate')
xlabel('temps [sec]')
ylabel(u'Fréquence respiratoire [1/min]')

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
# plot(t,resp,'.')
# plot(tnew,f(tnew),'-')
plot(smooth_resp, 'k')

time_scale = len(smooth_resp) / 140.
position_bars = smooth_resp.mean() - 2 * smooth_resp.std()

stim = ideal[1:141] - ideal[:140]
for i in range(len(stim)):
	if (stim[i] >= 0.5):
		debut = i
	if (stim[i] <= -0.5) or i == (len(stim) - 1) :
		plot((time_scale*debut, time_scale*i), (position_bars, position_bars), color='red', linewidth=2)

axis('tight')
# legend(['data', 'linear', 'cubic'], loc='best')
show()

