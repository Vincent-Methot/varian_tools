#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# 1 1 1
# 1 2 2
# 1 3 3
# 2 2 4
# 2 3 6
# 3 3 9
# 3 4 12
# 3 5 15
# 4 4 16
# 3 6 18
# 4 5 20
# 4 6 24
# 5 5 25
# 5 6 30
# 6 6 36
# 6 7 42
# 7 7 49

import numpy as np

def layout(n):
	a = b = 0
	m = n - 1
	while not(a):
		m += 1
		t = int(np.sqrt(m))
		# test if perfect square
		if not(m%t):
			a, b = t, m/t
		# test if square - 1 is divisor and m is bigger than 25
		elif m >= 25:
			if not(m%(t - 1)):
				a, b = t - 1, m/(t - 1)
		# test if square - 2 is divisor and m is bigger than 100
		elif m >= 100:
			if not(m%(t - 2)):
				a, b = t - 2, m/(t - 2)

	print "Number of frames to layout:", n
	print "Layout size:", m, '=', a, '*', b

	return a, b


