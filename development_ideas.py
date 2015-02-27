#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""Given any kind of sampling of the k-space, time domain, interpolation
between these different points would be possible. Furthermore, with the choice
of a good basis (wavelet, compressed sensing), it would be theoretically
possible to acquire data wisely (probably randomly) to shorter acquisition
without reducing image quality, contrast, etc.

Other parameters that can be sampled during acquisition would be TE (for long
echo plannar/spiral/random path acquisition) that vary during a single readout.
Maybe TR, flip angle, inversion times... (don't know exactly how). Diffusion,
flow encoding. The main goal would be to make a sequence that acquire in real
time the exact information at the exact position chosen by the operator. During
a single exam, information would be cumulated for single patients, confronted
with other exams in time and with a database of other patients/image/models.

Denoising, comparing, image transformations and analysis would be done on the
fly and in parallel. Informations of hearth beat, respiration, motion (detected
with an optical camera) would be used to correct for artefact (and navigation
echo).

To achieve this goal, many different time scale should be used:
	-Ultra-short: of the order of millisecond to account for matter properties
	-Short: of the order of seconds, for example functional signal
	-Long: comparison between many days (the brain should be the same)
	-Ultra long: scans months or years apart, the brain may have a different
	 structure
"""

"""writing a script to search a folder/database that take informations for the
 procpar files and show only those with a given parameter."""