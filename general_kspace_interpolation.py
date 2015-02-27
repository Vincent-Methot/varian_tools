#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Given any kind of sampling of the k-space, time domain, interpolation
between these different points would be possible. Furthermore, with the choice
of a good basis (wavelet, compressed sensing), it would be theoretically
possible to acquire data wisely (probably randomly) to shorter acquisition
without reducing image quality, contrast, etc.

Other parameters that can be sampled during acquisition would be TE (for long
echo plannar/spiral/random path acquisition) that vary during a single readout.
