"""
poisson_reconstruct.py
Fast Poisson Reconstruction in Python

Copyright (c) 2014 Jack Doerner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import copy
import math

import numpy
import scipy
import scipy.fftpack


def poisson_reconstruct(grady, gradx, boundarysrc):
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # Laplacian
    gyy = grady[1:, :-1] - grady[:-1, :-1]
    gxx = gradx[:-1, 1:] - gradx[:-1, :-1]
    f = numpy.zeros(boundarysrc.shape)
    f[:-1, 1:] += gxx
    f[1:, :-1] += gyy

    # Boundary image
    boundary = copy.deepcopy(boundarysrc)  # .copy()
    boundary[1:-1, 1:-1] = 0

    # Subtract boundary contribution
    f_bp = (
        -4 * boundary[1:-1, 1:-1]
        + boundary[1:-1, 2:]
        + boundary[1:-1, 0:-2]
        + boundary[2:, 1:-1]
        + boundary[0:-2, 1:-1]
    )
    f = f[1:-1, 1:-1] - f_bp

    # Discrete Sine Transform
    tt = scipy.fftpack.dst(f, norm="ortho")
    fsin = scipy.fftpack.dst(tt.T, norm="ortho").T

    # Eigenvalues
    (x, y) = numpy.meshgrid(
        range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True
    )
    denom = (2 * numpy.cos(math.pi * x / (f.shape[1] + 2)) - 2) + (
        2 * numpy.cos(math.pi * y / (f.shape[0] + 2)) - 2
    )

    f = fsin / denom

    # Inverse Discrete Sine Transform
    tt = scipy.fftpack.idst(f, norm="ortho")
    img_tt = scipy.fftpack.idst(tt.T, norm="ortho").T

    # New center + old boundary
    result = copy.deepcopy(boundary)
    result[1:-1, 1:-1] = img_tt

    return result
