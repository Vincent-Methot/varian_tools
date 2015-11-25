#!/usr/bin/env python2
# -*- coding: utf-8 -*-

########################################################################
# Author: Vincent Méthot                                               #
# Creation date: 2014/10/31                                            #
# Tools to work with data coming from a Varian small animal MRI        #
########################################################################

"""
Library of tools used to work with data coming from the Varian small
animal MRI scanner. Functions included are as follow:

    load_fid: Import a 'fid' file into a numpy array.
    load_procpar: Import a procpar file into a dictionary.
    print_procpar: Print important informations of a procpar.
    reconstruct_fsems: Reconstruct a fsems image given a fid and a procpar.
    save_nifti: Save a nifti file fron numpy array using procpar informations.
    reconstruct_keyhole: Reconstruct a keyhole image using fid and propar.
    reorder_interleave: Reorder interleaved slices in an image.
    fourier_transform: Correct for DC offset and run the fourier transform on kspace.
    read_petable: Return a numpy array from a PETable.
    explore_image: Shows diverse slices for quality control.
"""
import numpy as np
import nibabel as nib
from struct import unpack
from nmrglue.fileio.varian import read_procpar
import datetime
import sys
import os
from pylab import *

def load_fid(path_to_fid):
    """
    Take a fid file coming from a Varian NMR imager. Returns a 3D numpy 
    array (1: block number, 2: trace number, 3: element number). All
    necessary informations for reconstruction are inside the 'procpar'
    file (usually in the same directory as the 'fid' file.
    
    Parameter
    ---------
    path_to_fid: string path to the fid file
    
    Return
    ------
    data: 3D numpy array countaing the fid data
    """
    
    bytesInHeader = 32
    bytesPerBlockHeader = 28
    
    # Opening fid file
    fid = open(path_to_fid, 'rb')
    packedData = fid.read()
    # There is header at the beginning of the fid file with basic
    # information about data structure
    header = unpack('>6l2hl', packedData[:bytesInHeader])
    # Some of them are useless
    nblocks = header[0]     # number of blocks in file
    ntraces = header[1]     # number of traces per block
    nelements = header[2]   # number of elements per trace
    ebytes = header[3]      # number of bytes per element
    tbytes = header[4]      # number of bytes per trace
    bbytes = header[5]      # number of bytes per block
    vers_id = header[6]     # software version, file_id status bits
    status = header[7]      # status of whole file
    nbheaders = header[8]   # number of block header per block

    print 'Number of blocks =', nblocks
    print 'Number of traces per block =', ntraces
    print 'Number of elements per trace =', nelements/2

    if ebytes == 4:
        dataType = 'i'
    elif ebytes == 2:
        dataType = 'h'

    tmpData = np.empty([nblocks, ntraces * nelements], dtype=int)
    data = np.empty([nblocks,ntraces,nelements/2], dtype=complex)
    blockFormat = '>' + str((bbytes-bytesPerBlockHeader)/4) + dataType
    blockStart = np.array(bytesInHeader + np.arange(nblocks) * bbytes) + bytesPerBlockHeader
    blockEnd = blockStart + bbytes - bytesPerBlockHeader
    # Remove header and block headers (bottleneck)
    for block in range(nblocks):
        tmpData[block,...] = np.array(unpack(blockFormat, packedData[blockStart[block]:blockEnd[block]]))
    data = (tmpData[:,::2] + tmpData[:,1::2]*1j).reshape(nblocks, ntraces, nelements/2)

    print 'Shape of the data:', data.shape
    return data.astype('complex64')

def load_procpar(path_to_procpar):
    """
    Read a procpar file and return a dictionary
    
    Parameter
    ---------
    path_to_procpar: string path to the procpar file
    
    Result
    ------
    parameters: dictionary containing the procpar parameters
    """
    
    def format(s):
        """
        Transform string of lists of string into proper format
        (either int, float or string, remove unnecessary lists)
        """
        try:
            s = [int(k) for k in s]
        except ValueError:
            try:
                s = [float(k) for k in s]
            except ValueError:
                s = s
        if len(s) == 0:
            s = 'NA'
        if len(s) == 1:
            s = s[0]
        return s
    
    procpar = read_procpar(path_to_procpar)
    parameters = dict([(key, format(procpar.get(key, dict()).get('values', 'NA')))
                       for key in procpar.keys()])
    
    return parameters

def load_fdf(path_to_img):
    """
    Parse every file from a '.img' folder and transform them into a numpy array
    
    Parameter
    ---------
    path_to_img: string indicating a folder with fdf files and a procpar file
    
    Return
    ------
    image: numpy array containing an image
    """
    
    # TODO
    # Yet only works for single 2d acquisition
    
    fdflist = os.listdir(path_to_img)
    try:
        par = load_procpar(path_to_img + '/' + fdflist.pop(fdflist.index('procpar')))
    except ValueError:
        print 'No procpar found'
        
    fdflist = [n for n in fdflist if n.endswith('.fdf')]
    fdflist = [n for n in fdflist if not(n.startswith('.'))]
    if type(par['tr']) == list:
    	image = np.empty([par['ns'], par['nv'], par['np']/2, len(par['tr'])])
    elif type(par['te']) == list:
        image = np.empty([par['ns'], par['nv'], par['np']/2, len(par['te'])])
    else:
    	image = np.empty([par['ns'], par['nv'], par['np']/2, 1])

    for i in fdflist:
        # Open fdf files one at a time
        f = open(path_to_img + '/' + i, 'rb')
        # Read fdf header
        header = []
        for ligne in range(300):
            header.append(f.readline()[:-1])
            if header[-1] == '\x0c':
                break
        
        # Get infos from header
        headerSlice_no = [s for s in header if 'slice_no =' in s][0]
        sli = int(headerSlice_no[headerSlice_no.find('=') + 2: -1])
        headerDtype = [s for s in header if 'storage =' in s][0]
        dtype = headerDtype[ headerDtype.find('=')+3 : -2]
        headerBits = [s for s in header if 'bits =' in s][0]
        bits = int( headerBits[ headerBits.find('=')+2 : -1] )
        pix_per_slice = par['nv'] * par['np'] / 2
        dataSize = pix_per_slice * bits / 8
        precision = dtype + str(bits)
        array_index = [s for s in header if 'array_index =' in s][0]
        array_value = int(array_index[array_index.find('=') + 2: -1])

        # Read data from file
        f.read()
        f.seek(f.tell() - dataSize)
        debut = f.tell()
        packedData = f.read()
        data = np.array(unpack('>' + str(pix_per_slice) + precision[0], packedData))
        image[sli-1,:,:,array_value-1] = np.array(data).reshape([par['nv'], par['np']/2])
        f.close()

    return image

def reconstruct_fsems(fid, par):
    """
    Use procpar informations to reorder kspace data contained inside the fid

    Parameters
    ----------  
    fid : numpy array containing data from the fid file (use load_fid)
    par : dictionary containing parameters from procpar file
    """
    # TODO
    # Account for interleaving
    # Is there a phase-encore table (e.g. keyhole, fsems)
    # Is it a 3d acquisition (e.g. ge3d)
    
    fid = fid.squeeze()
    fid = fid.reshape(par['etl'], par['ns'], par['nv']/par['etl'], par['np']/2)
    fid = fid.transpose(1,0,2,3)
    fid = fid.reshape(par['ns'], par['nv'], par['np']/2)
    petable = read_petable(par['petable'])
    petable += abs(petable.min())

    kspace = np.empty(fid.shape, dtype=complex)
    for i in range(par['nv']):
        kspace[:, petable[i], :] = fid[:, i, :]
        
    return kspace
    
def reconstruct_gems(fid, par):
    """
    Use procpar informations to reorder kspace data contained inside the fid

    Parameters
    ----------  
    fid : numpy array containing data from the fid file (use load_fid)
    par : dictionary containing parameters from procpar file
    """
    # TODO
    # Account for interleaving
    # Is there a phase-encore table (e.g. keyhole, fsems)
    # Is it a 3d acquisition (e.g. ge3d)
    
    fid = fid.squeeze()
    fid = fid.reshape(par['arraydim'], par['nv'], par['ns'], par['np']/2)
    fid = fid.transpose(1, 3, 2, 0)
  
    return image

def print_procpar(par):
    """
    Returns a string with a formated version of the par file (use load_procpar)
    in the way of a lab-book.
    """
    
    # Values that are going to be printed. Order is print order. Change
    # to change printed informations.
    short_report = ['te', 'tr', 'arraydim', 'fliplist', 'layout',
    'axis', 'arraydim', 'esp', 'etl', 'thk', 'ns', 'nt', 'nv', 'np',
    'acqdim', 'fliplist', 'gain', 'orient', 'petable', 'lpe', 'lro']

    # long_report = ['layout', 'rfcoil', 'operator_', 'date',
    # 'orient', 'axis', 'gain', 'arraydim', 'acqcycles', 'tr', 'te', 'ti', 
    # 'esp', 'etl', 'thk', 'ns', 'pss', 'nt', 'nv', 'np', 'lpe', 'lro', 'dimX', 
    # 'dimY', 'dimZ', 'filter', 'acqdim', 'fliplist', 'gcrush', 'gf', 'gf1',
    # 'gpe' , 'mintr', 'minte', 'petable', 'pslabel', 'posX', 'posY', 'posZ',
    # 'pss0', 'studyid', 'tpe', 'trise','tn', 'B0', 'resto']

    # Other possible values: 'ap', 'math', 'echo', 'at', 'fn',
    # 'np', 'nv', 'fn1', 'fn', 'time_run', 'time_complete'
    time_run = par['time_run']
    time_complete = par['time_complete']
    start_time = datetime.datetime(int(time_run[:4]),
                                   int(time_run[4:6]),
                                   int(time_run[6:8]),
                                   int(time_run[9:11]),
                                   int(time_run[11:13]),
                                   int(time_run[13:15]))
    finish_time = datetime.datetime(int(time_complete[:4]),
                                    int(time_complete[4:6]),
                                    int(time_complete[6:8]),
                                    int(time_complete[9:11]),
                                    int(time_complete[11:13]),
                                    int(time_complete[13:15]))
    delta_time = finish_time - start_time

    if delta_time.days < 0:
        seconds = np.sum(par['tr']) * par['nv']
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        delta_time = datetime.time(int(h), int(m), int(s))

    if par['orient']=='cc' or par['orient']=='cor' or par['orient']=='cor90':
        orient = 'coronal   '
    elif par['orient']=='aa':
        orient = 'axial     '
    elif par['orient']=='trans':
        orient = 'transverse'
    else:
        orient = par['orient']

    if type(par['tr'])==list:
        tr = np.array(list(set(par['tr'])))
        if len(tr)==1:
            tr = tr[0]
    else:
        tr = np.array(par['tr'])

    if type(par['te'])==list:
        te = np.array(list(set(par['te'])))
        if len(te)==1:
            te = te[0]
    else:
        te = np.array(par['te'])

    line1 = """Paramètres d'acquisition de """ + str(par['seqfil']) + ' --- ' + \
        str(par['operator_']) + ' ' + str(par['date'])
    line2 = '******************************************************'
    line3 = str(par['layout']) + '       gain = ' + str(par['gain']) + '       ' + \
            orient + '       ' + str(par['np']/2) + ' x ' + str(par['nv'])
    line4 = 'tr = ' + str(1e3 * tr) + ' [ms]   te = ' +  \
            str(1e3 * te) + ' [ms]    ' + \
            str(round(10*par['lro'], 2))  + ' x ' + \
            str(round(10*par['lpe'], 2)) + ' [mm x mm]'
    line5 = 'flip = ' + str(par['fliplist'][0]) + '     slices = ' + \
            str(par['ns']) + '    thickness = ' + str(round(par['thk'], 2)) + ' [mm]'
    line6 = 'repetitions = ' + str(par['arraydim']) + '           averages = ' + \
            str(par['nt'])
    line7 = 'petable = ' + str(par['petable']) + '       total time = ' + \
            str(delta_time)

    report = '\n' + line1 + '\n' + line2 + '\n' + line3 + '\n' + line4 + '\n' + line5 + \
        '\n' + line6 + '\n' + line7 + '\n'
    return report


def save_nifti(name, data, par):
    """
    Make a nifti header for Varian image coming from the scanner
    
    Parameters
    ----------
    name: string containing the name of the file
    data: numpy array with the image elements
    par: dictionary containing the procpar parameters
    """
    
    print 'Dimensions du Nifti:', data.shape
 
    affine = np.eye(4)
    dx = par['lpe'] * 10. / data.shape[0]
    dy = par['lro'] * 10. / data.shape[1]
    dz = float(par['thk'])
    affine[np.eye(4) == 1] = [dx, dy, dz, 1]
    nifti = nib.Nifti1Image(data.astype('f32'), affine)
    nib.save(nifti, name)

def reconstruct_keyhole(data, par):
    """
    Reconstruct a 4D image from a keyhole acquisition.
    """

    # Get phase encode table from the scan parameters
    petable = read_petable(par['petable'])
    # Count number of zero in petable to know number of frame per acquisition
    frames_per_acq = np.sum(petable == 0)
    time_frames = frames_per_acq * data.shape[0]
    kline_per_frame = par['nv'] / frames_per_acq
    ns = par['ns']
    
    print 'Number of slices:', ns
    print 'Frames per acquisition:', frames_per_acq
    print 'K-lines per frame:', kline_per_frame

    # Prepare kspace array
    petable -= petable.min()
    kspace = np.empty([time_frames, ns, petable.max() + 1,
                       data.shape[-1]], dtype='complex64')
    kspace[...] = np.nan

    # Reorganizing kspace using petable.
    for b in range(par['arraydim']):
        for f in range(frames_per_acq):
            for k in range(kline_per_frame):
                for s in range(ns):
                    petable_index = k + f*kline_per_frame
                    time_point = b * frames_per_acq + f
                    kline = petable[petable_index]
                    kspace[time_point, s, kline, :] = data[b,
                                   petable_index*ns + s, :]
    kspace = kspace.transpose([2,3,1,0])
    # New data order is (phase-encode, freq encode, slice, time)

    def interp_keyhole(time_serie):
        """
        Takes a time serie (numpy array) with nans where data has not been
        acquired.
        Array dim: (time point, slice number, phase encode, freq. encode)
        """
        # Position points to non-nan elements (which has been acquired)
        position = -np.isnan(abs(time_serie))
        # For center of k-space, no keyhole interpolation required
        if -np.any(np.isnan(time_serie)):
            return time_serie

        # Interpolation is linear, with numpy function 'interp'
        time_frames = time_serie.shape[0]
        k0 = time_serie[position]
        t0 = np.arange(time_frames)[position]
        t1 = np.arange(time_serie.shape[0])
        fr = np.interp(t1, t0, np.real(k0), np.real(k0[0]), np.real(k0[-1]))
        fi = np.interp(t1, t0, np.imag(k0), np.imag(k0[0]), np.imag(k0[-1]))
        k1 = fr + fi*1j
        return k1

    kspace = np.apply_along_axis(interp_keyhole, -1, kspace)
    return kspace

def reorder_interleave(image):
    """
    Reorder slices in an interleaved 3d or 4d acquisition
    """
    image_reorder = np.empty_like(image)
    interleave_order = range(image.shape[2])
    interleave_order = interleave_order[::2] + interleave_order[1::2]
    for z in range(image.shape[2]):
        image_reorder[:,:,interleave_order[z], ...] = image[:,::-1, z, ...]
    return image_reorder

def fourier_transform(kspace):
    """
    Correct for DC offset and execute the Fourier transform a
    cartesian raster
    """
    
    kspace -= kspace.mean()
    image = np.abs((np.fft.fft2(kspace, axes=(0,1))))
    image = np.fft.fftshift(image, axes=(0, 1))
    phase = np.angle((np.fft.fft2(kspace, axes=(0,1))))
    phase = np.fft.fftshift(phase, axes=(0, 1))
    return image, phase

def read_petable(petable):
    f = open(petable)
    # Discard first line
    f.readline()
    petable = [l.split() for l in f.readlines()]
    f.close()
    petable = np.array(petable, dtype=int).ravel()
    return petable

def explore_image(image, time_point=0):
    """
    Show different slices / time selection of a 2-3-4d dataset
    """
    dim = image.shape

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
        # print "Number of frames to layout:", n
        # print "Layout size:", m, '=', a, '*', b
        return a, b

    if len(dim) == 2:
        "This is a 2D image"
        imshow(image, 'gray')
    if len(dim) == 3:
        "This is a 3D image"
        a, b = layout(dim[2])
        for i in range(dim[2]):
            subplot(a, b, i+1)
            imshow(image[..., i], 'gray')
            title('Slice ' + str(i+1))
    if len(dim) == 4:
        "This is a 3D + time image"
        # if (time_point >= dim[3]):
        #     print "Time point is too large, taking first frame"
        #     time_point = 0
        a, b = layout(dim[2])
        for i in range(dim[2]):
            subplot(a, b, i+1)
            imshow(image[..., i, time_point], 'gray')
            title('Slice ' + str(i+1))

def apply_to_nifti(input, output, function):
    """
    Opens a nifti file [input], then apply function point by point and
    saves it as the nifti file [output].
    """

    
