#!/usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################
# Author: Vincent Méthot                                               #
# Creation date: 2014/10/31                                            #
# Tools to work with data coming from a Varian small animal MRI        #
########################################################################

"""
Library of tools used to work with data coming from the Varian small
animal MRI scanner. Functions included are as follow:

    loadfid: Import a 'fid' file into a numpy array
    parseprocpar: Import a procpar file into a dictionary
    printprocpar: Print important informations of a procpar
"""

import numpy as np
import nibabel as nib
from struct import unpack
from nmrglue.fileio.varian import read_procpar
import datetime
import sys
import os


def loadfid(path_to_fid):
    """
    Take a fid file coming from a Varian NMR imager. Returns a 3D numpy 
    array (1: block number, 2: trace number, 3: element number). All
    necessary informations for reconstruction are inside de 'procpar'
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
    fid = open(path_to_fid,'rb')
    packedData = fid.read()
    # There is header at the beginning of the fid file with basic
    # information about data structure
    header = unpack('>6l2hl',packedData[:bytesInHeader])
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

    # Get block headers (probably useless...) and data
    blockHeader = np.empty([nblocks,9])
    tmpData = np.empty([nblocks,ntraces,nelements], dtype=int)
    data = np.empty([nblocks,ntraces,nelements/2], dtype=complex)

    # It seems data cannot be saved in complex integer format to save
    # space. Maybe saving 2 arrays would be fine...

    

    #Should be faster, but how?
    
    for block in range(nblocks):
        debutBlock = bytesInHeader + block * bbytes
        finBlockHeader = debutBlock + bytesPerBlockHeader
        blockHeader[block,:] = np.array(unpack('>4hl4f',packedData[debutBlock:finBlockHeader]))

        for trace in range(ntraces):
            debutTrace = finBlockHeader + trace * tbytes 
            finTrace = debutTrace + tbytes
            tmpData[block,trace,:] = np.array(unpack('>' + str(nelements) + dataType, packedData[debutTrace:finTrace])).astype(int)

            for p in range(0,nelements,2):
                data[block,trace,p/2] = np.complex(tmpData[block,trace,p], tmpData[block,trace,p+1])


    print 'Shape of the data:', data.shape
    return data.astype('complex64')


def parseprocpar(path_to_procpar):
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


def construct_fsems(fid, par):
    """
    Use procpar informations to reorder kspace data contained inside the fid

    Parameters
    ----------  
    fid : numpy array containing data from the fid file (use loadfid)
    par : dictionary containing parameters from procpar file
          (use parseprocpar)
    """
    # TODO
    # Account for interleaving
    # Is there a phase-encore table (e.g. keyhole, fsems)
    # Is it a 3d acquisition (e.g. ge3d)
    
    # for fsems
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


def printprocpar(parameters):
    """
    Print important parameters in the procpar in the way of a lab-book.
    """
    
    # Values that are going to be printed. Order is print order. Change
    # to change printed informations.
    valeurs_importantes = ['layout', 'rfcoil', 'operator_', 'date',
    'orient', 'axis', 'gain', 'arraydim', 'acqcycles', 'tr', 'te', 'ti', 
    'esp', 'etl', 'thk', 'ns', 'pss',  'nv', 'np', 'lpe', 'lro', 'dimX', 'dimY',
    'dimZ', 'filter', 'acqdim', 'fliplist', 'gcrush', 'gf', 'gf1', 'gpe'
    , 'mintr', 'minte', 'petable', 'pslabel', 'posX', 'posY', 'posZ',
    'pss0', 'studyid', 'tpe', 'trise','tn', 'B0', 'resto']
    # Other possible values: 'ap', 'math', 'echo', 'at', 'fn',
    # 'np', 'nv', 'fn1', 'fn', 'time_run', 'time_complete'
    time_run = parameters['time_run']
    time_complete = parameters['time_complete']
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

    print
    print """Paramètres d'acquisition de""", parameters['seqfil']
    print '**************************************************'
    for valeur in valeurs_importantes:
        if parameters.has_key(valeur):
            print valeur, ':', parameters[valeur]
        else:
            print valeur, ': NA'
    print 'total time:', str(delta_time)


def loadfdf(path_to_img):
    """
    Parse every file into a '.img' folder and make it into a numpy array
    
    Parameter
    ---------
    path_to_img: string indicating a folder with fdf files
    
    Return
    ------
    image: numpy array containing an image

    TODO
    ----
    Yet only works for single 2d acquisition
    """
    
    fdflist = os.listdir(path_to_img)
    try:
        par = parseprocpar(path_to_img + '/' + fdflist.pop(fdflist.index('procpar')))
    except ValueError:
        print 'No procpar found'
        return 0
        
    fdflist = [n for n in fdflist if n.endswith('.fdf')]
    image = np.empty([par['ns'], par['nv'], par['np']/2])

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
       
        # Read data from file
        f.read()
        f.seek(f.tell() - dataSize)
        debut = f.tell()
        packedData = f.read()
        data = np.array(unpack('>' + str(pix_per_slice) + precision[0], packedData))
        image[sli-1] = np.array(data).reshape([par['nv'], par['np']/2])
        f.close()

    return image


def niftiheader(name, data, par):
    """
    Make a nifti header for Varian image coming from the scanner
    """
    
    print 'Dimensions:', data.shape

    
    affine = np.eye(4)
    dx = par['lpe'] * 10 / par['nv']
    dy = par['lro'] * 20 / par['np']
    dz = par['thk']
    affine[np.eye(4) == 1] = [dx, dy, dz, 1]
    nifti = nib.Nifti1Image(data, affine)
    nib.save(nifti, name)


def keyhole(data, par):
    """
    Reconstruct a 4D image from akeyhole acquisition
    """


def read_petable(petable):
    f = open(petable)
    # Discard first line
    f.readline()
    petable = [l.split() for l in f.readlines()]
    f.close()
    petable = np.array(petable, dtype=int).ravel()
    return petable




