#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# rfp - radiation field propagator
# tools for manipulating genesis dfl field files
# J. Duris jduris@slac.stanford.edu

# would be nice to reorganize with fld and params passed in the same dict

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from rfp import *

from Bragg_mirror import *
def Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, showPlotQ=False, reflectQ=True, verboseQ=False, timeDomainQ=True, undo_slippageQ=False, kxspace_inQ=False, kxspace_outQ=False):

    # dt is sample time of field
    h_Plank = 4.135667696e-15;      # Plank constant [eV-sec]
    c_speed  = 299792458;           # speed of light[m/sec]
    nslice = fld.shape[0]
    hw0_eV = h_Plank * c_speed / xlamds
    Dhw_eV = h_Plank / dt; dhw_eV = Dhw_eV / (nslice - 1.)
    eph = hw0_eV + Dhw_eV / 2. * np.linspace(-1.,1.,nslice)
    eph = np.fft.ifftshift(eph)
    theta_0 = 45.0*np.pi/180.
    dx = 2. * dgrid / ncar
    Dkx = 2. * np.pi / dx
    Dtheta = Dkx * xlamds / 2. / np.pi
    theta = theta_0 + Dtheta / 2. * np.linspace(-1.,1.,ncar)
    theta = np.fft.ifftshift(theta)

    # go to reciprocal space
    if timeDomainQ: # we are in the time domain so go to frequency domain
        t0 = time.time()
        #fld = np.fft.fftshift(fft(fld, axis=0), axes=0)
        fld = fft(fld, axis=0)
        if verboseQ: print('took',time.time()-t0,'seconds for fft over t')
    if not kxspace_inQ:
        t0 = time.time()
        #fld = np.fft.fftshift(fft(fld, axis=1), axes=1)
        fld = fft(fld, axis=1)
        if verboseQ: print('took',time.time()-t0,'seconds for fft over x')

    if showPlotQ:
        t0 = time.time()
        spectrum = np.sum(np.sum(np.abs(fld)**2, axis=1), axis=1)
        if verboseQ: print('took',time.time()-t0,'seconds to calculate spectrum')
    #if showPlotQ:
        #plt.plot(eph, spectrum)
        #plt.xlabel('Photon energy (eV)'); plt.ylabel('Spectral intensity')
        #plt.tight_layout(); plt.show()

    t0 = time.time()
    if reflectQ: 
        R = Bragg_mirror_reflection(eph, theta, undo_slippageQ=undo_slippageQ).T
        ylabel = 'Bragg diffration intensity'
    else:
        R = Bragg_mirror_transmission(eph, theta).T
        ylabel = 'Forward diffration intensity'
    if verboseQ: print('took',time.time()-t0,'seconds to calculate Bragg filter')

    if showPlotQ:
        
        pi = np.pi; ncontours = 100
        Eph,Thetaurad = np.meshgrid(eph,1e6*(theta-pi/4));
        
        plt.contourf(Eph,Thetaurad,np.abs(R.T)**2,ncontours, label='filter')
        plt.contour(Eph,Thetaurad,np.sum(np.abs(fld)**2,axis=2).T,5, label='radiation')
        plt.contour(Eph,Thetaurad,np.abs(R.T)**2,3)
        plt.ylabel('Angle - 45 deg (urad)')
        plt.xlabel('Photon energy (eV)')
        plt.title(ylabel)
        plt.colorbar(); plt.legend(); plt.tight_layout(); plt.show()
        
        cut = Thetaurad == 0;
        
        plt.plot(Eph[cut],np.abs(R.T[cut])**2,label='filter')
        plt.plot(eph,spectrum/np.max(spectrum),dashes=[2, 1],label='radiation')
        plt.title(['Angle = 45 deg'])
        plt.xlabel('Photon energy (eV)')
        plt.ylabel(ylabel)
        plt.legend(); plt.tight_layout(); plt.show()

    # apply mask to field
    t0 = time.time()
    fld = np.einsum('ij,ijk->ijk',R,fld)
    if verboseQ: print('took',time.time()-t0,'seconds to apply Bragg filter')

    # return to real space
    if timeDomainQ: # we were in the time domain so return to time domain
        t0 = time.time()
        #fld = ifft(np.fft.ifftshift(fld, axes=0), axis=0)
        fld = ifft(fld, axis=0)
        if verboseQ: print('took',time.time()-t0,'seconds for ifft over t')
    if not kxspace_outQ:
        t0 = time.time()
        #fld = ifft(np.fft.ifftshift(fld, axes=1), axis=1)
        fld = ifft(fld, axis=1)
        if verboseQ: print('took',time.time()-t0,'seconds for ifft over x')
    
    return fld

def Bragg_mirror_transmit(fld, ncar, dgrid, xlamds, dt, showPlotQ=False, verboseQ=False, timeDomainQ=True, undo_slippageQ=False, kxspace_inQ=False, kxspace_outQ=False):
    return Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, showPlotQ=showPlotQ, reflectQ=False, verboseQ=verboseQ, timeDomainQ=timeDomainQ, undo_slippageQ=undo_slippageQ, kxspace_inQ=kxspace_inQ, kxspace_outQ=kxspace_outQ)

# from undulator exit, through ring cavity, to undulator entrance
def cavity_return_to_undulator(fld, settings):

    ncar = settings['ncar']
    dgrid = settings['dgrid']
    xlamds = settings['xlamds']
    zsep = settings['zsep']
    isradi = settings['isradi']
    dt = xlamds * zsep * max(1,isradi) / 299792458

    try:
        skipTimeFFTsQ = settings['skipTimeFFTsQ']
    except:
        skipTimeFFTsQ = 1
    try:
        skipSpaceFFTsQ = settings['skipSpaceFFTsQ']
    except:
        skipSpaceFFTsQ = 1
    try:
        showPlotQ = settings['showPlotQ']
    except:
        showPlotQ = 0
    try:
        savePlotQ = settings['savePlotQ']
    except:
        savePlotQ = 0
    try:
        verbosity = settings['verbosity']
    except:
        verbosity = 1

    t0total = time.time()
    

    # plot the field
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # drift from undulator to first mirror
    Ldrift = settings['l_cavity'] - settings['z_und_end']
    fld = rfp(fld, xlamds, dgrid, A=1, B=Ldrift, D=1, ncar=ncar, cutradius=0, dgridout=-1, kxspace_inQ=0, kxspace_outQ = skipSpaceFFTsQ)
    #fld = rfp_old(fld, xlamds, dgrid, A=1, B=8.706, D=1, ncar=ncar, cutradius=0, dgridout=1.001*dgrid, verboseQ=verbosity>2)
    #fld = rfp(fld, xlamds, dgrid, A=1, B=8.706, D=1, ncar=ncar, cutradius=0, dgridout=1.001*dgrid)
    # plot the field
    if verbosity: print('Field after',Ldrift,'m drift from undulator to first mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    if skipTimeFFTsQ:
        timeDomainQ = 0
        t0 = time.time()
        #fld = np.fft.fftshift(fft(fld, axis=0), axes=0)
        fld = fft(fld, axis=0)
        if verbosity: print('took',time.time()-t0,'seconds for fft over t')
        undo_slippageQ = 1
    else:
        timeDomainQ = 1
        undo_slippageQ = 1

        
    # reflect from the Bragg mirror
    fld = Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, showPlotQ=showPlotQ, verboseQ=1, timeDomainQ=timeDomainQ, undo_slippageQ=undo_slippageQ, kxspace_inQ = skipSpaceFFTsQ)
    # plot the field
    if verbosity: print('Field after 1st Bragg mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # 0.5 m drift, 61 m focal length lens, 0.5 m drift
    #R = matprod([Rdrift(0.5),Rlens(61),Rdrift(0.5)])
    Ldrift1 = settings['w_cavity']/2; flens = settings['flens1']; Ldrift2 = settings['w_cavity']/2
    R = matprod([Rdrift(Ldrift1),Rlens(flens),Rdrift(Ldrift2)])
    fld = rfp(fld, xlamds, dgrid, A=R[0,0], B=R[0,1], D=R[1,1], ncar=ncar, cutradius=0, dgridout=-1)
    # plot the field
    if verbosity: print('Field after',Ldrift1,'m drift,',flens,'m focal length lens, and',Ldrift2,'m drift')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # reflect from the Bragg mirror
    fld = Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, showPlotQ=showPlotQ, verboseQ=1, timeDomainQ=timeDomainQ, undo_slippageQ=undo_slippageQ, kxspace_outQ = skipSpaceFFTsQ)
    # plot the field
    if verbosity: print('Field after 2nd Bragg mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # drift 79 m
    Ldrift = settings['l_cavity']
    fld = rfp(fld, xlamds, dgrid, A=1, B=Ldrift, D=1, ncar=ncar, cutradius=0, dgridout=-1, kxspace_inQ = skipSpaceFFTsQ, kxspace_outQ = skipSpaceFFTsQ)
    # plot the field
    if verbosity: print('Field after',Ldrift,'m drift')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # reflect from the Bragg mirror
    fld = Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, showPlotQ=showPlotQ, verboseQ=1, timeDomainQ=timeDomainQ, undo_slippageQ=undo_slippageQ, kxspace_inQ = skipSpaceFFTsQ)
    # plot the field
    if verbosity: print('Field after 3rd Bragg mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # 0.5 m drift, 61 m focal length lens, 0.5 m drift
    #Ldrift1 = 0.5; flens = 61; Ldrift2 = 0.5 # more or less closed
    #Ldrift1 = 0.5; flens = 41; Ldrift2 = 0.5
    Ldrift1 = settings['w_cavity']/2; flens = settings['flens2']; Ldrift2 = settings['w_cavity']/2
    R = matprod([Rdrift(Ldrift1),Rlens(flens),Rdrift(Ldrift2)])
    fld = rfp(fld, xlamds, dgrid, A=R[0,0], B=R[0,1], D=R[1,1], ncar=ncar, cutradius=0, dgridout=-1)
    # plot the field
    if verbosity: print('Field after',Ldrift1,'m drift,',flens,'m focal length lens, and',Ldrift2,'m drift')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1)


    # reflect from the Bragg mirror
    fld = Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, showPlotQ=showPlotQ, verboseQ=1, timeDomainQ=timeDomainQ, undo_slippageQ=undo_slippageQ, kxspace_outQ = skipSpaceFFTsQ)
    # plot the field
    if verbosity: print('Field after 4th Bragg mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1)


    if skipTimeFFTsQ:
        t0 = time.time()
        #fld = np.fft.ifftshift(ifft(fld, axis=0), axes=0)
        fld = ifft(fld, axis=0)
        if verbosity: print('took',time.time()-t0,'seconds for ifft over t')
        timeDomainQ = 1
        undo_slippageQ = 0


    # drift 3.5 m
    Ldrift = settings['z_und_start']
    fld = rfp(fld, xlamds, dgrid, A=1, B=Ldrift, D=1, ncar=ncar, cutradius=0, dgridout=-1, kxspace_inQ = skipSpaceFFTsQ)
    # plot the field
    if verbosity: print('Field drifted',Ldrift,'m from mirror to undulator start')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1)
        

    print(time.time()-t0total,'seconds total time elapsed')
    

    return fld
    
def cavity_exit_from_mirror(fld, settings):

    ncar = settings['ncar']
    dgrid = settings['dgrid']
    xlamds = settings['xlamds']
    zsep = settings['zsep']
    isradi = settings['isradi']
    dt = xlamds * zsep * max(1,isradi) / 299792458

    skipTimeFFTsQ = 0
    try:
        skipSpaceFFTsQ = settings['skipSpaceFFTsQ']
    except:
        skipSpaceFFTsQ = 1
    try:
        showPlotQ = settings['showPlotQ']
    except:
        showPlotQ = 0
    try:
        savePlotQ = settings['savePlotQ']
    except:
        savePlotQ = 0
    try:
        verbosity = settings['verbosity']
    except:
        verbosity = 0

    t0total = time.time()

    # drift from undulator to first mirror
    Ldrift = settings['l_cavity'] - settings['z_und_end']
    fld = rfp(fld, xlamds, dgrid, A=1, B=Ldrift, D=1, ncar=ncar, cutradius=0, dgridout=-1, kxspace_inQ=0, kxspace_outQ=skipSpaceFFTsQ)
    #fld = rfp_old(fld, xlamds, dgrid, A=1, B=8.706, D=1, ncar=ncar, cutradius=0, dgridout=1.001*dgrid, verboseQ=verbosity>2)
    #fld = rfp(fld, xlamds, dgrid, A=1, B=8.706, D=1, ncar=ncar, cutradius=0, dgridout=1.001*dgrid)
    # plot the field
    if verbosity: print('Field after',Ldrift,'m drift from undulator to first mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field

        
    # reflect from the Bragg mirror
    fld = Bragg_mirror_transmit(fld, ncar, dgrid, xlamds, dt, showPlotQ=showPlotQ, verboseQ=1, timeDomainQ=1, kxspace_inQ=skipSpaceFFTsQ, kxspace_outQ=0)
    # plot the field
    if verbosity: print('Field transmitted through 1st Bragg mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field

        
    return fld

    
########################################################
    

def recirculate_to_undulator(settings):
    
    dt = settings['xlamds'] * settings['zsep'] * max(1,settings['isradi']) / 299792458
    showPlotQ = settings['showPlotQ']
    
    if settings['readfilename'] == None:
        saveFilenamePrefix = 'test'
    else:
        saveFilenamePrefix = settings['readfilename']

    if settings['readfilename'] == None:
        # make a new field
        t0 = time.time()
        dt = settings['xlamds'] * settings['zsep'] * max(1,settings['isradi']) / 299792458
        settings['ncar']=201; settings['dgrid']=750e-6; dt*=1.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=4096, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=500e-6; dt*=10.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=1024, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=200e-6; dt*=100.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=128, trms=3.e-15)
        print('took',time.time()-t0,'seconds total to make field with dimensions',fld.shape)
    else:
        ## import a field from a file on disk
        ##readfilename = '/nfs/slac/g/beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        #readfilename = '/gpfs/slac/staas/fs1/g/g.beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        ##readfilename = '/u/ra/jduris/code/genesis_dfl_tools/rfp_radiation_field_propagator/myfile.dfl'
        ##readfilename = sys.argv[1]
        print('Reading in',settings['readfilename'])
        t0 = time.time()
        fld = read_dfl(settings['readfilename'], ncar=settings['ncar']) # read the field from disk
        print('took',time.time()-t0,'seconds total to read in and format the field with dimensions',fld.shape)

    ## pad time
    #padt = np.array([1,1]) * np.int(((2**np.ceil(np.log(fld.shape[0])/np.log(2))) - fld.shape[0] - 1) / 2)
    #fld = pad_dfl_t(fld, padt)


    # plot initial beam
    #plot_fld_marginalize_3(fld, dgrid=settings['dgrid'], dt=dt, title='Initial beam')
    plot_fld_marginalize_t(fld, settings['dgrid'], dt=dt, saveFilename=saveFilenamePrefix+'_init_xy.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-2, saveFilename=saveFilenamePrefix+'_init_tx.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-1, saveFilename=saveFilenamePrefix+'_init_ty.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_init_t.png',showPlotQ=showPlotQ) # plot the imported field


    
    # propagate through cavity to return to undulator
    fld = cavity_return_to_undulator(fld, settings)
    plot_fld_marginalize_t(fld, settings['dgrid'], dt=dt, saveFilename=saveFilenamePrefix+'_recirculated_xy.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-2,saveFilename=saveFilenamePrefix+'_recirculated_tx.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-1,saveFilename=saveFilenamePrefix+'_recirculated_ty.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_recirculated_t.png',showPlotQ=showPlotQ) # plot the imported field


    #fld = unpad_dfl_t(fld, padt)
    #plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-2) # plot the imported field

    #if 0:
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        #plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
    # drift 10 m
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+10m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+20m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+30m_xy.png',showPlotQ=showPlotQ) # plot the imported field
        #fld = rfp(fld, xlamds, dgrid, A=1, B=30, D=1, ncar=ncar, cutradius=0, dgridout=-1)
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        #fld = rfp(fld, xlamds, dgrid, A=1, B=30, D=1, ncar=ncar, cutradius=0, dgridout=-1)
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
    
    ## write field to disk
    if settings['readfilename'] != None and settings['writefilename'] != None:
        try:
            print('Writing to',settings['writefilename'])
            #writefilename = readfilename + 'r'
            write_dfl(fld, settings['writefilename'])
        except:
            print('ERROR: Could not write field to file',settings['writefilename'])
    
    ## transmit through mirror
    #fld = cavity_return_to_undulator(fld, settings)
    
    return fld


def extract_from_cavity(settings):
    
    dt = settings['xlamds'] * settings['zsep'] * max(1,settings['isradi']) / 299792458
    showPlotQ = settings['showPlotQ']
    
    if settings['readfilename'] == None:
        saveFilenamePrefix = 'test'
    else:
        saveFilenamePrefix = settings['readfilename']

    if settings['readfilename'] == None:
        # make a new field
        t0 = time.time()
        dt = settings['xlamds'] * settings['zsep'] * max(1,settings['isradi']) / 299792458
        settings['ncar']=201; settings['dgrid']=750e-6; dt*=1.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=4096, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=500e-6; dt*=10.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=1024, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=200e-6; dt*=100.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=128, trms=3.e-15)
        print('took',time.time()-t0,'seconds total to make field with dimensions',fld.shape)
    else:
        ## import a field from a file on disk
        ##readfilename = '/nfs/slac/g/beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        #readfilename = '/gpfs/slac/staas/fs1/g/g.beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        ##readfilename = '/u/ra/jduris/code/genesis_dfl_tools/rfp_radiation_field_propagator/myfile.dfl'
        ##readfilename = sys.argv[1]
        print('Reading in',settings['readfilename'])
        t0 = time.time()
        fld = read_dfl(settings['readfilename'], ncar=settings['ncar']) # read the field from disk
        print('took',time.time()-t0,'seconds total to read in and format the field with dimensions',fld.shape)

    ## pad time
    #padt = np.array([1,1]) * np.int(((2**np.ceil(np.log(fld.shape[0])/np.log(2))) - fld.shape[0] - 1) / 2)
    #fld = pad_dfl_t(fld, padt)


    # plot initial beam
    #plot_fld_marginalize_3(fld, dgrid=settings['dgrid'], dt=dt, title='Initial beam')
    plot_fld_marginalize_t(fld, settings['dgrid'], dt=dt, saveFilename=saveFilenamePrefix+'_init_xy.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-2, saveFilename=saveFilenamePrefix+'_init_tx.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-1, saveFilename=saveFilenamePrefix+'_init_ty.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_init_t.png',showPlotQ=showPlotQ) # plot the imported field


    # propagate through cavity to return to undulator
    fld = cavity_exit_from_mirror(fld, settings)
    plot_fld_marginalize_t(fld, settings['dgrid'], dt=dt,saveFilename=saveFilenamePrefix+'_extracted_xy.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-2,saveFilename=saveFilenamePrefix+'_extracted_tx.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-1,saveFilename=saveFilenamePrefix+'_extracted_ty.png',showPlotQ=showPlotQ) # plot the imported field
    plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_extracted_t.png',showPlotQ=showPlotQ) # plot the imported field

    #fld = unpad_dfl_t(fld, padt)
    #plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-2) # plot the imported field

    #if 0:
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        #plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
    # drift 10 m
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_extracted+10m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_extracted+20m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_extracted+30m_xy.png',showPlotQ=showPlotQ) # plot the imported field
        #fld = rfp(fld, xlamds, dgrid, A=1, B=30, D=1, ncar=ncar, cutradius=0, dgridout=-1)
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        #fld = rfp(fld, xlamds, dgrid, A=1, B=30, D=1, ncar=ncar, cutradius=0, dgridout=-1)
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
    
    ## write field to disk
    #writefilename = '/gpfs/slac/staas/fs1/g/g.beamphysics/jduris/lh_shaping/cbxfel/testfield.dfl'
    #writefilename = '.'.join(readfilename.split('.')[:-1]) + '.rfp.dfl'
    #writefilename = readfilename + 'r'
    #write_dfl(fld, writefilename)
    
    ## transmit through mirror
    #fld = cavity_return_to_undulator(fld, settings)
    
    return fld

def main():
    
    settings = {}
    
    bad_args = False; example_cmd = ''; example_cmd_names = ''
    if len(sys.argv) < 2:
        bad_args = True
        example_cmd_names = sys.argv[0] + ' input_dflfilepath output_dflfilepath'
        example_cmd = sys.argv[0] + ' test test '
    try: 
        settings['readfilename'] = sys.argv[1]; iarg = 1
        if settings['readfilename'].lower() == 'none' or settings['readfilename'].lower() == 'test' or settings['readfilename'].lower() == 'testin': 
            settings['readfilename'] = None
    except:
        settings['readfilename'] = None
    
    try:
        iarg+=1; settings['writefilename'] = int(sys.argv[iarg])
        if settings['readfilename'] == None or settings['writefilename'].lower() == 'none' or settings['writefilename'].lower() == 'test': 
            settings['writefilename'] = None
    except:
        settings['writefilename'] = None
        
    try:
        iarg+=1; settings['ncar'] = int(sys.argv[iarg])
    except:
        example_cmd_names += ' ncar'
        example_cmd += ' ' + str(251)
    try:
        iarg+=1; settings['dgrid'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' dgrid'
        example_cmd += ' ' + str(7.5e-04)
    try:
        iarg+=1; settings['xlamds'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' xlamds'
        example_cmd += ' ' + str(1.261043e-10) # 9831.87 eV # on the Bragg reflection resonance
    #settings['xlamds'] = 1.301000e-10 # 9529.92 eV
    #settings['xlamds'] = 1.261034e-10 # 9831.95 eV # slightly off the center of the Bragg resonance
    #settings['xlamds'] = 1.261043e-10 # 9831.87 eV # on the Bragg reflection resonance
    try:
        iarg+=1; settings['zsep'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' zsep'
        example_cmd += ' ' + str(4.000000e+01)
    try:
        iarg+=1; settings['isradi'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' isradi'
        example_cmd += ' ' + str(0)
    
    settings['skipTimeFFTsQ'] = 1
    settings['skipSpaceFFTsQ'] = 1
    settings['showPlotQ'] = False
    
    # length and width of ring cavity
    settings['l_cavity'] = 149.
    settings['w_cavity'] = 1.
    # position of the undulator segment within the cavity
    try:
        iarg+=1; settings['z_und_start'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' z_und_start'
        example_cmd += ' ' + str(40)
    try:
        iarg+=1; settings['z_und_end'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' z_und_end'
        example_cmd += ' ' + str(40+80)
    # where to focus and what size
    try:
        iarg+=1; settings['z_focus'] =float(sys.argv[iarg])
    except:
        example_cmd_names += ' z_focus'
        example_cmd += ' ' + str(40+20)
    try:
        iarg+=1; settings['waist_focus'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' waist_focus'
        example_cmd += ' ' + str(20.e-6)
    # output from the undulator (should calculate from the field itself)
    try:
        iarg+=1; settings['waist_undexit'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' waist_undexit'
        example_cmd += ' ' + str(2.*57.e-6)
    try:
        iarg+=1; settings['dwaistdz_undexit'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' dwaistdz_undexit'
        example_cmd += ' ' + str(2.*1.e-6)
        
    if bad_args:
        print('Usage: ',example_cmd_names)
        print('Example: ',example_cmd)
        print('Note: set input_dflfilepath to test or none to try to use an ideal Gaussian beam')
        print('Note: set output_dflfilepath to none to suppress writing to disk')
        return
        
    print('Rayleigh range of this focused mode is',np.pi*settings['waist_focus']**2/settings['xlamds'],'m')
    
    # focal lengths of lenses 
    # to refocus the radiation to waist waist_focus at position z_focus
    lcavity = settings['l_cavity']; wcavity = settings['w_cavity']
    ray0x = settings['waist_undexit']; ray0xp = settings['dwaistdz_undexit']
    divergence = settings['xlamds'] / np.pi / settings['waist_focus']
    zundstart = settings['z_und_start']; zundend = settings['z_und_end']; zfocus = settings['z_focus']
    settings['flens1'] = ((lcavity + wcavity) * (2.*ray0x + ray0xp * (2.*lcavity + wcavity - 2.*zundend))) / ((2.*ray0x + ray0xp*(4.*lcavity + 3.*wcavity - 2.*zundend)) - (wcavity + 2.*zfocus) * divergence)
    settings['flens2'] = ((lcavity + wcavity) * divergence * (wcavity + 2.*zfocus)) / (divergence * (2.*lcavity + 3.*wcavity + 2.*zfocus) - (2.*ray0x + ray0xp * (2.*lcavity + wcavity - 2.*zundend)))
    #settings['flens1'] = (np.pi * settings['waist_focus'] * (settings['l_cavity'] + settings['w_cavity']) * (2. * settings['waist_undexit'] + 
        #settings['dwaistdz_undexit'] * (2 * settings['l_cavity'] + settings['w_cavity'] - 2 * settings['z_und_end']))) / (np.pi * settings['w_cavity'] * (2 * settings['waist_undexit'] + 
        #settings['dwaistdz_undexit'] * (4 * settings['l_cavity'] + 3 * settings['w_cavity'] - 2 * settings['z_und_end'])) - (settings['w_cavity'] + 
        #4 * settings['z_focus']) * settings['xlamds'])
    #settings['flens2'] = ((settings['l_cavity'] + settings['w_cavity']) * (settings['w_cavity'] + 
        #4 * settings['z_focus']) * settings['xlamds']) / (-np.pi * settings['waist_focus'] * (2 * settings['waist_undexit'] + 
        #settings['dwaistdz_undexit'] * (2 * settings['l_cavity'] + settings['w_cavity'] - 2 * settings['z_und_end'])) + (2 * settings['l_cavity'] + 
        #3 * settings['w_cavity'] + 4 * settings['z_focus']) * settings['xlamds'])
    print('Calculated lens focal lengths to be',settings['flens1'],'m and',settings['flens2'],'m')

    print(settings)
    
    if settings['readfilename'] == None:
        saveFilenamePrefix = 'test'
    else:
        saveFilenamePrefix = settings['readfilename']
    showPlotQ = settings['showPlotQ']
    
    # propagate beam from end of undulator to beginning of undulator
    # save dfl and make plots
    fld = recirculate_to_undulator(settings)
        
    # see how this mode evolves
    fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=20, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+20m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=20, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+40m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    
    # propagate beam past the first Bragg mirror
    # dont save dfl but do save plots
    fld = extract_from_cavity(settings)
    
    # see how this mode evolves
    fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=20, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_extracted+20m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=20, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_extracted+40m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    
    
if __name__ == '__main__':
    main()