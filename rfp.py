#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# rfp - radiation field propagator
# tools for manipulating genesis dfl field files
# J. Duris jduris@slac.stanford.edu

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time

try: # much faster to parallelize
    #DISABLE_PYFFTW
    # pyfftw is not without problems (especially when caching)
    # https://stackoverflow.com/questions/6365623/improving-fft-performance-in-python https://gist.github.com/fnielsen/99b981b9da34ae3d5035
    import pyfftw, multiprocessing
    pyfftw.interfaces.cache.enable() # caching was working fine and giving 3x speedups
    #pyfftw.interfaces.cache.disable() # but had to disable to fix some random corruption that randomly started
    # manually plan ffts? https://stackoverflow.com/questions/55014239/how-to-do-100000-times-2d-fft-in-a-faster-way-using-python
    threads = multiprocessing.cpu_count()
    overwrite_input = True
    #planner_effort = None
    planner_effort = 'FFTW_ESTIMATE' # slightly suboptimal plan being used, but with a substantially quicker first-time planner step
    #planner_effort = 'FFTW_MEASURE' # default?
    def fft(array, axis=None, overwrite_input=overwrite_input):
        return pyfftw.interfaces.numpy_fft.fft(array,axis=axis,threads=threads, planner_effort=planner_effort,overwrite_input=overwrite_input)
    def fftn(array, axes=None, overwrite_input=overwrite_input):
        return pyfftw.interfaces.numpy_fft.fftn(array,axes=axes,threads=threads, planner_effort=planner_effort,overwrite_input=overwrite_input)
    def fft2(array, axes=None, overwrite_input=overwrite_input):
        return pyfftw.interfaces.numpy_fft.fft2(array,axes=axes,threads=threads, planner_effort=planner_effort,overwrite_input=overwrite_input)
    def ifft(array, axis=None, overwrite_input=overwrite_input):
        return pyfftw.interfaces.numpy_fft.ifft(array,axis=axis,threads=threads, planner_effort=planner_effort,overwrite_input=overwrite_input)
    def ifftn(array, axes=None, overwrite_input=overwrite_input):
        return pyfftw.interfaces.numpy_fft.ifftn(array,axes=axes,threads=threads, planner_effort=planner_effort,overwrite_input=overwrite_input)
    def ifft2(array, axes=None, overwrite_input=overwrite_input):
        return pyfftw.interfaces.numpy_fft.ifft2(array,axes=axes,threads=threads, planner_effort=planner_effort,overwrite_input=overwrite_input)
except:
    print('WARNING: Could not load pyfftw for faster, parallelized ffts. (At command line, try: pip install pyfftw)')
    def fft(array, axis=None):
        return np.fft.fft(array,axis=axis)
    def fftn(array, axes=None):
        return np.fft.fftn(array,axes=axes)
    def fft2(array, axes=None):
        return np.fft.fft2(array,axes=axes)
    def ifft(array, axis=None):
        return np.fft.ifft(array,axis=axis)
    def ifftn(array, axes=None):
        return np.fft.ifftn(array,axes=axes)
    def ifft2(array, axes=None):
        return np.fft.ifft2(array,axes=axes)
    
# change default plotting font size
import matplotlib
font = {'family' : 'normal', 'size' : 14} # reasonable for on-screen displays
font = {'family' : 'normal', 'size' : 22} # for smaller plots in figures
#font['weight'] = 'bold'
matplotlib.rc('font', **font)

# jetvar: jet color map with white min
# https://stackoverflow.com/questions/9893440/python-matplotlib-colormap
cdict = {'red': ((0., 1, 1), (0.05, 1, 1), (0.11, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
         'green': ((0., 1, 1), (0.05, 1, 1), (0.11, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
         'blue': ((0., 1, 1), (0.05, 1, 1), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))}
jetvar_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
cmap = jetvar_cmap
# cmap = 'jet'
#cmap = 'viridis'

try:
    # inferno reversed colormap with white background
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    inferno_r_cmap = cm.get_cmap('inferno_r')
    # my_cmap.set_under('w') # don't seem to work
    xr = np.linspace(0, 1, 256)
    inferno_r_cmap_listed = inferno_r_cmap(xr)
    inferno_r_whitebg_cmap_listed = np.vstack((np.array([np.ones(4)+(inferno_r_cmap_listed[0]-np.ones(4))*x for x in np.linspace(0,1,int(256/8))]),inferno_r_cmap_listed[:-int(256/16)]))
    inferno_r_whitebg_cmap = ListedColormap(inferno_r_whitebg_cmap_listed)
    cmap = inferno_r_whitebg_cmap
except:
    pass
    


def Rdrift(L):
    return np.array([[1.,L],[0.,1.]])
    
def Rlens(f):
    return np.array([[1.,0.],[-1./f,1.]])
    
def matprod(matrix_list):
    # arrange matrices in order of application (first to last)
    mat = np.eye(len(matrix_list[0]))
    for m in matrix_list:
        mat = np.dot(m, mat)
    return mat
    
    
def fwhm(array, nkeep=2, plotQ=False, relcut=0.5, abscut=None):
    array_max = array.max()
    arg_max = array.argmax()
    if type(abscut) is type(None):
        scaled = array-relcut*array_max
    else:
        scaled = array-abscut
    inds = np.array(range(len(array)))
    
    # find lower crossing
    try:
        xlow = np.max(inds[(inds < arg_max) * (scaled < 0.)])
        xlows = xlow + np.array(range(nkeep)) - (nkeep-1)/2
        xlows = xlows[xlows > 0]
        xlows = np.array(xlows, dtype=np.int)
        if len(xlows):
            pfl = np.polyfit(xlows,scaled[xlows],1)
            xlow = -pfl[1]/pfl[0]
        else:
            xlow = np.nan
    except:
        xlow = np.nan
    
    # find upper crossing
    try:
        xhigh = np.min(inds[(inds > arg_max) * (scaled < 0.)])
        xhighs = xhigh + np.array(range(nkeep)) - (nkeep-1)/2 - 1
        xhighs = xhighs[xhighs > 0]
        xhighs = np.array(xhighs, dtype=np.int)
        if len(xhighs):
            pfh = np.polyfit(xhighs,scaled[xhighs],1)
            xhigh = -pfh[1]/pfh[0]
        else:
            xhigh = np.nan
    except:
        xhigh = np.nan
                
    if plotQ:
        import matplotlib.pyplot as plt
        try:
            plt.plot(xlows,scaled[xlows],'o')
            plt.plot(xlow,0,'o')
            plt.show()
        except:
            pass
        try:
            plt.plot(xhighs,scaled[xhighs],'o')
            plt.plot(xhigh,0,'o')
            plt.show()
        except:
            pass
        
    # determine width
    return np.array([np.abs(xhigh-xlow),np.abs(arg_max-xlow),np.abs(arg_max-xhigh),xlow,xhigh])


# maximum width at half max
def mwhm(array, nkeep=2, plotQ=False, relcut=0.5, abscut=None):
    array_max = array.max()
    arg_max = array.argmax()
    if type(abscut) is type(None):
        scaled = array-relcut*array_max
    else:
        scaled = array-abscut
    inds = np.array(range(len(array)))
    
    # find lower crossing
    try:
        xlow = np.min(inds[(inds < arg_max) * (scaled > 0.)])
        xlows = xlow + np.array(range(nkeep)) - (nkeep-1)/2
        xlows = xlows[xlows > 0]
        xlows = np.array(xlows, dtype=np.int)
        if len(xlows):
            pfl = np.polyfit(xlows,scaled[xlows],1)
            xlow = -pfl[1]/pfl[0]
        else:
            xlow = np.nan
    except:
        xlow = np.nan
    
    # find upper crossing
    try:
        xhigh = np.max(inds[(inds > arg_max) * (scaled > 0.)])
        xhighs = xhigh + np.array(range(nkeep)) - (nkeep-1)/2 - 1
        xhighs = xhighs[xhighs > 0]
        xhighs = np.array(xhighs, dtype=np.int)
        if len(xhighs):
            pfh = np.polyfit(xhighs,scaled[xhighs],1)
            xhigh = -pfh[1]/pfh[0]
        else:
            xhigh = np.nan
    except:
        xhigh = np.nan
                
    if plotQ:
        try:
            plt.plot(xlows,scaled[xlows],'o')
            plt.plot(xlow,0,'o')
            plt.show()
        except:
            pass
        try:
            plt.plot(xhighs,scaled[xhighs],'o')
            plt.plot(xhigh,0,'o')
            plt.show()
        except:
            pass
        
    # determine width
    return np.array([np.abs(xhigh-xlow),np.abs(arg_max-xlow),np.abs(arg_max-xhigh),xlow,xhigh])


def make_gaus_slice(ncar=251, dgrid=400.e-6, w0=40.e-6):
    
    xs = np.linspace(-1,1,ncar) * dgrid
    ys = xs
    xv, yv = np.meshgrid(xs, ys) 
    
    sigx2 = (w0 / 2.)**2;
    fld = np.exp( -0.25 * (xv**2 + yv**2) / sigx2 ) + 1j * 0
    fld = fld[None,:,:]
    
    return fld

def make_gaus_beam(ncar=251, dgrid=400.e-6, w0=40.e-6, dt=1e-6/3e8, t0=0., nslice=11, trms=2e-6/3e8):
    
    xs = np.linspace(-1,1,ncar) * dgrid
    ys = xs
    xv, yv = np.meshgrid(xs, ys) 
    
    sigx2 = (w0 / 2.)**2;
    fld = np.exp( -0.25 * (xv**2 + yv**2) / sigx2 )
    
    ts = dt * np.arange(nslice); ts -= np.mean(ts)
    amps = np.exp(-0.25 * ((ts-t0)/trms)**2)
    
    fld0 = np.zeros([nslice,ncar,ncar]) + 1j * 0.
    
    for ia, a in enumerate(amps):
        fld0[ia] = a * fld
    
    return fld0

# note: if slice < 0, then this thing marginalizes axis -slice == [1,2,3]
def plot_fld_slice(fld, dgrid = 400.e-6, dt=1e-6/3e8, slice=None, ax=None, saveFilename=None, showPlotQ=True, savePlotQ=True, plotPowerQ=False, logScaleQ=False):
    try:
        if slice == None:
            power = np.abs(fld[0])**2
        elif slice == -1:
            power = np.sum(np.abs(fld)**2, axis=1)
        elif slice == -2:
            power = np.sum(np.abs(fld)**2, axis=2)
        elif slice == -3:
            power = np.sum(np.abs(fld)**2, axis=0)
        else:
            power = np.abs(fld[slice])**2
        nslice = fld.shape[0]
    except:
        power = np.abs(fld)**2
        nslice = 1
    ncar = fld.shape[1]
    norm = np.sum(power)
    xproj = np.sum(power, axis=1)
    yproj = np.sum(power, axis=0)
    
    transverse_grid = np.linspace(-1,1,ncar) * dgrid * 1e6
    ts = dt * np.arange(nslice); ts -= np.mean(ts); temporal_grid = ts * 1e15
    
    if slice == -1:
        xs = temporal_grid; ys = transverse_grid
        xlabel = 'Time (fs)'; ylabel = 'y (um)'; xu = 'fs'; yu = 'um'; xn = 't'; yn = 'y'
    elif slice == -2:
        xs = temporal_grid; ys = transverse_grid
        xlabel = 'Time (fs)'; ylabel = 'x (um)'; xu = 'fs'; yu = 'um'; xn = 't'; yn = 'x'
    else:
        xs = transverse_grid; ys = transverse_grid
        xlabel = 'x (um)'; ylabel = 'y (um)'; xu = 'um'; yu = 'um'; xn = 'x'; yn = 'y'
    xmean = np.dot(xs, xproj) / norm
    ymean = np.dot(ys, yproj) / norm
    xrms = np.sqrt(np.dot(xs**2, xproj) / norm - xmean**2)
    yrms = np.sqrt(np.dot(ys**2, yproj) / norm - ymean**2)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    xfwhm = fwhm(xproj) * dx
    yfwhm = fwhm(yproj) * dy
    energy_uJ = norm*dt * 1e6
    
    ndecimals = 1
    xmean = np.around(xmean, ndecimals); ymean = np.around(ymean, ndecimals)
    xrms = np.around(xrms, ndecimals); yrms = np.around(yrms, ndecimals)
    xfwhm = np.around(xfwhm, ndecimals)[0]; yfwhm = np.around(yfwhm, ndecimals)[0]
    energy_uJ = np.around(energy_uJ, ndecimals)
    #print('norm =',norm,'   x,y mean =',xmean,ymean, '   x,y rms =', xrms,yrms, '   wx,wy =', 2*xrms,2*yrms)
    if ndecimals == 0:
        xmean = int(xmean); ymean = int(ymean)
        xrms = int(xrms); yrms = int(yrms)
        energy_uJ = int(energy_uJ)
        try:
            xfwhm = int(xfwhm); yfwhm = int(yfwhm)
        except:
            pass
    
    
    #xmean *= 1e6; ymean *= 1e6; xrms *= 1e6; yrms *= 1e6;
    print('norm =',norm,'   ','energy =',energy_uJ,' uJ   ',xn,',',yn,' mean =',xmean,xu,',',ymean, yu,'    ',xn,',',yn,' rms =', xrms,xu,',',yrms, yu, '    w'+xn,', w'+yn,'=', 2*xrms,xu,',',2*yrms,yu,'   ',xn,',',yn,' fwhm =',xfwhm,xu,',',yfwhm, yu)
#     annotation1 = 'energy '+'{:e}'.format(energy_uJ)+' uJ\n'
    annotation1 = 'energy '+str(energy_uJ)+' uJ\n'
    annotation1 += yn+' mean '+str(ymean)+' '+yu+'\n'
    annotation1 += yn+' rms '+str(yrms)+' '+yu+'\n'
    annotation1 += yn+' fwhm '+str(yfwhm)+' '+yu
    annotation2 = xn+' mean '+str(xmean)+' '+xu+'\n'
    annotation2 += xn+' rms '+str(xrms)+' '+xu+'\n'
    annotation2 += xn+' fwhm '+str(xfwhm)+' '+xu
    
    aspect = (min(xs)-max(xs)) / (min(ys)-max(ys))
    # if ax is defined, then collecting plots
    showPlotQ &= (ax == None); savePlotQ &= (ax == None)
    if ax == None: ax = plt.gca()
    if plotPowerQ:
        ax.plot(xs,xproj)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Power (W)')
#         annotation2 = 'energy '+'{:e}'.format(energy_uJ)+' uJ\n' + annotation2
        annotation2 = 'energy '+str(energy_uJ)+' uJ\n' + annotation2
        ax.text(min(xs)+0.01*(max(xs)-min(xs)),min(xproj)+0.86*(max(xproj)-min(xproj)),annotation2,fontsize=10)
    else:
        ax.imshow(power.T, extent=(min(xs),max(xs),min(ys),max(ys)), origin='lower', aspect=aspect, cmap=cmap)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); plt.tight_layout(); 
        ax.text(min(xs)+0.01*(max(xs)-min(xs)),min(ys)+0.86*(max(ys)-min(ys)),annotation1,fontsize=10)
        ax.text(min(xs)+0.01*(max(xs)-min(xs)),min(ys)+0.01*(max(ys)-min(ys)),annotation2,fontsize=10)
        ax.plot(xs, min(ys)+xproj/max(xproj)*0.15*(max(ys)-min(ys)),'k')
        ax.plot(min(xs)+yproj/max(yproj)*0.15*(max(xs)-min(xs)), ys,'k')
    if logScaleQ:
        ax.set_yscale('log')
    if saveFilename != None and savePlotQ:
        plt.savefig(saveFilename, bbox_inches='tight')
    if showPlotQ:
        plt.show()
    plt.close()
    
def plot_fld_power(fld, dt, ax=None, saveFilename=None, showPlotQ=True, savePlotQ=True, logScaleQ=False):
    plot_fld_slice(fld, 400e-6, dt, slice=-1, ax=ax, saveFilename=saveFilename, showPlotQ=showPlotQ, savePlotQ=savePlotQ, plotPowerQ=True, logScaleQ=logScaleQ)
    
def plot_fld_marginalize_3(fld, dgrid, dt, title=None):
    fig, axs = plt.subplots(1,3)
    fig.suptitle(title)
    plot_fld_slice(fld, dgrid=dgrid, dt=dt, slice=-3, ax=axs[0])
    plot_fld_slice(fld, dgrid=dgrid, dt=dt, slice=-2, ax=axs[1])
    plot_fld_slice(fld, dgrid=dgrid, dt=dt, slice=-1, ax=axs[2])
    plt.tight_layout(); 
    #plt.subplot_tool()
    #plt.subplots_adjust(left=0.1, 
                    #bottom=0.1,  
                    #right=0.9,  
                    #top=0.9,  
                    #wspace=0.4,  
                    #hspace=0.4) 
    plt.show()

def plot_fld_marginalize_t(fld, dgrid = 400.e-6, dt=1e-6/3e8, saveFilename=None, showPlotQ=True, savePlotQ=True):
    plot_fld_slice(fld, dgrid=dgrid, dt=dt, slice=-3, saveFilename=saveFilename, showPlotQ=showPlotQ, savePlotQ=savePlotQ)

# https://stackoverflow.com/questions/10917495/matplotlib-imshow-in-3d-plot
# https://stackoverflow.com/questions/30464117/plotting-a-imshow-image-in-3d-in-matplotlib
# voxels (slow?) https://terbium.io/2017/12/matplotlib-3d/
# slider https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
def plot_fld_3d(fld, dgrid = 400.e-6, dt=1e-6/3e8):
    
    # time grid
    nslice = fld.shape[0]
    ts = dt * np.arange(nslice); ts -= np.mean(ts)
    ts *= 1e15
    print(ts)
    
    # transverse grid
    ncar = fld.shape[1]
    dgridum = dgrid * 1e6
    xs = np.linspace(-1,1,ncar) * dgridum; ys = xs
    xv, yv = np.meshgrid(xs, ys) 
    
    # power vs t
    power_vs_t = np.sum(np.sum(fld, axis=1),axis=1)
    power_vs_t /= np.max(power_vs_t)
    
    # make figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    view_z_along_xaxis = True

    # plot slices
    for s in range(nslice):
    
        # transparency gradient colormap 
        # https://kbkb-wx-python.blogspot.com/2015/12/python-transparent-colormap.html
        import matplotlib.colors as mcolors
        ncontourlevels = 21
        colors = [(1,0,0,c*(c>0)) for c in power_vs_t[s] * np.linspace(-1./(ncontourlevels-1),1,ncontourlevels)]
        my_cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=ncontourlevels)
        my_cmap.set_under(color='k', alpha=0)
    
        #ax.imshow(np.abs(fld[s])**2, zs=ts[s], extent=(-dgridum,dgridum,-dgridum,dgridum), 
                   #origin='lower', interpolation='none', cmap=my_cmap, vmin=.001)
        
        if view_z_along_xaxis:
            cset = ax.contourf(np.abs(fld[s])**2, yv, xv, ncontourlevels, zdir='x', offset = ts[s], cmap=my_cmap)
        else:
            cset = ax.contourf(xv, yv, np.abs(fld[s])**2, ncontourlevels, zdir='z', offset = ts[s], cmap=my_cmap)
    
    if view_z_along_xaxis:
        ax.set_xlim([min(ts),max(ts)])
        ax.set_ylim([min(xs),max(xs)])
        ax.set_zlim([min(ys),max(ys)])
        ax.set_zlabel('y (um)')
        ax.set_ylabel('x (um)')
        ax.set_xlabel('t (fs)')
    else:
        ax.set_zlim([min(ts),max(ts)])
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_zlabel('t (fs)')

    plt.tight_layout(); plt.show()

def read_dfl(filename, ncar=251, verboseQ=1, conjugate_field_for_genesis=True, swapxyQ=True):
    
    t0 = time.time()
    fld = np.fromfile(filename, dtype='complex128')
    if verboseQ: print(time.time()-t0,'seconds to read in fld array',fld.shape)
    
    nslice = int(len(fld) / ncar / ncar)
    if verboseQ: print('read',nslice,'slices')
    
    t0 = time.time()
    fld = fld.reshape(nslice,ncar,ncar)
    if verboseQ: print(time.time()-t0,'seconds to reshape fld array',fld.shape)
    
    if swapxyQ:
        t0 = time.time()
        fld = np.einsum('ikj->ijk',fld)
        if verboseQ: print(time.time()-t0,'seconds to swap x and y in array',fld.shape)
    
    if conjugate_field_for_genesis:
        return fld.conjugate()
    else:
        return fld

def write_dfl(filename, fld, conjugate_field_for_genesis=True, swapxyQ=True):
    
    if swapxyQ:
        t0 = time.time()
        fld = np.einsum('ikj->ijk',fld)
        if verboseQ: print(time.time()-t0,'seconds to swap x and y in array',fld.shape)
    
    f=open(filename,"wb")
    if conjugate_field_for_genesis:
        fld.astype('complex128').conjugate().tofile(f)
    else:
        fld.astype('complex128').tofile(f)
    f.close()
    
# pad the grids for better frequency resolution with ffts

def pad_dfl(fld, pads):
    # pads should be of the form [[i0lo,i0hi],[i1lo,i1hi],[i2lo,i2hi]]
    fld = np.pad(fld, pads)
    return fld
    
def pad_dfl_t(fld, pads):
    fld = np.pad(fld, [pads,[0,0],[0,0]])
    return fld
    
def pad_dfl_x(fld, pads):
    fld = np.pad(fld, [[0,0],pads,[0,0]])
    return fld
    
def pad_dfl_y(fld, pads):
    fld = np.pad(fld, [[0,0],[0,0],pads])
    return fld
    
def pad_dfl_xy(fld, pads):
    fld = np.pad(fld, [[0,0],pads,pads])
    return fld
    
def unpad_dfl(fld, pads):
    fld = fld[pads[0,0]:-pads[0,1],pads[1,0]:-pads[1,1],pads[2,0]:-pads[2,1]]
    return fld
    
def unpad_dfl_t(fld, pads):
    fld = fld[pads[0]:-pads[1],:,:]
    return fld
    
def unpad_dfl_x(fld, pads):
    fld = fld[:,pads[0]:-pads[1],:]
    return fld
    
def unpad_dfl_y(fld, pads):
    fld = fld[:,:,pads[0]:-pads[1]]
    return fld
    
def unpad_dfl_xy(fld, pads):
    fld = fld[:,pads[0]:-pads[1],pads[0]:-pads[1]]
    return fld

# slip field forward along the temporal grid and slip in zeros in the tail
def slip_fld(fld, dt, slippage_time, trailing_scale_factor=1e-6):
    nslip = int(np.floor(slippage_time / dt))
    fld2 = np.roll(fld,nslip,axis=0)
    fld2[:nslip] *= trailing_scale_factor
    # maybe should advance the phase of the radiation by the remainder slippage time also...
    return fld2

# use the c++ version (single-thread so slower)
def run_rfp_cpp_binary(filenamein, filenameout, xlamds, dgrid, A, B, D, intensity_scale_factor=1., ncar=0, nslip=0, verboseQ=0, cutradius=0, dgridout=-1):
    
    import os
    cmd = 'rfp ' + filenamein + ' ' 
    cmd += filenameout + ' '
    cmd += str(xlamds) + ' ' # xlamds
    cmd += str(dgrid) + ' ' # dgrid
    cmd += str(A) + ' ' # A
    cmd += str(B) + ' ' # B
    cmd += str(D) + ' ' # D
    cmd += str(intensity_scale_factor) + ' ' # [intensity_scale_factor] 
    cmd += str(ncar) + ' ' # [ncar]
    cmd += str(nslip) + ' ' # [nslip]
    cmd += str(int(verboseQ)) + ' ' # [verboseQ]
    cmd += str(cutradius) + ' ' # [cutradius]
    cmd += str(dgridout) + ' ' # [dgridout]
    if verboseQ: print(cmd)
    os.system(cmd)

# function to package up rfp for ease of use
def rfp_cpp_binary(fld, xlamds, dgrid, A, B, D, intensity_scale_factor=1., ncar=0, nslip=0, verboseQ=0, cutradius=0, dgridout=-1):
    
    # temp file paths
    filename = '/dev/shm/tmp'+str(np.random.randint(1000000000))+'.dfl'
    filenamerfp = filename+'.rfp'
    print(filename)
    print(filenamerfp)
    
    # write this field to disk
    t0 = time.time()
    write_dfl(filename, fld)
    print('took',time.time()-t0,'seconds total to write fld to disk with shape',fld.shape)

    # apply rfp to the file
    t0 = time.time()
    run_rfp_cpp_binary(filename, filenamerfp, xlamds, dgrid, A, B, D, intensity_scale_factor=intensity_scale_factor, ncar=ncar, nslip=nslip, verboseQ=verboseQ, cutradius=cutradius, dgridout=dgridout)
    print('took',time.time()-t0,'seconds total to run rfp on fld with shape',fld.shape)

    # read back in the processed file
    t0 = time.time()
    fld = read_dfl(filenamerfp, ncar=ncar) # read the resulting field in from disk
    print('took',time.time()-t0,'seconds total to read fld with shape',fld.shape)

    # remove the temporary files
    os.system('rm '+filename)
    os.system('rm '+filenamerfp)
    
    return fld 
    
    
# select only slices with significant power
def threshold_slice_selection(fld, slice_processing_relative_power_threshold = 1e-6, verboseQ=False):
        #slice_processing_relative_power_threshold = 1e-6 # only propagate slices where there's beam (0 disables)
        pows = np.sum(np.abs(fld)**2,axis=(1,2))
        slice_selection = pows >= np.max(pows) * slice_processing_relative_power_threshold
        #if verboseQ:
        u0 = np.sum(pows); u1 = np.sum(pows[slice_selection])
        print('INFO: threshold_slice_selection - Fraction of power lost is',1.-u1/u0,'for slice_processing_relative_power_threshold of',slice_processing_relative_power_threshold)
        return slice_selection

# Siegman collimating transform
# NOTE: dx should be 2*dgrid/ncar
def st2(fld, lambda_radiation, dgridin, dgridout, ABDlist, ncar, verboseQ = False, outQ = False):

   tau = 6.283185307179586
   scale=1.; M=dgridout/dgridin

   if outQ:
      dx = 2.*dgridout/(ncar-1.)
      phasefactor = (1./M-ABDlist[2])*dx*dx*tau/2./lambda_radiation/ABDlist[1];
      scale = dgridout; #for genesis, each cell is intensity so don't need this
      if verboseQ:
          print("(1./M-ABDlist[2]) = ",(1./M-ABDlist[2]), "\tdx*dx*tau/2. = ", dx*dx*tau/2., "\tlambda = " , lambda_radiation , "\tABDlist[1] = " , ABDlist[1]);

   else:
      dx = 2.*dgridin/(ncar-1.);
      phasefactor = (M-ABDlist[0])*dx*dx*tau/2./lambda_radiation/ABDlist[1];
      scale = dgridin; #//for genesis, each cell is intensity so don't need this
      if verboseQ:
          print("(M-ABDlist[0]) = ", (M-ABDlist[0]), "\tdx*dx*tau/2. = ", dx*dx*tau/2., "\tlambda = ", lambda_radiation, "\tABDlist[1] = ", ABDlist[1])

   dxscale = dx / scale;
   
   # calculate the phase mask
   igrid = np.arange(ncar)-np.floor(ncar/2)
   phases = phasefactor * (igrid ** 2.)
   pxv, pyv = np.meshgrid(phases, phases)
   phasor_mask = np.exp(1j * (pxv + pyv))
   
   # apply the phase mask to each slice
   fld *= phasor_mask
   
   return fld

# inverse Siegman collimating transform
def ist2(fld, lambda_radiation, dgridin, dgridout, ABDlist, ncar, verboseQ = False):
   return st2(fld, lambda_radiation, dgridin, dgridout, ABDlist, ncar, verboseQ, True)

# Siegman collimated Huygen's kernel
def sk2(fld, lambda_radiation, dgridin, dgridout, ABDlist, ncar, verboseQ = False):
#/* Propagate radiation stored in data. */

   tau = 6.283185307179586;
   M=dgridout/dgridin;
   
   #//Nc = M*pow(dgridin,2.)/lambda_radiation/ABDlist[1]; # collimated Fresnel number 
   Nc = M*np.power(2.*dgridin,2.)/lambda_radiation/ABDlist[1]; # collimated Fresnel number 
   phasefactor = tau/2./Nc;
   
   # calculate the phase mask
   midpoint = np.floor(ncar/2)
   igrid = np.fft.ifftshift(np.arange(ncar)-midpoint)
   phases = phasefactor * igrid ** 2.
   pxv, pyv = np.meshgrid(phases, phases)
   phasor_mask = np.exp(1j * (pxv + pyv))
   
   # apply the phase mask to each slice
   fld *= phasor_mask
   
   return fld
   
# function to package up rfp for ease of use
def rfp(fld, xlamds, dgridin, A, B, D, intensity_scale_factor=1., ncar=0, nslip=0, verboseQ=0, cutradius=0, dgridout=-1, kxspace_inQ=False, kxspace_outQ=False, slice_processing_relative_power_threshold=0):
        
    if A == 1. and B == 0. and D == 1.:
        if kxspace_inQ == kxspace_outQ:
            return fld
        else:
            use_siegman_transform = False
            use_siegman_kernel = False
    else:
        use_siegman_transform = True
        use_siegman_kernel = True
    
    ABDlist = [A,B,D]
    if dgridout < 0:
        dgridout = dgridin
    M = dgridout/dgridin
    use_siegman_transform &= (1./M-ABDlist[2] != 0. or M-ABDlist[0] != 0.)
    
    # process only slices with power
    if slice_processing_relative_power_threshold > 0:
        t0 = time.time()
        fld0 = fld
        slice_selection = threshold_slice_selection(fld, slice_processing_relative_power_threshold, verboseQ=verboseQ)
        #print('np.shape(fld) =',np.shape(fld))
        fld = fld[slice_selection]
        #print('np.shape(fld) =',np.shape(fld))
        #if verboseQ: 
        if verboseQ: print('took',time.time()-t0,'seconds for selecting only',len(fld),'slices with power / max(power) >',slice_processing_relative_power_threshold,'for processing')

    #plot_fld_marginalize_t(fld, dgrid) # plot the imported field

    # siegman collimating beam transform
    if use_siegman_transform:
        if kxspace_inQ:
            print('ERROR: applying Siegman collimating transform to reciprocal space instead of real space!')
        t0 = time.time()
        fld = st2(fld, xlamds, dgridin, dgridout, ABDlist, ncar)
        if verboseQ: print('took',time.time()-t0,'seconds total to apply Siegman collimating transform to fld with shape',fld.shape)
        
    #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
    
    # fft
    if kxspace_inQ:
        t0 = time.time()
        fld = fft(fld,axis=2)
        if verboseQ: print('took',time.time()-t0,'seconds total to apply y fft to fld with shape',fld.shape)
    else:
        t0 = time.time()
        #fld = fft2(fld) # defaults to last two axes
        fld = fft(fld, axis=1)
        fld = fft(fld, axis=2)
        if verboseQ: print('took',time.time()-t0,'seconds total to apply x and y ffts to fld with shape',fld.shape)
    
    #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
    
    # siegman collimated Huygen's kernel
    if use_siegman_kernel:
        t0 = time.time()
        fld = sk2(fld, xlamds, dgridin, dgridout, ABDlist, ncar)
        if verboseQ: print('took',time.time()-t0,'seconds total to apply Siegman collimated Huygens kernel to fld with shape',fld.shape)
    
    #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
    
    # ifft
    if kxspace_outQ:
        t0 = time.time()
        fld = ifft(fld,axis=2)
        if verboseQ: print('took',time.time()-t0,'seconds total to apply y ifft to fld with shape',fld.shape)
    else:
        t0 = time.time()
        #fld = ifft2(fld) # defaults to last two axes
        fld = ifft(fld, axis=1)
        fld = ifft(fld, axis=2)
        if verboseQ: print('took',time.time()-t0,'seconds total to apply x and y iffts to fld with shape',fld.shape)
    
    # inverse siegman collimating beam transform
    if use_siegman_transform:
        if kxspace_outQ:
            print('ERROR: applying Siegman collimating transform to reciprocal space instead of real space!')
        t0 = time.time()
        fld = ist2(fld, xlamds, dgridin, dgridout, ABDlist, ncar)
        if verboseQ: print('took',time.time()-t0,'seconds total to apply Siegman collimating transform to fld with shape',fld.shape)
    
    # release processing mask
    if slice_processing_relative_power_threshold > 0:
        t0 = time.time()
        fld0 *= 0. # clear field
        fld0[slice_selection] = fld # overwrite field with processed field
        fld = fld0
        if verboseQ: print('took',time.time()-t0,'seconds to release selection for slices with power / max(power) >',slice_processing_relative_power_threshold,'for processing')
    
    
    #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
    
    return fld 


def main():
    import sys
    
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
    try:
        iarg+=1; settings['A'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' A'
        example_cmd += ' ' + str(1)
    try:
        iarg+=1; settings['B'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' B'
        example_cmd += ' ' + str(0)
    try:
        iarg+=1; settings['D'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' D'
        example_cmd += ' ' + str(1)
    try:
        iarg+=1; settings['dgridout'] = float(sys.argv[iarg])
    except:
        dgridout = -1
        example_cmd_names += ' dgridout'
        example_cmd += ' ' + str(-1)
    try:
        iarg+=1; settings['showPlotQ'] = int(sys.argv[iarg])
    except:
        example_cmd_names += ' showPlotQ'
        example_cmd += ' ' + str(0)
    try:
        iarg+=1; settings['savePlotQ'] = int(sys.argv[iarg])
    except:
        example_cmd_names += ' savePlotQ'
        example_cmd += ' ' + str(1)
    try:
        iarg+=1; settings['slice_processing_relative_power_threshold'] = float(sys.argv[iarg])
    except:
        example_cmd_names += ' slice_processing_relative_power_threshold'
        example_cmd += ' ' + str(0)
    try:
        iarg+=1; settings['verboseQ'] = int(sys.argv[iarg])
    except:
        settings['verboseQ'] = 1
        example_cmd_names += ' verboseQ'
        example_cmd += ' ' + str(1)
    
    if bad_args:
        print('Usage: ',example_cmd_names)
        print('Example: ',example_cmd)
        print('Note: set input_dflfilepath to test or none to try to use an ideal Gaussian beam')
        print('Note: set output_dflfilepath to none to suppress writing to disk')
        return
    
    print(settings); s = settings
    
    if settings['readfilename'] == None:
        saveFilenamePrefix = 'test'
    else:
        saveFilenamePrefix = settings['readfilename']
    showPlotQ = settings['showPlotQ']
    savePlotQ = settings['savePlotQ']
    verboseQ = settings['verboseQ']
    
    # load field
    dt = s['xlamds'] * s['zsep'] * max(1,s['isradi']) / 299792458
    if settings['readfilename'] == None:
        # make a new field
        t0 = time.time()
        settings['ncar']=201; settings['dgrid']=750e-6; dt*=1.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=4096, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=500e-6; dt*=10.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=1024, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=200e-6; dt*=100.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=128, trms=3.e-15)
        if verboseQ: print('took',time.time()-t0,'seconds total to make field with dimensions',fld.shape)
    else:
        ## import a field from a file on disk
        ##readfilename = '/nfs/slac/g/beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        #readfilename = '/gpfs/slac/staas/fs1/g/g.beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        ##readfilename = '/u/ra/jduris/code/genesis_dfl_tools/rfp_radiation_field_propagator/myfile.dfl'
        ##readfilename = sys.argv[1]
        print('Reading in',settings['readfilename'])
        t0 = time.time()
        fld = read_dfl(settings['readfilename'], ncar=settings['ncar'], verboseQ=verboseQ) # read the field from disk
        if verboseQ: print('took',time.time()-t0,'seconds total to read in and format the field with dimensions',fld.shape)
    
    # plot the field
    if showPlotQ or savePlotQ:
        import logging
        logging.getLogger('matplotlib.font_manager').disabled = True # suppress findfont warnings
        plot_fld_marginalize_t(fld, settings['dgrid'], dt=dt, saveFilename=saveFilenamePrefix+'_init_xy.png',showPlotQ=showPlotQ,savePlotQ=savePlotQ) # plot the imported field
        if fld.shape[0] > 1:
            plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-2, saveFilename=saveFilenamePrefix+'_init_tx.png',showPlotQ=showPlotQ,savePlotQ=savePlotQ) # plot the imported field
            plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-1, saveFilename=saveFilenamePrefix+'_init_ty.png',showPlotQ=showPlotQ,savePlotQ=savePlotQ) # plot the imported field
            plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_init_t.png',showPlotQ=showPlotQ,savePlotQ=savePlotQ) # plot the imported field
    
    # transport field
    fld = rfp(fld, s['xlamds'], s['dgrid'], A=s['A'], B=s['B'], D=s['D'], ncar=s['ncar'], cutradius=0, dgridout=s['dgridout'], kxspace_inQ=0, kxspace_outQ = 0, slice_processing_relative_power_threshold=s['slice_processing_relative_power_threshold'], verboseQ=verboseQ)
    
    # plot the field
    if showPlotQ or savePlotQ:
        plot_fld_marginalize_t(fld, settings['dgrid'], dt=dt, saveFilename=saveFilenamePrefix+'_prop_xy.png',showPlotQ=showPlotQ,savePlotQ=savePlotQ) # plot the imported field
        if fld.shape[0] > 1:
            plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-2, saveFilename=saveFilenamePrefix+'_prop_tx.png',showPlotQ=showPlotQ,savePlotQ=savePlotQ) # plot the imported field
            plot_fld_slice(fld, settings['dgrid'], dt=dt, slice=-1, saveFilename=saveFilenamePrefix+'_prop_ty.png',showPlotQ=showPlotQ,savePlotQ=savePlotQ) # plot the imported field
            #plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_prop_t.png',showPlotQ=showPlotQ,savePlotQ=savePlotQ) # plot the imported field
    
    ## write field to disk
    if settings['readfilename'] != None and settings['writefilename'] != None:
        try:
            print('Writing to',settings['writefilename'])
            #writefilename = readfilename + 'r'
            write_dfl(settings['writefilename'], fld, verboseQ=verboseQ)
        except:
            print('ERROR: Could not write field to file',settings['writefilename'])
    
if __name__ == '__main__':
    main()
