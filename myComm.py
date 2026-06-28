# Module: myComm.py
# My Communication Functions
# 4-26-22

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.signal.windows as ssw
import json
rng = np.random.default_rng()


def computeFT(tt,xt,thr=1e-6,deg=1,swapf=1):
    """
    Compute FFT approximation to FT
    with frequency and time shift/swap
    and with threshold for phase suppression
    returns ff [Hz], absXf, argXf [rad] or [deg] for deg=0 or 1
    returns |Xf| in dB with lower limit thr if thr<0
    """
    N = tt.size
    Fs = (N-1)/(tt[-1]-tt[0])   # extract sampling rate
    ixp = np.where(tt>=0)
    ixn = np.where(tt<0)
    xt = np.hstack((xt[ixp], xt[ixn]))   # swap pos/neg time axes
    Df = Fs/float(N)    # frequency resolution
    ff = Df*np.arange(N)   # frequency axis
    Xf = np.fft.fft(xt)/float(Fs)
    if swapf:           # swap pos/neg frequency axes
        ixp = np.where(ff<Fs/2.0)
        ixn = np.where(ff>=Fs/2.0)
        ff = np.hstack((ff[ixn]-Fs, ff[ixp]))
        Xf = np.hstack((Xf[ixn], Xf[ixp]))
    absXf = np.abs(Xf)
    argXf = np.zeros(Xf.size)
    if thr<0:
        linthr = 10**(thr/20.0)  # linear lower threshold
        absXfdB = linthr*np.ones(absXf.size)
        ix = np.where(absXf>=linthr)  # indexes where absXf>=linthr
        absXfdB[ix] = absXf[ix]
        absXf = 20*np.log10(absXfdB)
    else:
        ix = np.where(absXf>=thr)
    argXf[ix] = np.angle(Xf[ix])
    if deg:
        argXf = 180/np.pi*argXf
    return ff, absXf, argXf


def computePSDw(xt, Fs, thr, NB, ovlp=0, W='boxcar', fftbins=True):
    """
    DFT/FFT approximation to power spectral density of x(t)
    with blocklength NB, overlap ovlp [0...1) and window function W
    fftbins=True for "periodic" window (for use with DFT/FFT)
    fftbins=False for symmetric window (for filter design)
    thr<0 => compute in dB with lower limit thr
    thr=0 => linear display
    Window W one of (see scipy.signal.windows.get_window)
        'boxcar'
        'triang'
        'blackman'
        'hamming'
        'hann'
        'bartlett'
        'flattop'
        'kaiser' (needs beta), e.g., ('kaiser',2.0)
        'gaussian' (needs standard deviation)
        and more
    """
    NB = int(min(NB, len(xt)))
    Novlp = round(ovlp*NB)  # length of overlap
    Nadv = NB-Novlp         # length of advance
    NN = int(np.floor((xt.size-Novlp)/float(Nadv)))  # number of blocks
    if np.any(np.iscomplex(xt)):
        xNN = (1+1j)*np.zeros((NN, NB))
    else:
        xNN = np.zeros((NN, NB))      
    for i in range(NN):
        xNN[i] = xt[i*Nadv:i*Nadv+NB]
    ww = ssw.get_window(W, NB, fftbins)  # window function 
    Pw = np.mean(np.power(ww, 2.0))  # window avg power
    xNN = ww*xNN   # windowed time sequence array   
    Sxf = np.power(np.abs(np.fft.fft(xNN)), 2)/float(NB*Fs)/Pw
    ff = Fs*np.array(np.arange(NB),np.int64)/float(NB)  # frequ axis
    ixp = np.where(ff<Fs/2.0)[0]
    ixn = np.where(ff>=Fs/2.0)[0]
    ff = np.hstack((ff[ixn]-Fs, ff[ixp]))  # new freq axis
    Sxf = np.hstack((Sxf[:,ixn], Sxf[:,ixp]))
    Sxavgf = np.mean(Sxf, axis=0)
    if thr<0:
        linthr = 10**(thr/10)
        ix = np.where(Sxf<linthr)
        Sxf[ix] = linthr
        Sxf = 10*np.log10(Sxf)
        ix = np.where(Sxavgf<linthr)
        Sxavgf[ix] = linthr
        Sxavgf = 10*np.log10(Sxavgf)
    return ff, Sxf, Sxavgf          
    

def psd4mplot(ff, Sxf, ax=None, ylbl='dBm', xlbl='Hz', lgnd=True):
    """
    Add a 4M (min,max,mean,median) PSD axes to the current plot.
    ff: frequency axis
    Sxf: array of PSD blocks
    """
    if ax is None:
        ax = plt.gca()
    Sxfsample = Sxf[rng.integers(np.shape(Sxf)[0])]
    Sxfsample = np.ndarray.flatten(Sxfsample)
    Sxfmax = np.max(Sxf, axis=0)
    Sxfmean = np.mean(Sxf, axis=0)
    Sxfmedian = np.median(Sxf, axis=0)
    Sxfmin = np.min(Sxf, axis=0)
    line = ax.plot(ff, 10*np.log10(Sxfsample), ':y', label='sample')
    line = ax.plot(ff, 10*np.log10(Sxfmax), '-r', label='max')
    line = ax.plot(ff, 10*np.log10(Sxfmean), '-b', label='mean')
    line = ax.plot(ff, 10*np.log10(Sxfmedian), '-g', label='median')
    line = ax.plot(ff, 10*np.log10(Sxfmin), '-k', label='min')
    ax.set_ylabel(ylbl)
    ax.set_xlabel(xlbl)
    ax.grid()
    if lgnd:
        ax.legend()


def filtFIR(xt, ht, Fs):
    """
    Delay compensated pseudo CT FIR filter with impulse response ht
    """
    N = len(ht)-1   # filter order
    N2 = int(round(N/2.0))   # half of filter length
    yt = ss.lfilter(ht, 1, np.hstack((xt, np.zeros(N2))))/float(Fs)
    return yt[N2:] 
    
    
def filtFIR32(xt, ht, Fs):
    """
    Delay compensated pseudo CT FIR filter with impulse response ht
    32-bit float or 64-bit complex version
    """
    N = len(ht)-1   # filter order
    N2 = int(round(N/2.0))   # half of filter length
    cflag = 0  # complex number flag
    if (np.iscomplexobj(xt)):
        xt = np.array(xt, np.complex64)
        cflag = 1
    else:
    	  xt = np.array(xt, np.float32)
    if (np.iscomplexobj(ht)):
        ht = np.array(ht, np.complex64)
        cflag = 1
    else:
    	  ht = np.array(ht, np.float32)
    if cflag:
        yt = np.array(ss.lfilter(ht,1,
             np.hstack((xt,np.array(np.zeros(N2),np.float32))))/float(Fs),np.complex64)
    else:
        yt = np.array(ss.lfilter(ht,1,
             np.hstack((xt,np.array(np.zeros(N2),np.float32))))/float(Fs),np.float32)
    return yt[N2:] 
    
    
def trapfilt(xt, Fs, fparms, k=10, alfa=0.2, beta=0):
    """
    Delay compensated FIR filter with trapezoidal
    frequency response, (1-alfa)*fL<|f|<(1+alfa)*fL transition
    Cutoff at |t|=k/(2*fL)
    Kaiser window function added.
    fparms = [fL, fc, fshift]
    fshift = 'cos' or 'exp'  for bandpass filters
    """
    if (type(fparms)==int or type(fparms)==float):
        fL, fc, fshift = fparms, 0, 'cos'
    elif len(fparms)==1:
        fL, fc, fshift = fparms[0], 0, 'cos'
    elif len(fparms)==2:
        fL, fc, fshift = fparms[0], fparms[1], 'cos'
    else:
        fL, fc, fshift = fparms[0], fparms[1], fparms[2].lower()
    cflag = 0  # complex number flag
    if (np.iscomplexobj(xt) or fshift=='exp'):
        cflag = 1
    ixk = round(Fs*k/(2.0*fL))
    tt = np.arange(-ixk,ixk+1)/float(Fs)
    ht = 2*fL*np.sinc(2*fL*tt)
    if cflag:
        ht = ht + 1j*np.zeros(tt.size)
    N = len(ht)-1   # filter order
    if alfa>0:
        ht = ht*np.sinc(2*alfa*fL*tt)
    if beta>0:
        ht = ht*np.kaiser(N+1,beta)
    if fc != 0:
        if fshift=='exp':
            ht = ht*np.exp(2j*np.pi*fc*tt)
        else:
            ht = ht*np.cos(2*np.pi*fc*tt)
    yt = filtFIR(xt, ht, Fs)
    return yt, N
    
    
def trapfilt32(xt, Fs, fparms, k=10, alfa=0.2, beta=0):
    """
    Delay compensated FIR filter using 32-bit float
    or 64-bit complex number representation with trapezoidal
    frequency response, (1-alfa)*fL<|f|<(1+alfa)*fL transition
    Cutoff at |t|=k/(2*fL)
    Kaiser window function added.
    fparms = [fL, fc, fshift]
    fshift = 'cos' or 'exp'  for bandpass filters
    """
    if (type(fparms)==int or type(fparms)==float):
        fL, fc, fshift = fparms, 0, 'cos'
    elif len(fparms)==1:
        fL, fc, fshift = fparms[0], 0, 'cos'
    elif len(fparms)==2:
        fL, fc, fshift = fparms[0], fparms[1], 'cos'
    else:
        fL, fc, fshift = fparms[0], fparms[1], fparms[2].lower()
    cflag = 0  # complex number flag
    if (np.iscomplexobj(xt) or fshift=='exp'):
        cflag = 1
    ixk = round(Fs*k/(2.0*fL))
    tt = np.array(np.arange(-ixk,ixk+1)/float(Fs), np.float32)
    ht = 2*fL*np.sinc(2*fL*tt)
    if cflag:
        ht = ht + 1j*np.array(np.zeros(tt.size), np.float32)
    N = len(ht)-1   # filter order
    if alfa>0:
        ht = ht*np.sinc(2*alfa*fL*tt)
    if beta>0:
        ht = ht*np.array(np.kaiser(N+1,beta), float32)
    if fc != 0:
        if fshift=='exp':
            ht = ht*np.exp(2j*np.pi*fc*tt)
        else:
            ht = ht*np.cos(2*np.pi*fc*tt)
    yt = filtFIR32(xt, ht, Fs)
    return yt, N
    
    
# apd (amplitude probability distribution) function
"""
[a, p] = apd(s) estimates the amplitude probability distribution function
input parameters:
s = amplitude samples
return variables:
a = ordered amplitudes
p = probability that the ordered amplitude is exceeded
"""
def apd(s):
    if (np.isrealobj(s) and np.min(s)>=0):
        a = np.sort(s)
        N = a.size
        p = 1 - np.arange(N)/float(N)
    else:
        print('Input values must be amplitudes, i.e., real and non-negative.')
    return a, p
    
    
# apdplot function
"""
Generates APD plot axes from (a, p) pairs on Rayleigh graph "paper" where:
a = ordered amplitudes
p = probability that amplitude a is achieved or exceeded
sig2 = sigma^2 of underlying Gaussian I/Q processes
ax = axes for plot
fmt = format for plot (color, markers, etc)
"""
def apdplot(a, p, sig2=1, ax=None, fmt='-b'):
    if ax is None:
        ax = plt.gca()
    # x-axis labels
    pticks = np.array([0.0001,0.01,0.1,1,5,10,20,30,40,50,60,70,80,90,95,98,99])/100.0
    plabels = ['0.0001','0.01','0.1','1','5','10','20','30','40','50','60',
               '70','80','90','95','98','99']
    P0 = pticks[0]   # P(X>x) at origin of graph
    # x-axis Rayleigh scaling
    xxticks = 10*np.log10(-np.log(P0))-10*np.log10(-np.log(pticks))
    ix1 = np.where(p==1)[0]  # avoid log(p)=0
    p[ix1] = 0.9999
    x = 10*np.log10(-np.log(P0))-10*np.log10(-np.log(p))
    x0001 = 10*np.log10(-np.log(P0))-10*np.log10(-np.log(pticks[0]))
    ix0001 = np.where(x>=x0001)[0]
    x99 = 10*np.log10(-np.log(P0))-10*np.log10(-np.log(pticks[-1]))
    ix99 = np.where(x99<=x)[0]
    a[ix99] = a[ix99[-1]]  # set minimum a value
    # map a and plot axes
    y = 20*np.log10(a)-10*np.log10(2*sig2)
    line = ax.plot(x, y, fmt)
    ax.set_ylim([y[ix99[-1]],np.ceil(y[ix0001[-1]])])
    ax.set_xlim([xxticks[0],xxticks[-1]])
    ax.set_xticks(xxticks)
    ax.set_xticklabels(plabels)
    ax.grid()
    # label y axis
    ax.set_ylabel('dBV')  # dB relative to 1 volt
    # label x axis
    ax.set_xlabel('percent exceeding ordinate')
    return        


def parsetar(tar):
    """
    Find .sigmf-meta and .sigmf-data file indexes in .sigmf tar file
    """
    m_files = []   # meta file indexes
    d_files = []   # data file indexes
    for i in range(len(tar.getmembers())):
        if tar.getmembers()[i].name.find('.sigmf-meta') >= 0:
            m_files.append(i)
        if tar.getmembers()[i].name.find('.sigmf-data') >= 0:
            d_files.append(i)
    return m_files, d_files


def parsemeta(meta, datalen):
    """
    Parse metadata and return sample_rate, collection name
    and capture metadata
    """           
    Fs = meta['global']['core:sample_rate']
    coll = meta['global']['core:collection']
    # get capture data from meta file
    capts = meta['captures']   # list of dictionaries
    Ncapts = len(capts)
    capstart = np.zeros(Ncapts)
    capsfreq = np.zeros(Ncapts)
    capsdtim = []
    for i in range(Ncapts):
        capstart[i] = capts[i]['core:sample_start']
        capsfreq[i] = capts[i]['core:frequency']
        capsdtim.append(capts[i]['core:datetime'])
    capslens = np.diff(np.hstack((capstart, datalen))) 
    capstart = np.array(capstart, int)   
    capslens = np.array(capslens, int)   
    return Fs, coll, Ncapts, capstart, capslens, capsfreq, capsdtim
	
	
def parsemeta1(meta, datalen):
    """
    Parse metadata and return sample_rate, collection name
    and capture metadata
    """           
    Fs = meta['global']['core:sample_rate']
    # coll = meta['global']['core:collection']
    # get capture data from meta file
    capts = meta['captures']   # list of dictionaries
    Ncapts = len(capts)
    capstart = np.zeros(Ncapts)
    capsfreq = np.zeros(Ncapts)
    capsdtim = []
    for i in range(Ncapts):
        capstart[i] = capts[i]['core:sample_start']
        capsfreq[i] = capts[i]['core:frequency']
        capsdtim.append(capts[i]['core:datetime'])
    capslens = np.diff(np.hstack((capstart, datalen))) 
    capstart = np.array(capstart, int)   
    capslens = np.array(capslens, int)   
    return Fs, Ncapts, capstart, capslens, capsfreq, capsdtim
	
	
def resamp(y1t,Fs1,US,DS,fL,k=10,alfa=0.2):
    """
    Resample y1t at Fs1 with upsampling factors US and downsampling
    factors DS, e.g., US=[8,6,4], DS=[7,5,5] to convert
    Fs1=14 MHz to Fs2=15.36 MHz
    fL: cutoff frequency of (-6 dB) LPF
    k:  "tail" length (multiples of 1/2fL) of filter response hLt
    alfa: passband to stopband transition (2*alfa*fL)
    """
    y2t = y1t   # initialize
    Fs2 = Fs1
    for i in range(len(US)):
        Fs2 = US[i]*Fs2
        A = (1+1j)*np.zeros((US[i], y2t.size))
        A[0] = y2t   # expand y2t with US[i]-1 zeros inserted
        y2t = np.reshape(A, (1, A.size), order='F').flatten()
        y2t, ford = trapfilt(y2t, Fs2, fL, k, alfa)
        y2t = US[i]*y2t  # amplitude compensation
        Fs2 = Fs2/float(DS[i])
        y2t = y2t[::DS[i]]   # downsample
    return y2t, Fs2
	
	
def resamp32(y1t,Fs1,US,DS,fL,k=10,alfa=0.2):
    """
    Resample y1t at Fs1 with upsampling factors US and downsampling
    factors DS, e.g., US=[8,6,4], DS=[7,5,5] to convert
    Fs1=14 MHz to Fs2=15.36 MHz
    fL: cutoff frequency of (-6 dB) LPF
    k:  "tail" length (multiples of 1/2fL) of filter response hLt
    alfa: passband to stopband transition (2*alfa*fL)
    32-bit float or 64-bit complex version
    """
    y2t = np.array(y1t,np.complex64)    # initialize
    Fs2 = Fs1
    for i in range(len(US)):
        Fs2 = US[i]*Fs2
        A = np.array((1+1j)*np.zeros((US[i], y2t.size)),np.complex64)
        A[0] = y2t   # expand y2t with US[i]-1 zeros inserted
        y2t = np.reshape(A, (1, A.size), order='F').flatten()
        y2t, ford = trapfilt32(y2t, Fs2, fL, k, alfa)
        y2t = US[i]*y2t  # amplitude compensation
        Fs2 = Fs2/float(DS[i])
        y2t = y2t[::DS[i]]   # downsample
    return y2t, Fs2
	
	
def resampIQ(y1t,Fs1,US,DS,fL,k=10,alfa=0.2):
    """
    Resample y1t at Fs1 with upsampling factors US and downsampling
    factors DS, e.g., US=[8,6,4], DS=[7,5,5] to convert
    Fs1=14 MHz to Fs2=15.36 MHz
    fL: cutoff frequency of (-6 dB) LPF
    k:  "tail" length (multiples of 1/2fL) of filter response hLt
    alfa: passband to stopband transition (2*alfa*fL)
    Resample I and Q individually to avoid memory overflow
    """
    y2It = np.real(y1t)   # initialize
    y2Qt = np.imag(y1t)
    Fs2 = Fs1
    for i in range(len(US)):
        Fs2 = US[i]*Fs2
        AI = np.zeros((US[i], y2It.size))
        AQ = np.zeros((US[i], y2Qt.size))
        AI[0] = y2It   # expand y2It with US[i]-1 zeros inserted
        AQ[0] = y2Qt   # expand y2Qt with US[i]-1 zeros inserted
        y2It = np.reshape(AI, (1, AI.size), order='F').flatten()
        y2Qt = np.reshape(AQ, (1, AQ.size), order='F').flatten()
        y2It, ford = trapfilt(y2It, Fs2, fL, k, alfa)
        y2Qt, ford = trapfilt(y2Qt, Fs2, fL, k, alfa)
        y2It = US[i]*y2It  # amplitude compensation
        y2Qt = US[i]*y2Qt  # amplitude compensation
        Fs2 = Fs2/float(DS[i])
        y2It = y2It[::DS[i]]   # downsample
        y2Qt = y2Qt[::DS[i]]   # downsample
    return y2It+1j*y2Qt, Fs2
	
	
	