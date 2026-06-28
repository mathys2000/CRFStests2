# Module: myLTE.py
# My LTE Functions
# 5-20-24

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.signal.windows as ssw
import json
rng = np.random.default_rng()


def filtFIR(xt, ht, Fs):
    """
    Delay compensated pseudo CT FIR filter with impulse response ht
    """
    N = len(ht)-1   # filter order
    N2 = int(round(N/2.0))   # half of filter length
    yt = ss.lfilter(ht, 1, np.hstack((xt, np.zeros(N2))))/float(Fs)
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


def SpecGram(xt, Fs, Nfft, Navg, W='boxcar', yscale='dB', normalize=True):
    """
    Create spectrogram image from IQ samples in xt, sampled at rate Fs.
    Nfft is the DFT/FFT blocklength used for time to frequency domain conversion.
    Navg is the number of DFT/FFT blocks over which the average is taken
    for each output block.
    W is the time-domain window used before each DFT/FFT
    yscale='lin' outputs linearly scaled PSD values
    yscale='dB' outputs PSD values in dB
    If normalize=True, the spectrogram magnitudes are normalized (in 0...1 range)
    Output: ff (frequency axis), and
            Sximg (spectrogram image, y-axis is frequency, x-axis is time)
    
    Output of computePSDw is (NumB x Nfft) matrix Sximg
    To average over Navg blocks, use Avg @ Sxf:
    Example: Nfft=4, Navg=2, NumB=6, (len(xt)=24), Nout=3
                    [S00 S01 S02 S03]
    [1 1 0 0 0 0]   [S10 S11 S12 S13]
    [0 0 1 1 0 0] @ [S20 S21 S22 S23]
    [0 0 0 0 1 1]   [S30 S31 S32 S33]
                    [S40 S41 S42 S43]
                    [S50 S51 S52 S53]
    """
    Nout = int(np.floor(len(xt)/(Nfft*Navg)))    # number of (averaged) output blocks
    NumB = Nout*Navg                             # number of FFT blocks
    xt = xt[:int(Nout*Nfft*Navg)]                # crop IQ samples to multiple of Nfft*Navg
    ff, Sxf, Sxavgf = computePSDw(xt, Fs, 0, Nfft, 0, W, fftbins=True)   # get PSD matrix
    Avg = np.zeros((1,Nout*NumB), int)           # prepare Avg matrix for averaging
    for i in range(Nout):
        ix = i*(NumB + Navg)
        Avg[0,ix:ix+Navg] = np.ones((1,Navg))
    Avg = np.reshape(Avg,(Nout,-1))/Navg         # average over Navg blocks
    Sximg = Sxf.T @ Avg.T                        # transposed for freq on y-axis in spectrogram image  
    if yscale=='dB':                             # linear PSD values if yscale != 'dB'
        Sximg = 10*np.log10(Sximg)
    if normalize:
        Sxmin = np.min(np.min(Sximg))
        Sxmax = np.max(np.max(Sximg))
        Sximg = (Sximg - Sxmin)/(Sxmax - Sxmin)     # normalized magnitudes (dB or linear)
    return ff[0::Navg], Sximg
    
    
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
	
	
def getBWcfg(BW):
    """
    Get LTE bandwidth configuration parameters for bandwidth BW
    BW in ['1.4MHz','3MHz','5MHz','10MHz','15MHz','20MHz']
    """
    BW20  = {'Fs_MHz':30.72, 'NFFT':2048, 'NRB':100, 'NSC':1200, 'CP0':160, 'CP1p':144}
    BW15  = {'Fs_MHz':23.04, 'NFFT':1536, 'NRB': 75, 'NSC': 900, 'CP0':120, 'CP1p':108}
    BW10  = {'Fs_MHz':15.36, 'NFFT':1024, 'NRB': 50, 'NSC': 600, 'CP0': 80, 'CP1p': 72}
    BW5   = {'Fs_MHz': 7.68, 'NFFT': 512, 'NRB': 25, 'NSC': 300, 'CP0': 40, 'CP1p': 36}
    BW3   = {'Fs_MHz': 3.84, 'NFFT': 256, 'NRB': 15, 'NSC': 180, 'CP0': 20, 'CP1p': 18}
    BW1_4 = {'Fs_MHz': 1.92, 'NFFT': 128, 'NRB':  6, 'NSC':  72, 'CP0': 10, 'CP1p':  9} 
    BWsel = {'1.4MHz': BW1_4, '3MHz': BW3, '5MHz': BW5,
             '10MHz': BW10, '15MHz': BW15, '20MHz': BW20}
    return BWsel[BW]


def addPrePostamble(tt_in, yt_in, Fs, t_pre=0.1e-3, t_post=0.2e-3, A_noise=1e-4):
    """
    Insert t_pre sec of preamble and t_post sec of postamble Normal noise with
    amplitude A_noise to (complex-valued) waveform yt
    Returns: tt_out, yt_out
    """
    # Insert 0.1 ms of preamble and 0.2 ms of postamble (noise)
    N_pre = int(np.round(t_pre*Fs))
    nt_pre = rng.standard_normal(N_pre)+1j*rng.standard_normal(N_pre)  # Gaussian noise
    N_post = int(np.round(t_post*Fs))
    nt_post = rng.standard_normal(N_post)+1j*rng.standard_normal(N_post)  # Gaussian noise
    yt_out = np.hstack((A_noise*nt_pre, yt_in, A_noise*nt_post))
    tt_out = np.arange(yt_out.size)/float(Fs) - t_pre
    return tt_out, yt_out


def SigDet(yt, Fs, LParms=[50,10,0.2], thr=[0.7,0.5,10], gsp=100, brw=20, comp=[-8000,8000]):
    """
    IQ signal (yt) power detection to find start and end indexes of signals with power above
    some threshold
    yt: input signal, complex-valued IQ samples
    Fs: sampling rate of yt
    LParms=[DS,k,alfa]: lowpass filter parameters
    DS: downsampling factor
    k, alfa: impulse response length and excess BW, repectively
    thr=[high,low,Nmx]: thresholds for power detection, relative to Pmax from Nmx
    thH=high*Pmax: threshold to switch from power off to on
    thL=low*Pmax: threshold to switch from power on to off (low <= high)
    Nmx: number of power maxima to use for computing Pmax
    gsp: guard space required around power peaks
    brw: bounce reject width (number of non-bounce samples required before pwr on/off switch)
    comp=[start,stop] start/stop delay compensation in samples, respectively
    returns:
    ix_onoff=[ix_on;ix_off]: array with signal start/stop indices
    PdecUS: power decisions (1: on, 0: off) for yt signal, upsampled to Fs
    """
    Pyt = np.abs(yt)**2        # signal power
    DS, k, alfa = LParms
    PytLP, ford = trapfilt(Pyt, Fs, Fs/DS, k, alfa)  # filter
    PytLP = PytLP[::DS]        # downsample by DS
    Paux = np.copy(PytLP)      # auxiliary signal power
    APmax = np.zeros(thr[2])
    for i in range(thr[2]):
        ix = np.argmax(Paux)
        APmax[i] = Paux[ix]
        ix1 = ix - gsp
        if ix1<0:
            ix1 = 0
        ix2 = ix + gsp
        if ix2>Paux.size:
            ix2 = Paux.size
        Paux[ix1:ix2] = 0       # clear current Pmax
    Pmax = np.mean(APmax)
    Pavg = np.mean(PytLP)
    thH = thr[0]*Pmax    # threshold to change from high to low
    thL = thr[1]*Pmax    # threshold to change from low to high
    Pdet = np.zeros(PytLP.size, int)     # power detection array
    Pdec = np.zeros(PytLP.size, int)     # power decision array
    flg = 0    # current power decision on/off flag
    for i in range(brw,PytLP.size):
        if flg:
            if PytLP[i]<thL:
                Pdet[i] = 0
                if (np.sum(Pdet[i-brw:i])==0):
                    Pdec[i] = 0 
                    flg = 0
            else:
                Pdet[i] = 1
                Pdec[i] = 1
        else:
            if PytLP[i]>thH:
                Pdet[i] = 1
                if (brw-np.sum(Pdet[i-brw:i])==0):
                    Pdec[i] = 1
                    flg = 1
            else:
                Pdet[i] = 0
                Pdec[i] = 0
    APdec = np.outer(np.ones(DS), Pdec)
    PdecUS = np.reshape(APdec, -1, order='F')
    PdiffUS = np.diff(np.hstack([0, PdecUS]))
    ix_on = np.where(PdiffUS>0)[0]
    ix_on = ix_on + comp[0]
    ix = np.where(ix_on<0)[0]
    if len(ix)>0:
        ix_on[ix] = 0
    ix = np.where(ix_on>Pyt.size)[0]
    if len(ix)>0:
        ix_on[ix] = Pyt.size
    ix_off = np.where(PdiffUS<0)[0]
    if ix_off.size<ix_on.size:
        ix_off = np.hstack([ix_off, Pyt.size])
    ix_off = ix_off + comp[1]
    ix = np.where(ix_off<0)[0]
    if len(ix)>0:
        ix_off[ix] = 0
    ix = np.where(ix_off>Pyt.size)[0]
    if len(ix)>0:
        ix_off[ix] = Pyt.size
    return np.vstack([ix_on, ix_off]), PdecUS 


def SigDet2(yt, Fs, LParms=[50,10,0.2], thr=[0.7,0.5,10], gsp=100, brw=20, comp=[-8000,8000]):
    """
    IQ signal (yt) power detection to find start and end indexes of signals with power above
    some threshold
    yt: input signal, complex-valued IQ samples
    Fs: sampling rate of yt
    LParms=[DS,k,alfa]: lowpass filter parameters
    DS: downsampling factor
    k, alfa: impulse response length and excess BW, repectively
    thr=[high,low,Nmx]: thresholds for power detection, relative to Pmax from Nmx
    thH=high*Pmax: threshold to switch from power off to on
    thL=low*Pmax: threshold to switch from power on to off (low <= high)
    Nmx: number of power maxima to use for computing Pmax
    gsp: guard space required around power peaks
    brw: bounce reject width (number of non-bounce samples required before pwr on/off switch)
    comp=[start,stop] start/stop delay compensation in samples, respectively
    returns:
    ix_onoff=[ix_on;ix_off]: array with signal start/stop indices
    PdecUS: power decisions (1: on, 0: off) for yt signal, upsampled to Fs
    """
    Pyt = np.abs(yt)**2        # signal power
    DS, k, alfa = LParms
    PytLP, ford = trapfilt(Pyt, Fs, Fs/DS, k, alfa)  # filter
    PytLPFs = np.copy(PytLP)
    PytLP = PytLP[::DS]        # downsample by DS
    Paux = np.copy(PytLP)      # auxiliary signal power
    APmax = np.zeros(thr[2])
    for i in range(thr[2]):
        ix = np.argmax(Paux)
        APmax[i] = Paux[ix]
        ix1 = ix - gsp
        if ix1<0:
            ix1 = 0
        ix2 = ix + gsp
        if ix2>Paux.size:
            ix2 = Paux.size
        Paux[ix1:ix2] = 0       # clear current Pmax
    Pmax = np.mean(APmax)
    Pavg = np.mean(PytLP)
    thH = thr[0]*Pmax    # threshold to change from high to low
    thL = thr[1]*Pmax    # threshold to change from low to high
    Pdet = np.zeros(PytLP.size, int)     # power detection array
    Pdec = np.zeros(PytLP.size, int)     # power decision array
    flg = 0    # current power decision on/off flag
    for i in range(brw,PytLP.size):
        if flg:
            if PytLP[i]<thL:
                Pdet[i] = 0
                if (np.sum(Pdet[i-brw:i])==0):
                    Pdec[i] = 0 
                    flg = 0
            else:
                Pdet[i] = 1
                Pdec[i] = 1
        else:
            if PytLP[i]>thH:
                Pdet[i] = 1
                if (brw-np.sum(Pdet[i-brw:i])==0):
                    Pdec[i] = 1
                    flg = 1
            else:
                Pdet[i] = 0
                Pdec[i] = 0
    APdec = np.outer(np.ones(DS), Pdec)
    PdecUS = np.reshape(APdec, -1, order='F')
    PdiffUS = np.diff(np.hstack([0, PdecUS]))
    ix_on = np.where(PdiffUS>0)[0]
    ix_on = ix_on + comp[0]
    ix = np.where(ix_on<0)[0]
    if len(ix)>0:
        ix_on[ix] = 0
    ix = np.where(ix_on>Pyt.size)[0]
    if len(ix)>0:
        ix_on[ix] = Pyt.size
    ix_off = np.where(PdiffUS<0)[0]
    if ix_off.size<ix_on.size:
        ix_off = np.hstack([ix_off, Pyt.size])
    ix_off = ix_off + comp[1]
    ix = np.where(ix_off<0)[0]
    if len(ix)>0:
        ix_off[ix] = 0
    ix = np.where(ix_off>Pyt.size)[0]
    if len(ix)>0:
        ix_off[ix] = Pyt.size
    return np.vstack([ix_on, ix_off]), PdecUS, PytLPFs


def CPcorr(yn, NFFT, CP, Lcorr, ix_range=0, step=1, Ncorr=1, Wcorr=[0,0]):
    """
    Cyclic prefix correlation (CPcorr) for OFDM signals with regular CP length
    yn:       IQ samples of waveform to analyze
    NFFT:     FFT size
    CP = [CP0, CP1]: Cyclic prefix lengths (CP0 for first symbol, CP1 for other symbols
    Lcorr:    Length of yn (in samples) to use
    ix_range: Index range for sliding correlation comb window (default: ix_range=NFFT)
    step:     Step size (in samples) for stepping through ix_range (default: step=1)
    Ncorr:    Number of cyclic prefixes over which correlation is taken (default: Ncorr=1)
    Wcorr = [Wcorr[0], Wcorr[1]]: Correlation window width for CP0 and CP1 (default Wcorr=[CP0,CP1])

  |<-------------------------------------------- Lcorr --------------------------------------->|
  |                |<-------------------------------- Lcorr-NFFT ----------------------------->|
  |                |                                                                           |
  -----+-----+----------+-----+-----+----------+-----+     ...     +-----+----------+-----+-----
       | CP0 |          | CP0 | CP1 |          | CP1 |             | CPx |          | CPx |
  -----+-----+----------+-----+-----+----------+-----+     ...     +-----+----------+-----+-----
             |<---- NFFT ---->|                |     |                              |     |
                   |    |     |                |     |                              |     |
                   -----+-----+----------+-----+-----+----------+-----+     ...     +-----+---
                        | CP0 |          | CP0 | CP1 |          | CP1 |             | CPx |  ...
                   -----+-----+----------+-----+-----+----------+-----+     ...     +-----+---
                   |
                   |                    Ncorr CPs
                    ________________________/|__________________________
                   /                                                    |
                   |Wcorr|                |     |                 |     |
                   |<--->|                |<--->|       ...       |<--->|
                   |<---------------------------Ncorr*(CPi+NFFT) ------------------------>|
                   |     |                |
                  ix    ix+CP0         ix+CP0+NFFT
                   |<-------------------------------- ix_range ---------------------------------->|

    If correlation peak occurs at ix, start of FFT is at ix+CP0
    """
    if Lcorr < yn.size:
        Lcorr = yn.size
    yn = yn[:Lcorr]
    if ix_range==0:
        ix_range = NFFT
    if Wcorr[0]==0:
        Wcorr[0] = CP[0]
    if Wcorr[1]==0:
        Wcorr[1] = CP[1]
    pad = (1+1j)*np.zeros(NFFT)    # padding
    Xyy = np.hstack((yn,pad))*np.conj(np.hstack((pad,yn)))   # correlation product
    Xyy = Xyy[NFFT:Lcorr]
    if ix_range>Xyy.size:
        ix_range = Xyy.size
    comb = np.zeros(Ncorr*(CP[0]+NFFT),int)   # initialize comb window
    Nsym = 0     # current number of symbols in comb window
    ixc = 0      # start index in comb
    CP0pad = np.hstack((np.zeros(int((CP[0]-Wcorr[0])/2),int), np.ones(Wcorr[0],int)))
    CP0pad = np.hstack((CP0pad, np.zeros(CP[0]+NFFT-CP0pad.size,int)))
    CP1pad = np.hstack((np.zeros(int((CP[1]-Wcorr[1])/2),int), np.ones(Wcorr[1],int)))
    CP1pad = np.hstack((CP1pad, np.zeros(CP[1]+NFFT-CP1pad.size,int)))
    while Nsym < Ncorr:
        if Nsym%7==0:
            comb[ixc:ixc+NFFT+CP[0]] = CP0pad
            ixc += (NFFT+CP[0])
        else:
            comb[ixc:ixc+NFFT+CP[1]] = CP1pad
            ixc += (NFFT+CP[1])
        Nsym += 1
    comb = comb[:ixc-NFFT]
    Xyy = np.hstack((Xyy, (1+1j)*np.zeros(comb.size), (1+1j)*np.zeros(ix_range)))
    Acorr = (1+1j)*np.zeros(ix_range)
    for ix in range(0,ix_range,step):
        Acorr[ix] = np.sum(comb*Xyy[ix:ix+comb.size])
    return Acorr     # correlation array (complex valued)


def CorrPeaks(Acorr, thr=[0.7,0.5,2], gsp=20, brw=5, tol=[2,5]):
    """
    Search for and return indexes of peaks in correlation array Acorr
    Acorr: array of correlations (complex-valued)
    thr=[high,low,Nmx]: thresholds for power detection, relative to Pmax from Nmx
    thH=high*Pmax: threshold to switch from power off to on
    thL=low*Pmax: threshold to switch from power on to off (low <= high)
    Nmx: number of power maxima to use for computing Pmax
    gsp: guard space required around power peaks
    brw: bounce reject width (number of non-bounce samples required before pwr on/off switch)
    tol=[brw_tol,peak_tol]: tolerance for bounce reject and peak verification, respectively
    returns:
    absAcorrix: indexes of power peak maxima
    valAcorrix: correlation values (complex-valued) at power peak maxima
    Pdec: power decisions (1: on, 0: off) for Acorr array
    """
    Pcorr = np.abs(Acorr)     # correlation power
    Paux = np.copy(Pcorr)     # auxiliary correlation power
    APmax = np.zeros(thr[2])
    for i in range(thr[2]):
        ix = np.argmax(Paux)
        APmax[i] = Paux[ix]
        ix1 = ix - gsp
        if ix1<0:
            ix1 = 0
        ix2 = ix + gsp
        if ix2>Pcorr.size:
            ix2 = Pcorr.size
        Paux[ix1:ix2] = 0    # clear current Pmax
    Pmax = np.mean(APmax)
    Pavg = np.mean(Pcorr)
    thH = thr[0]*Pmax    # threshold to change from high to low
    thL = thr[1]*Pmax    # threshold to change from low to high
    Pdet = np.zeros(Pcorr.size, int)     # power detection array
    Pdec = np.zeros(Pcorr.size, int)     # power decision array
    flg = 0    # current power decision on/off flag
    for i in range(brw,Pcorr.size):
        if flg:
            if Pcorr[i]<thL:
                Pdet[i] = 0
                if (np.sum(Pdet[i-brw:i])<=tol[0]):
                    Pdec[i] = 0 
                    flg = 0
            else:
                Pdet[i] = 1
                Pdec[i] = 1
        else:
            if Pcorr[i]>thH:
                Pdet[i] = 1
                if (brw-np.sum(Pdet[i-brw:i])<=tol[0]):
                    Pdec[i] = 1
                    flg = 1
            else:
                Pdet[i] = 0
                Pdec[i] = 0
    Pdecc = 1-Pdec        # complement of Pdec
    absAcorrpk = Pdec*Pcorr
    absAcorrix = np.zeros(Pcorr.size)
    ix = 0    # pointer for absAcorrix array
    n = 0     # pointer into absAcorrpk array
    while n<Pcorr.size:
        aux = np.where(Pdec[n:]>0)[0]
        if aux.size>0:
            n1 = n + aux[0]
        else:
            n1 = n
        aux = np.where(Pdecc[n1:]>0)[0]
        if aux.size>0:
            n2 = n1 + aux[0]
        else:
            n2 = n1
        n1gsp = n1 - gsp
        if n1gsp<0:
            n1gsp = 0
        n2gsp = n2 + gsp
        if n2gsp>Pcorr.size:
            n2gsp = Pcorr.size
        if n2>n1:
            if (sum(Pdec[n1gsp:n2gsp])-sum(Pdec[n1:n2]))<=tol[1]:
                absAcorrix[ix] = n1 + np.argmax(absAcorrpk[n1:n2])
                ix += 1
        n = n2gsp
    absAcorrix = np.array(absAcorrix[:ix],int)
    valAcorrix = Acorr[absAcorrix]
    return absAcorrix, valAcorrix, Pdec




