import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy import io as sio


def interpft(x,ny,dim=0):                                                         #input torc array, new sampling, which dimension to interpolate on
    '''
    Function to interpolate using FT method, based on matlab interpft()
    :param x: array for interpolation
    :param ny: length of returned vector post-interpolation
    :param dim: performs interpolation along dimension DIM, default 0
    :return: interpolated data
    '''

    if dim >= 1:                                                                   #if interpolating along columns, dim = 1
        x = np.swapaxes(x,0,dim)                                                   #temporarily swap axes so calculations are universal regardless of dim
    if len(x.shape) == 1:
        x = np.expand_dims(x,axis=1)

    siz = x.shape                                                                  #what is the torc size overall
    [m, n] = x.shape                                                               #unique var for each torc dimension

    if ny > m:                                                                     #if you will be increasing samples (should be)
        incr = 1                                                                   #assign this variable - not very useful but doesn't hurt, could be useful

    a = np.fft.fft(x,m,0)                                                          #do FT along rows, shape unaltered
    nyqst = int(np.ceil((m+1)/2))                                                  #nyqst num calculated
    b = np.concatenate((a[0:nyqst,:], np.zeros(shape=(ny-m,n)), a[nyqst:m, :]),0)  #insert a field of zeros to expand dim to new, using nyqst as break point

    if np.remainder(m,2)==0:                                                       #this hasn't come up yet
        b[nyqst,:] = b[nyqst,:]/2                                                  #presumably dealing with remainder
        b[nyqst+ny-m,:] = b[nyqst,:]                                               #somehow

    y = np.fft.irfft(b,b.shape[0],0)                                               #take inverse FT (real) using new dimension generated along dim 0 of b

    #if all(np.isreal(x)):                                                         #checks to make sure everything is real
    #    y = y.real                                                                #it is, don't know when this would come up

    y = y * ny / m                                                                 #necessary conversion...
    #y = y[1:ny:incr,:]                                                            #

    y = np.reshape(y, [y.shape[0],siz[1]])                                         #should preserve shape

    y = np.squeeze(y)

    if dim >= 1:                                                                   #as above, if interpolating along columns
        y = np.swapaxes(y,0,dim)                                                   #swap axes back and y will be correct

    return y                                                                       #returned value



def insteadofbin(resp,binsize,mf=1):
    '''
    ###SUBFUNCTION - insteadofbin()#####################################################
    #Downsample spike histogram from resp (resolution mf) to resolution by binsize(ms)##
    #"Does by sinc-filtering and downsampling instead of binning," whatever that means##
    #function dsum = insteadofbin(resp,binsize,mf);                                   ##

    :param resp: response data (comes from spikeperiod
    :param binsize: calculated size of bins (basep/stimtime = binsize)
    :param mf: multiplication factor, default 1
    :return: returns downsampled spike histogram
    '''

    if len(resp.shape) >= 2:                                                       #added in jackN phase ot account for Dsum input having one dimension in bython (250x1)
        [spikes,records] = resp.shape                                              #break response into its dimensions
    else:
        resp = np.expand_dims(resp, axis=1)
        [spikes,records] = resp.shape

    outlen = int(spikes / binsize / mf)                                            #

    if outlen - np.floor(outlen/1) * 1 > 1:                                        #basically what matlab fxn mod(x,y)
        print('Non-integer # bins. Result may be distorted')                       #check probably for special circumstances

    outlen = np.round(outlen)                                                      #original comments say "round or ceil or floor?" oh well
    dsum = np.zeros([outlen,records])                                              #empty an array to fill below, going to be output

    for rec in range(records):                                                     #going through all records
        temprec = np.fft.fft(resp[:,rec])                                          #fft for each

        if outlen % 2 == 0:                                                        #if even length, create middle point
            temprec[np.ceil((outlen-1)/2)+1] = np.abs(temprec[np.ceil(outlen-1)/2]+1)

        dsum[:, rec] = np.fft.ifft(np.concatenate((temprec[0:int(np.ceil((outlen - 1) / 2) + 1)], np.conj(
            np.flipud(temprec[1:int(np.floor((outlen - 1) / 2) + 1)]))))).real

    return dsum


def makepsth(dsum,fhist,startime,endtime,mf=1):
    '''
    Creates a period histogram according to period implied by input freq fhist
    :param dsum: spike data
    :param fhist: the frequency for which the histogram is performed
    :param startime: the start of the histogram data (ms)
    :param endtime: the end of the histogram data (ms)
    :param mf: multiplication factor
    :return: psth
    '''
    if fhist == 0:
        fhist = 1000/(endtime-startime)
    dsum = dsum[:]

    period = int(1000 * (1/fhist) * mf)                                            # in samples
    startime = startime * mf                                                       #      "
    endtime = endtime * mf                                                         #      "

    if endtime > len(dsum):
        endtime = len(dsum)

    fillmax = int(np.ceil(endtime/period) * period)
    if fillmax > endtime:
        dsum[endtime+1:fillmax] = np.nan
        endtime = fillmax

    dsum[:startime] = np.nan
    repcount = int(fillmax / period)
    dsum = np.reshape(dsum[:endtime],(period,repcount),order='F')

    wrapdata = np.nansum(dsum,1)                                                  #get 250 list of how many 1s there were
    cnorm =  np.sum(np.logical_not(np.isnan(dsum)),1)                             #get 250 list of how many 1s were possible

    return wrapdata,cnorm

    ##There's extra code that SVD 'hacked' to allow for includsion of the first TORC


def torcmaker(TORC,Params):
    '''
    Returns the TORC - option to plot each torc commented out
    This is a fairly core calculation
    :param TORC: TORC data from torc dictionary
    :param Params: TorcObject containing info about the torc
    :return: the torc itself
    '''
    lfreq = TORC['LowestFrequency']
    hfreq = TORC['HighestFrequency']
    Scales = TORC['Scales']
    Amplitude = TORC['RippleAmplitude']
    Phase = TORC['Phase']
    Rate = TORC['Rates']

    octaves = np.log2(hfreq)-np.log2(lfreq)
    normed_scales = [s*octaves for s in Scales]
    cycles_per_sec = 1000/Params['T']
    normed_tempmod= [t/cycles_per_sec for t in Rate]
    numcomp = Params['numcomp']
    leng = Params['leng']

    # somehow we've figured out that final spectrogram should be
    # (numcomp spectral dimension rows) X (leng time samples per TORC cycle)

    stimHolder = np.zeros((numcomp,leng),dtype=complex)
    c = [np.floor(numcomp/2), np.floor(leng/2)]

    for i, (vel,phs,amp,scl) in enumerate(zip(normed_tempmod,Phase,Amplitude,normed_scales)):
        #print("ripple {}: vel={}, phi={}, amp={}, scl={}".format(i,vel,phs,amp,scl))

        # figure out index in fourier domain for each ripple
        v1=int(vel+c[1])
        v2=int(c[1]-vel)
        s1=int(scl+c[0])
        s2=int(c[0]-scl)

        stimHolder[s1,v1] = (amp/2)*np.exp(1j*(phs-90)*np.pi/180)
        stimHolder[s2,v2] = (amp/2)*np.exp(-1j*(phs-90)*np.pi/180)
        #print("ripple {}: stimHolder[s1,v1]={}, stimholder[s2,v2]={}".format(i,stimHolder[s1,v1],stimHolder[s2,v2]))
        #######if you want to look at your ripple at any point
        #plt.figure()
        #plt.imshow(np.abs(stimHolder))

    y_sum = (np.fft.ifft2(np.fft.ifftshift(stimHolder*(leng*numcomp)))).real
    return y_sum

