import nems_lbhb.baphy as nb
import nems_lbhb.io as nio
from nems import epoch as ep
import os

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy import signal as sgn
import scipy.io as sio
import collections

from nems_lbhb.strf.torc_subfunctions import interpft, insteadofbin, makepsth, torcmaker

####Sample Data- works as test#####
mfilename = "/auto/data/daq/Amanita/AMT005/AMT005c05_p_TOR.m"
cellid = 'AMT005c-12-1' #one being used in Matlab
fs=1000
###########################

def strf_tensor(mfilename,cellid,fs=1000):
    '''
     Creates matrices used as features and labels of tensor
     :param mfilename: File with your data in it
     :param cellid: Name of cell
     :param fs: sampling frequency, default 1000
     :return: 375x750 matrix of stimulus, with time delay & 7500x1 matrix of response
     '''
    rec = nb.baphy_load_recording_file(mfilename=mfilename, cellid=cellid, fs=fs, stim=False)  # fs=1000
    globalparams, exptparams, exptevents = nio.baphy_parm_read(mfilename)
    signal = rec['resp'].rasterize(fs=fs)  # rasterize the signal

    epoch_regex = "^STIM_TORC_.*"  # pick all epochs that have STIM_TORC_...
    epochs_to_extract = ep.epoch_names_matching(signal.epochs, epoch_regex)  # find those epochs
    r = signal.extract_epochs(
        epochs_to_extract)  # extract them, r.keys() yields names of TORCS that can be looked through as dic r['name']...can be np.squeeze(0, np.mean(

    all_arr = list()  # create empty list
    for val in r.values():  # for the 30 TORCs in r.values()
        fval = np.swapaxes(np.squeeze(val), 0, 1)  # create a var to get rid of the third dim (which was a one) and switch the other two axes
        all_arr.append(fval)  # make all_arr have that swap
    stacked = np.stack(all_arr,axis=2)  # rasters                                       #stack the #cell on to make like 'r' from MATLAB (time x sweeps x recordings)

    TorcObject = exptparams["TrialObject"][1]["ReferenceHandle"][1]  # will be strf_core_est input
    PreStimbin = int(TorcObject['PreStimSilence'] * fs)  # how many bins in prestimsilence
    PostStimbin = int(TorcObject['PostStimSilence'] * fs)  # how many bins in poststimsilence
    numbin = stacked.shape[0]  # total bins in total time length
    stacked = stacked[PreStimbin:(numbin - PostStimbin), :, :]  # slice array from first dimensions, bins in pre and post silence, isolate middle 750
    ###


    TorcNames = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Names"]
    RefDuration = TorcObject['Duration']
    mf = int(fs/1000)
    stdur = int(RefDuration*1000)

    ###change nesting to TORCs(StimParam(...))
    TorcParams = dict.fromkeys(TorcNames)                                                    #Create dict from TorcNames
    all_freqs = list()                                                                       #Create empty list of freqs
    all_velos = list()                                                                       #Create empty list of velos
    all_hfreq = list()
    all_lfreq = list()

    for tt, torc in enumerate(TorcNames):                                                    #Number them
        TorcParams[torc] = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Params"][tt + 1]     #insert Params 1-30 to torcs 1-30 now TORCs(Params(...)) nested other way
        freqs = TorcParams[torc]['Scales']                                                   #Add all TORCs' Scales value as var
        velos = TorcParams[torc]['Rates']                                                    #Add all TORCs' Rates value as var
        all_freqs.append(freqs)                                                              #
        all_velos.append(velos)                                                              #
        highestfreqs = TorcParams[torc]['HighestFrequency']
        lowestfreqs = TorcParams[torc]['LowestFrequency']
        all_hfreq.append(highestfreqs)
        all_lfreq.append(lowestfreqs)

    frqs = np.unique(np.concatenate(all_freqs))  # Smoosh to one array and output unique elements
    vels = np.unique(np.concatenate(all_velos))  # temporal spectra
    HighestFrequency = int(np.unique(all_hfreq))
    LowestFrequency = int(np.unique(all_lfreq))
    Octaves = np.log2(HighestFrequency / LowestFrequency)

    Params = dict()
    W = vels                                                                                 #array of ripple velocities
    T = int(np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0]))))))
    Params['T'] = T

    Ompos = [x for x in frqs if x >= 0]                                                      #Get positive frequencies
    Omnegzero = np.flip([x for x in frqs if x <= 0])                                         #just used for populating an array a few lines down

    Omega = np.swapaxes(np.stack((Ompos,Omnegzero)),0,1)                                     #Make an array for main output Omega

    numvels = len(W)                                                                         #
    numfrqs = np.size(Omega,0)                                                               #Used to make empty array to be populated by params
    numstim = len(TorcNames)

    waveParams = np.empty([2,numvels,numfrqs,2,numstim])

    ##This part in MATLAB makes T, octaves, maxv, maxf, saf, numcomp
    basep = int(np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0]))))))
    maxvel = np.max(np.abs(W))
    maxfrq = np.max(np.abs(Omega))
    saf = int(np.ceil(maxvel*2 + 1000/basep))
    numcomp = int(np.ceil(maxfrq*2*Octaves + 1))
    Params['numcomp'] = numcomp

    ##function [ststims,freqs]=stimprofile(waveParams,W,Omega,lfreq,hfreq,numcomp,T,saf);
    [ap,Ws,Omegas,lr,numstim] = waveParams.shape                               #wave params is 5D, define size of each dimension
    [a,b] = Omega.shape                                                        #splitting ripple freqs to matrix nums
    [d] = W.shape                                                              #splitting W into same

    if a*b*d != Omegas*Ws*lr:
        print('Omega and.or W do not match waveParams')

    sffact = saf/1000                                                        #lower sample rate
    leng = int(np.round(T*sffact))                                           #stim duration with lower sampling rounded
    Params['leng'] = leng

    ###this part is important###
    TorcValues = dict()                                                                #make new dict for function output
    for key,value in TorcParams.items():                                               #cycle through with all the different TORC names and respective values
        y_sum = torcmaker(value, Params)                                                #output of function (TORC) assigned to variable
        TorcValues[key] = y_sum                                                        #update dict with the key you are on and the value the function just returned

    ModulationDepth = 0.9                                                              #default set in matlab program
    xSize = int(np.round(10*numcomp/Octaves))                                          #new x sampling rate
    tSize = int(10*saf*basep/1000)                                                     #new t sampling rate
    ScaledTorcs = dict()                                                               #new dictionary for scaled torcs
    for key,value in TorcValues.items():                                               #go through this with all torcs
        [xsiz, tsiz] = value.shape                                                     #basic dims of torc
        temp = value                                                                   #pull out numbers to usable variable
        if xSize != xsiz & tSize != tsiz:                                              #when new sampling rate doesn't equal old vals
            temp1 = interpft(interpft(temp, xSize, 0), tSize, 1)                           #add points, yielding bigger array

            scl = np.max(np.abs([np.min(np.min(temp1)), np.max(np.max(temp1))]))       #largest |value| is scale factor

            temp2 = temp*ModulationDepth/scl                                   #transform original torc values with moddep and calc'd scale
        ScaledTorcs[key] = temp2

    # Create S* 375x25x30, stimulus data
    DelayTorcs = dict()
    for tt, (key, value) in enumerate(ScaledTorcs.items()):
        for jj in range(value.shape[1]):
            rolledtor = np.roll(value, (jj),axis=1)
            if jj == 0:
                torcdel = rolledtor
                #firsttenStim = rolledtor
            else:
                torcdel = np.concatenate((torcdel, rolledtor), axis=0)
                #if jj <= 9:
                    #firsttenStim = np.concatenate((firsttenStim, rolledtor), axis=0)
        DelayTorcs[key] = torcdel
        if tt == 0:
            stimall = torcdel
            #tenStim = firsttenStim
        else:
            stimall = np.concatenate((stimall, torcdel), axis=1)
            #tenStim = np.concatenate((tenStim,firsttenStim), axis=1)

    #stackedtorc = np.stack(list(DelayTorcs.values()), axis=2)

    stimall = np.swapaxes(stimall,0,1)
    #tenStim = np.swapaxes(tenStim,0,1)

    # ####trying Mateo's way, same result as mine
    # vertTorcs = []
    # for dd, (key,value) in enumerate(ScaledTorcs.items()):
    #     laglist = []
    #     for hh in range(value.shape[1]):
    #         rolled2 = np.roll(value,(hh),axis=1)
    #         laglist.append(rolled2)
    #     vertTorcs.append(np.concatenate(laglist,axis=0))
    # stimalltwo = np.concatenate(vertTorcs,axis=1)
    # stimalltwo = np.swapaxes(stimalltwo,0,1)
    # ######

    [stimX,stimT] = temp.shape                                                                   #have var for dims of torc
    binsize = int(basep/stimT)

    if stacked.shape[0] <= fs/4:
        realrepcount = np.max(np.logical_not(np.isnan(stacked[1,:,1])).ravel().nonzero())+1
    else:
        realrepcount = np.max(np.logical_not(np.isnan(stacked[int(np.round(fs/4))+1,:,1])).ravel().nonzero())+1

    FirstStimTime = basep

    # create response data normalized to bins and averaged across repetitions
    RespCat = dict()
    for rep in range(realrepcount):
        for rec in range(numstim):

            spkdata = stacked[:,rep,rec]

            if fs != 1000:
                spkdata = sp.signal.resample(stacked,1000)

            if len(spkdata) < stdur:
                spkdata = np.concatenate((spkdata,np.ones(stdur-len(spkdata))*np.nan),axis=None)

            [Dsum,cnorm] = makepsth(spkdata.copy(), int(1000 / basep), int(FirstStimTime), stdur, mf)

            #Time to normalize by #cycles. May be variable for TORCs, since they could be shorter than length in exptparams
            Dsum = Dsum / (cnorm + (cnorm == 0))

            if binsize > 1/mf:
                Dsum = insteadofbin(Dsum, binsize, mf)
            Dsum = Dsum * (1000/binsize)                                           #normalization by bin size

            if rec == 0:
                respcatval = Dsum
            else:
                respcatval = np.concatenate((respcatval,Dsum),axis=None)

        RespCat[rep] = respcatval

    avgResp = np.mean(np.array(list(RespCat.values())),axis=0)



    return stimall,avgResp