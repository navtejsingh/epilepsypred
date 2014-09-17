"""
    Extract Features from EEG time series
    -------------------------------------
    Peak in Band (PIB) features are extracted from EEG time series data.
    Six PIB per minute are calculated. This results in 960 features for
    10 minute of EEG recording in 16 channels.
    
    Reference
    ---------
    http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0081920
"""

import numpy as np
from glob import glob
import scipy.io as _scio
import scipy.signal as _scsg
import multiprocessing as mp


def readdata(matfile):
    """
    Routine to read input MATLAB matrix as store in memory.
    
    Parameters
    ----------
    matfile : string
        MATLAB binary file 
            
    Returns
    -------
    Tuple of 
    eeg_record : numpy array
        EEG time series
        
    eeg_record_length_min : int
        EEG time series length in minutes
        
    eeg_sampling_frequency : float
        EEG time series sampling frequency
        
    """
    try:
        data = _scio.loadmat(matfile)
    except IOError:
        raise IOError("Error opening MATLAB matrix " + matfile)
            
    # Get the length of data for each channel
    data_key = ""
    for key in data.iterkeys():
        if type(data[key]) == np.ndarray:
            data_key = key  

    # Copy data for the channel and return it back
    eeg_record = np.copy(data[data_key]["data"][0,0])
    eeg_record_length_min = int(data[data_key]["data_length_sec"]/60.)
    eeg_sampling_frequency = float(data[data_key]["sampling_frequency"])
    del data
        
    return (eeg_record, eeg_record_length_min, eeg_sampling_frequency)


def determine_pib(X, eeg_sampling):
    """
    Calculate power in bands (PIB).
    
    Parameters
    ----------
    X : numpy arrat
        Time series
        
    eeg_sampling : float
        EEG sampling frquency
        
    Returns
    -------
    pib : numpy array
        Power in band (6 elements)
    """
    freq, Pxx = _scsg.welch(X, fs = eeg_sampling, noverlap = None, scaling = "density")

    pib = np.zeros(6)    
            
    # delta band (0.1-4hz)
    ipos = (freq >= 0.1) & (freq < 4.0)
    pib[0] = np.trapz(Pxx[ipos], freq[ipos]) 
    
    # theta band (4-8hz)
    ipos = (freq >= 4.0) & (freq < 8.0)
    pib[1] = np.trapz(Pxx[ipos], freq[ipos]) 
        
    # alpha band (8-12hz) 
    ipos = (freq >= 8.0) & (freq < 12.0)
    pib[2] = np.trapz(Pxx[ipos], freq[ipos]) 
        
    # beta band (12-30hz) 
    ipos = (freq >= 12.0) & (freq < 30.0)
    pib[3] = np.trapz(Pxx[ipos], freq[ipos]) 
        
    # low-gamma band (30-70hz)
    ipos = (freq >= 30.0) & (freq < 70.0)
    pib[4] = np.trapz(Pxx[ipos], freq[ipos]) 
         
    # high-gamma band (70-180hz)
    ipos = (freq >= 70.0) & (freq < 180.0)
    pib[5] = np.trapz(Pxx[ipos], freq[ipos]) 
    
    return pib        
                    

def worker(eegfile):
    """
    Multiprocessing pool worker function. Extracts features from EEG time
    series.
    
    Parameters
    ----------
    eegfile : EEG recording file name
    
    Returns
    -------
    feature_arr : numpy array
       Extracted feature array 
    """
    print "Processing file : ", eegfile
        
    # Read the MATLAB binary file and extract data
    eeg_record, egg_length_min, eeg_sampling = readdata(eegfile)

    n_channels = eeg_record.shape[0]

    feature_arr = np.zeros((1, 6 * n_channels * egg_length_min))
    
    # Extract features. Features are sum of power in power spectrum of
    # time series. Summed power in 6 bands ==> delta(0.1-4hz), theta(4-8hz),
    # alpha(8-12hz),beta(12-30hz), low-gamma(30-70hz), high-gamma(70-180hz)
    # This is done for 1 minute segment of the time series.
    start, stop = 0, 6
    for channel in range(n_channels):
        time_chunks = np.array_split(eeg_record[channel], egg_length_min)
        
        # iterate through the chunks and calculate summed spectral power of 
        # the band. This will give 6 features/1 minute for 1 channel or 96/minute 
        # for 16 channels. This will give us 960 features for 1 dataset
        for chunk in time_chunks:
            feature_arr[0,start:stop] = determine_pib(chunk, eeg_sampling)
            start, stop = stop, stop + 6
    
    return feature_arr
    
            
def main(args):
    """
    Main function
    
    Parameters
    ----------
    args : list
        List of EEG recording file names
        
    feature_arr : numpy array
        Consolidated array of features for the training set
    """
    # Determine number of input files
    nfiles = len(args)
    
    # Create an array to store extracted features
    feature_arr = np.zeros((nfiles, 960))
    
    # Number of cpus
    n_cpus = mp.cpu_count()
    
    # Create a pool of worker functions
    pool = mp.Pool(n_cpus)
    result = pool.map(worker, args)
    
    # Feature array
    for i in range(len(result)):
        nrecs = result[i].shape[1]
        feature_arr[i,:nrecs] = result[i]
        
    del result
    
    return feature_arr
        

if __name__ == "__main__":
    preictal_files = glob("../Dog_1/*preictal_segment*.mat")
    feature_arr_1 = main(preictal_files)
    
    interictal_files = glob("../Dog_1/*interictal_segment*.mat")
    feature_arr_2 = main(interictal_files)
    
    X = np.vstack((feature_arr_1, feature_arr_2))
    y = np.concatenate((np.ones(feature_arr_1.shape[0]), np.zeros(feature_arr_2.shape[0])))
    
    np.savez("../Dog_1/Dog_1_features.npz", X = X, y = y)