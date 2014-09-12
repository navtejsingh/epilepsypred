"""
    Epilepsy Seizure Prediction
    ===========================

Classification problem to make prediction model for epilepsy seizure
using EEG recordings.

"""
from glob import glob
import numpy as np
import scipy.io as _scio
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pylab as plt

DATA_POINT_LEN = 239766


def readdata(args):
    """
    Routine to read input MATLAB matrix as store in memory.
    
    Parameters
    ----------
    channel_num : int
        EEG channel number
        
    args : strings
        Name of MATLAB matrices
    
    Returns
    -------
    eeg_records : numpy array
        EEG records in [n_samples, n_features]
    """
    # Iterate through all the EEG matrices
    nfiles = len(args)
    eeg_records = np.zeros((16*nfiles, DATA_POINT_LEN))
    
    start, stop = 0, 16
    for matname in args:
        try:
            data = _scio.loadmat(matname)
        except IOError:
            raise IOError("Error opening MATLAB matrix " + matname)
            
        # Get the length of data for each channel
        data_key = ""
        for key in data.iterkeys():
            if type(data[key]) == np.ndarray:
                data_key = key  

        # Copy data for the channel and return it back
        eeg_records[start:stop, 0:DATA_POINT_LEN] = np.copy(data[data_key]["data"][0,0])
        del data
        start, stop = stop, stop + 16 
        
    #print eeg_records.shape
    return eeg_records


def model(X, y):
    """
    Classification model to classify EEG recordings.
    
    Parameters
    ----------
    X : numpy array
        X data points with features
        
    y : numpy array
        Y values. 1 for Preictal and 0 for Interictal
    
    Returns
    -------
    
    """
    clf = RandomForestClassifier(n_estimators = 10, n_jobs = -1)
    clf = clf.fit(X, y)

    return clf


if __name__ == "__main__":
    preictal_files = glob("../Dog_1/*preictal_segment_000*.mat")
    preictal_records = readdata(preictal_files)
    
    interictal_files = glob("../Dog_1/*interictal_segment_000*.mat")
    interictal_records = readdata(interictal_files)    
    
    X = np.vstack((preictal_records, interictal_records))
    y = np.concatenate((np.ones(preictal_records.shape[0]), np.zeros(interictal_records.shape[0])))
    
    # Random Forest Classification
    clf = RandomForestClassifier(n_estimators = 10, n_jobs = -1)
    scores = cross_validation.cross_val_score(clf, X, y, cv = 5)
    
    print("Random Forest Classification Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    # Logistic regression classification
    #clf = LogisticRegression()
    #scores = cross_validation.cross_val_score(clf, X, y, cv = 5)
    
    print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    #time = np.linspace(0.0,10.0,DATA_POINT_LEN)
    #fig = plt.figure()
    #plt.xlabel("Time (min)")
    #plt.ylabel("Signal (arbitrary unit)")
    #plt.title("EEG Recording Signal")
    #plt.plot(time, preictal_records[10,:], label = "Preictal Signal")
    #plt.plot(time, interictal_records[10,:], label = "Interictal Signal")
    #plt.legend()
    #plt.show()
