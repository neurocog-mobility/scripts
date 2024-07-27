# author: Ahomagai

from scipy.signal import butter, filtfilt
def butter_filter(data, fs, cutfreq, order, btype):
    """ Filters data using a Butterworth filter. The nature of the filter can be specified in the parameters of the funciton.

    Parameters
    ----------
    data : ndarray
        The data to be filtered.
    fs : int
        The sampling frequency of the data.
    cutfreq : float
        The cutoff frequency of the filter.
    order : int
        The order of the filter.
    btype : str
        The type of filter to be applied. Can be 'low' or 'high'.

    Returns
    -------
    ndarray
        The filtered data as a one-dimensional array.
     
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutfreq / nyquist

    b,a = butter(order, normalized_cutoff, btype=btype, analog=False)

    return filtfilt(b, a, data)
