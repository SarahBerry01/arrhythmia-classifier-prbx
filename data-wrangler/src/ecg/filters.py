import scipy.signal


def bandpass(sig, low, high, fs):
    nyq = 0.5 * fs
    high = high / nyq
    low = low / nyq
    b, a = scipy.signal.butter(1, [low, high], btype='bandpass')
    y = scipy.signal.filtfilt(b, a, sig)
    return y


def apply_filters(sig):
    return bandpass(sig, 0.5, 45, 360)
