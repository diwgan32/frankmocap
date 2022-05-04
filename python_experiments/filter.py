import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import core_constants

def nanHelper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def hampelFilter(arr, windowSize):
    n = arr.size
    k = 1.4826  # scale factor for Gaussian distribution
    indices = []

    for i in range((windowSize), (n - windowSize)):
        data = arr[(i - windowSize):(i + windowSize)]
        #print(i - windowSize, i + windowSize)
        c = np.count_nonzero(np.isnan(data))
        #print(data, arr[i], c)
        
        if (c >= (2 * windowSize) - 1):
            arr[i] = np.nan

    for i in range((windowSize), (n - windowSize)):
        if (np.all(np.isnan(arr[(i - windowSize):(i + windowSize)])) or np.isnan(arr[i])):
            continue

        x0 = np.median(arr[(i - windowSize):(i + windowSize)])
        S0 = k * np.nanmedian(np.abs(arr[(i - windowSize):(i + windowSize)] - x0))
        
        if (np.abs(arr[i] - x0) > 2 * S0):
            arr[i] = x0
            indices.append(i)
    return arr, indices


def bFillNan(arr):
    """ Backward-fill NaNs """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), mask.shape[0] - 1)
    idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
    out = arr[idx]
    return out


def calcMask(arr, maxgap):
    """ Mask NaN gaps longer than `maxgap` """
    isnan = np.isnan(arr)
    cumsum = np.cumsum(isnan).astype('float')
    diff = np.zeros_like(arr)
    diff[~isnan] = np.diff(cumsum[~isnan], prepend=0)
    diff[isnan] = np.nan
    diff = bFillNan(diff)
    return (diff < maxgap) | ~isnan


def movingAverageWithNans(a, n):
    ret = np.cumsum(a.filled(0))
    ret[n:] = ret[n:] - ret[:-n]
    counts = np.cumsum(~a.mask)
    counts[n:] = counts[n:] - counts[:-n]
    ret[~a.mask] /= counts[~a.mask]
    ret[a.mask] = np.nan

    return ret


def smoothArray(a, n, default_value=0):
    # Get locations of all stretches of NaNs longer than core_constants.MAX_NAN_WINDOW
    a = a.astype(np.float32)
    mask = calcMask(a, core_constants.MAX_NAN_WINDOW)
    # Discard outliers
    a, indices = hampelFilter(a, 3)
    # Interpolate the nans that are smaller than core_constants.MAX_NAN_WINDOW
    masked = np.where(mask, a, np.zeros_like(a))
    nans, x = nanHelper(masked)
    masked[nans] = np.interp(x(nans), x(~nans), masked[~nans])
    # Remember to put back the NaNs that are longer than core_constants.MAX_NAN_WINDOW
    masked = np.where(mask, masked, a)
    mx = np.ma.masked_array(masked, np.isnan(masked))
    # Apply moving average filter on top of that
    filter_1 = movingAverageWithNans(mx, n)
    filter_2_array = np.ma.masked_array(filter_1, np.isnan(filter_1))
    
    return movingAverageWithNans(filter_2_array, n)

def min_bucket(arr, n):
    """Externally used function that loops through an array and computes the min 
    of each consecutive `n` elements and stores that min in a new array

    Input:
        - arr, the input array
        - n, the size of each min-bucket
    Output:
        - arr: the input array bucketed with mins
    Example:
        >>> # Min bucketing a RULA score
        >>> min_arr = min_bucket(arr, 4)
    
        In the above example, an array like [1, 2, 3, 4, 2, 5, 6, 7] would become 
        [1, 2]
    """

    # If already right size then padding not necessary
    if (arr.size % n == 0):
        return np.nanmin(arr.reshape(-1, n), axis=1)
    return np.nanmin(np.pad(arr, (0, n - arr.size % n), mode='constant', constant_values=np.nan).reshape(-1, n), axis=1)

if __name__ == "__main__":
    a = np.zeros(338, dtype=np.int64)
    m = smoothArray(a, 3)
    print (m)
