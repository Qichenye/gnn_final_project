import numpy as np
from scipy.signal import decimate

from utils import *
from detect import *
from align import *


class Transformation():
    """
    Class of transformations, which modify an EEG recording.
    """
    def __init__(self):
        pass


class Differences(Transformation):
    """
    Class of nth differences transformations.
    """
    def __init__(self, n=0):
        """
        Create nth differences transformation.

        Arguments:
        n - the order of differences
        """
        self.n = n

    def __repr__(self):
        return f"{self.n}. differences transformation"

    def set_order(self, n):
        self.n = n

    def __call__(self, eeg):
        """
        Apply nth differences transformation.

        Arguments:
        eeg - a timeseries with dimension (n_timepoints)

        Return:
        a transformed timeseries with dimension (n_timepoints - self.n)
        """
        if self.n <= 0:
            return eeg
        else:
            return np.diff(eeg, self.n)


class DownsampleTime(Transformation):
    """
    Class of downsampling in time, which reduces the number
    of points in an eeg timeseries.
    """
    def __init__(self, factor=0, own=False, n=None, 
                 ftype="iir", axis=-1, zero_phase=True):
        """
        Create time downsampling transformation.

        Arguments:
        factor - downsampling factor
        own    - whether to use own function or scipy
        n, ftype, axis, zero_phase - arguments of scipy function
        """
        self.factor = factor
        self.own = own
        self.n = n
        self.ftype = ftype
        self.axis = axis
        self.zero_phase = zero_phase

    def __repr__(self):
        result = f"downsampling in time with factor {self.factor}"
        if self.own:
            result = result + " using own function"
        else:
            result = result + " using scipy.signal.decimate"

        return result

    def set_factor(self, factor):
        self.factor = factor

    def __call__(self, eeg):
        """
        Apply downsampling in time.

        Arguments:
        eeg - a timeseries with dimension (n_timepoints)

        Return:
        a transformed timeseries with dimension (n_timepoints / self.factor)
        """
        if self.factor <= 0:
            return eeg

        if self.own:
            # Implement if needed
            return None
        else:
            return decimate(eeg, self.factor, n=self.n, ftype=self.ftype,
                            axis=self.axis, zero_phase=self.zero_phase)


class DownsampleSpace(Transformation):
    """
    Class of downsampling in space, which reduces
    the precision of timeseries data and ensures
    that it has an integer datatype.
    """
    def __init__(self, n_bits=64, max_val = None,min_val = None):
        """
        Create space downsampling transformation.

        Arguments:
        n_bits - the number of bits for the new representation
        """
        self.n_bits = n_bits
        self.max_val = max_val
        self.min_val = min_val

    def __repr__(self):
        return f"downsampling in space to precision of {self.n_bits} bits"

    def set_n_bits(self, n_bits):
        self.n_bits = n_bits

    def __call__(self, eeg):
        """
        Apply downsampling in space.

        Arguments:
        eeg - a timeseries with dimension (n_timepoints)

        Return:
        a transformed timeseries with dimension (n_timepoints)
        """
        if self.min_val is None:
            abs_max = self.max_val.astype(np.float64) if self.max_val != None else np.max(np.abs(eeg)).astype(np.float64)
            if abs_max == 0:
                return eeg.astype(np.int64)
            scale = (2 ** (self.n_bits - 1) - 1) / abs_max
            result = np.round(eeg * scale).astype(np.int64)
        else: 
            #Rescaling with minimum and maximum values
            max_val = self.max_val
            min_val = self.min_val
            scale = (max_val-min_val)/((2<<self.n_bits- 1)-1)
            result = np.round(eeg/scale).astype(np.int64)
        return result

class DownsampleSigmoidSpace(Transformation):
    """
    Class of downsampling in space, which reduces
    the precision of timeseries data and ensures
    that it has an integer datatype.
    """
    def __init__(self, n_bits=16, slowdown=2000, max_val = None, min_val = None):
        """
        Create space downsampling transformation.

        Arguments:
        n_bits - the number of bits for the new representation
        """
        self.n_bits = n_bits
        self.max_val = max_val
        self.min_val = min_val
        self.slowdown = slowdown

    def __repr__(self):
        return f"logarithmic downsampling in space to precision of {self.n_bits} bits"

    def set_n_bits(self, n_bits):
        self.n_bits = n_bits

    def __call__(self, eeg):
        """
        Apply downsampling in space.

        Arguments:
        eeg - a timeseries with dimension (n_timepoints)

        Return:
        a transformed timeseries with dimension (n_timepoints)
        """
        return np.floor(2 ** (self.n_bits) / (1+np.e ** (-eeg / self.slowdown)) - 2 ** (self.n_bits-1)) # np.round produces -2^(n_bits-1) to 2^(n_bits-1)

                
class TemporalCoding():
    """
    Class of temporal coding transformations,
    which generate a TNN-ready representation.
    """
    def __init__(self, shape="2d", negate=True, onoff=False, 
                 n_bits=64, width=0, mode="flat"):
        """
        Create a temporal coding object.

        Arguments:
        shape  - "1d" for no change or "2d" for flattened multiple-n-hot
        negate - whether to transform to temporal represntation here
                 or not (leaving intermediate representation for k-means), 
                 default True
        onoff  - whether to append negation of original signal in the 1d case
                 before negate is applied, default False
        n_bits - the number of bits for the 2d representation
        width  - if width is k, then 2d representation is multiple-(2k+1)-hot
        mode   - "flat" or "triangle"
        """
        self.shape = shape
        self.negate = negate
        self.onoff = onoff
        self.n_bits = n_bits
        self.width = width
        self.mode = mode

    def __repr__(self):
        if self.shape == "1d":
            result = "timeseries"
            if self.negate:
                result = result + " where high signals are early,"
            else:
                result = result + " where high signals are late,"

            if self.onoff:
                result = result + " with on/off"
            else:
                result = result + " without on/off"
        elif self.shape == "2d":
            dim = 2 ** self.n_bits + 2 * self.width
            result = f"flattened 2d representation with dimension ({dim})"

        return result

    def update_params(self, shape=None, negate=None, onoff=None, 
                      n_bits=None, width=None, mode=None):
        if shape is not None:
            self.shape = shape
        if negate is not None:
            self.negate = negate
        if onoff is not None:
            self.onoff = onoff
        if n_bits is not None:
            self.n_bits = n_bits
        if width is not None:
            self.width = width
        if mode is not None:
            self.mode = mode

    def encode(self, windows):
        if windows.ndim == 1:
            if self.shape == "1d":
                if self.onoff:
                    #result = get_onoff(windows, maximum=2 ** (self.n_bits - 1) - 1)
                    result = get_onoff(windows, maximum=None)
                else:
                    result = windows

                if self.negate:
                    result = transform_data(result, nhot=False, max_val=None)
                return result
            elif self.shape == "2d":
                result = timeseries_to_multiple_nhot(windows, 2 ** self.n_bits, 
                                                     self.width, self.mode)
                if self.negate:
                    result = transform_data(result, nhot=True, max_val=1)
                return result
        elif windows.ndim == 2:
            return np.array([self.encode(window) for window in windows])

    def __call__(self, windows):
        """
        Apply temporal coding.

        Arguments:
        windows - an array of windows containing spikes, with dimensions
                  (n_windows, window_size)

        Return:
        -if self.shape is "2d", then an array with dimensions 
        (n_windows, [2**self.n_bits+2*self.width]*window_size)
        -if self.shape is "1d", and self.onoff is False, then
        an array with dimensions (n_windows, window_size)
        -if self.shape is "1d", and self.onoff is True, then
        an array with dimensions (n_windows, 2*window_size)
        """
        return self.encode(windows)
