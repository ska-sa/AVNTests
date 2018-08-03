import contextlib
import pandas as pd
import logging
import numpy as np
import os
import pwd
import random
import signal
import subprocess
import time
import warnings
import struct
import socket

from nosekatreport import Aqf

from nose.plugins.attrib import attr

# LOGGER = logging.getLogger(__name__)
LOGGER = logging.getLogger('avn_tests')

from dotenv import load_dotenv
from dotenv import find_dotenv


load_dotenv(find_dotenv())

class Credentials:
    msg = "Check and ensure that your $(pwd)/.env file exists."
    username = str(os.getenv("USERNAME"))
    assert username, msg
    password = str(os.getenv("PASSWORD"))
    assert password, msg
    hostip = str(os.getenv("HOSTIP"))
    assert hostip, msg
    katcpip = str(os.getenv("KATCPIP"))
    assert katcpip, msg


def ip2int(ipstr): return struct.unpack('!I', socket.inet_aton(ipstr))[0]


def int2ip(n): return socket.inet_ntoa(struct.pack('!I', n))


def complexise(input_data):
    """Convert input data shape (X,2) to complex shape (X)
    :param input_data: Xeng_Raw
    """
    return input_data[:, 0] + input_data[:, 1] * 1j


def magnetise(input_data):
    """Convert input data shape (X,2) to complex shape (X) and
       Calculate the absolute value element-wise.
       :param input_data: Xeng_Raw
    """
    id_c = complexise(input_data)
    id_m = np.abs(id_c)
    return id_m


def normalise(input_data):
    # Max range of the integers coming out of VACC
    VACC_FULL_RANGE = float(2**31)
    return input_data / VACC_FULL_RANGE


def normalised_magnitude(input_data):
    return normalise(magnetise(input_data))


def loggerise(data, dynamic_range=70, normalise=False, normalise_to=None):
    with np.errstate(divide='ignore'):
        log_data = 10 * np.log10(data)
    if normalise_to:
        max_log = normalise_to
    else:
        max_log = np.max(log_data)
    min_log_clip = max_log - dynamic_range
    log_data[log_data < min_log_clip] = min_log_clip
    if normalise:
        log_data = np.asarray(log_data) - np.max(log_data)
    return log_data



def disable_warnings_messages():
    """This function disables all error warning messages
    """
    import matplotlib
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
    # Ignoring all warnings raised when casting a complex dtype to a real dtype.
    warnings.simplefilter("ignore", np.ComplexWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    ignored_loggers = [
        "casperfpga", "casperfpga.casperfpga", "casperfpga.bitfield", "casperfpga.katcp_fpg",
        "casperfpga.memory", "casperfpga.register", "casperfpga.transport_katcp",
        "casperfpga.transort_skarab", "corr2.corr_rx", "corr2.fhost_fpga", "corr2.fhost_fpga",
        "corr2.fxcorrelator_engops", "corr2.xhst_fpga", "katcp", "spead2", "tornado.application"
    ]
    # Ignore all loggings except Critical if any
    for logger_name in ignored_loggers:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger('nose.plugins.nosekatreport').setLevel(logging.INFO)


class TestTimeout:
    """
    Test Timeout class using ALARM signal.
    :param: seconds -> Int
    :param: error_message -> Str
    :rtype: None
    """

    class TestTimeoutError(Exception):
        """Custom TestTimeoutError exception"""
        pass

    def __init__(self, seconds=1, error_message='Test Timed-out'):
        self.seconds = seconds
        self.error_message = ''.join(
            [error_message, ' after {} seconds'.format(self.seconds)])

    def handle_timeout(self, signum, frame):
        raise TestTimeout.TestTimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


@contextlib.contextmanager
def RunTestWithTimeout(test_timeout, errmsg='Test Timed-out'):
    """
    Context manager to execute tests with a timeout
    :param: test_timeout : int
    :param: errmsg : str
    :rtype: None
    """
    try:
        with TestTimeout(seconds=test_timeout):
            yield
    except Exception:
        LOGGER.exception(errmsg)
        Aqf.failed(errmsg)
        Aqf.end(traceback=True)


def executed_by():
    """Get who ran the test."""
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
        if user == 'root':
            raise OSError
        Aqf.hop('Test ran by: {} on {} system on {}.\n'.format(user, os.uname()[1].upper(),
                                                               time.strftime("%Y-%m-%d %H:%M:%S")))
    except Exception as e:
        _errmsg = 'Failed to detemine who ran the test with %s' % str(e)
        LOGGER.error(_errmsg)
        Aqf.hop('Test ran by: Jenkins on system {} on {}.\n'.format(os.uname()[1].upper(),
                                                                    time.ctime()))
class CSV_Reader(object):
    """
    Manual Tests CSV reader

    Parameters
    ---------
        csv_filename: str, Valid path to csv file/url
        set_index: str, If you want to change the index, set name
    Returns
    -------
        result: Pandas DataFrame
    """

    def __init__(self, csv_filename, set_index=None):
        self.csv_filename = csv_filename
        self.set_index = set_index

    @property
    def load_csv(self):
        """
        Load csv file

        Parameters
        ----------
            object

        Returns
        -------
            result: Pandas DataFrame
        """
        try:
            assert self.csv_filename
            df = pd.read_csv(self.csv_filename)
            df = df.replace(np.nan, "TBD", regex=True)
            df = df.fillna(method='ffill')
        except:
            return False
        else:
            return df.set_index(self.set_index) if self.set_index else df

    def csv_to_dict(self, ve_number=None):
        """
        CSV contents to Dict

        Parameters
        ----------
            ve_number: Verification Event Number e.g. CBF.V.1.11

        Returns
        -------
        result: dict
        """
        return dict(self.load_csv.loc[ve_number]) if ve_number else None


def wipd(f):
    """
    - "work in progress decorator"
    Custom decorator and flag.

    # Then "nosetests -a wip" can be used at the command line to narrow the execution of the test to
    # the ones marked with @wipd

    Usage:
        @widp
        def test_channelistion(self):
            pass
    """
    return attr('wip')(f)

def channel_center_freqs(self):
    """
    Calculates the center frequencies of all channels.
    First channel center frequency is 0.
    Second element can be used as the channel bandwidth

    Return
    ---------
    List: channel center frequencies
    """
    n_chans = float(self.n_chans)
    bandwidth = float(self.bandwidth)
    ch_bandwidth = bandwidth / n_chans
    f_start = 0.  # Center freq of the first channel
    return f_start + np.arange(n_chans) * ch_bandwidth


def calc_freq_samples(self, chan, samples_per_chan, chans_around=0):
    """Calculate frequency points to sweep over a test channel.

    Parameters
    =========
    chan : int
       Channel number around which to place frequency samples
    samples_per_chan: int
       Number of frequency points per channel
    chans_around: int
       Number of channels to include around the test channel. I.e. value 1 will
       include one extra channel above and one below the test channel.

    Will put frequency sample on channel boundary if 2 or more points per channel are
    requested, and if will place a point in the centre of the channel if an odd number
    of points are specified.

    """
    assert samples_per_chan > 0
    assert chans_around > 0
    assert 0 <= chan < self.n_chans
    assert 0 <= chan + chans_around < self.n_chans
    assert 0 <= chan - chans_around < self.n_chans
    n_chans = float(self.n_chans)
    bandwidth = float(self.bandwidth)
    ch_bandwidth = bandwidth / n_chans

    start_chan = chan - chans_around
    end_chan = chan + chans_around
    if samples_per_chan == 1:
        return channel_center_freqs(self)[start_chan:end_chan + 1]
    start_freq = channel_center_freqs(self)[start_chan] - ch_bandwidth / 2
    end_freq = channel_center_freqs(self)[end_chan] + ch_bandwidth / 2
    sample_spacing = ch_bandwidth / (samples_per_chan - 1)
    num_samples = int(np.round((end_freq - start_freq) / sample_spacing)) + 1
    return np.linspace(start_freq, end_freq, num_samples)
