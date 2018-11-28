#!/usr/bin/env python

from __future__ import division, print_function

import argparse
import functools
import glob
import logging
import os
import sys
import time

import h5py
import katcp
import numpy as np
import paramiko

from avn_tests.utils import Credentials, retry

class AllowAnythingPolicy(paramiko.MissingHostKeyPolicy):
    @classmethod
    def missing_host_key(cls, client, hostname, key):
        return


# This class could be imported from a utility module
class LoggingClass(object):
    @property
    def logger(self):
        name = '.'.join(
            [os.path.basename(sys.argv[0]), self.__class__.__name__])
        return logging.getLogger(name)


class AVN_Rx(LoggingClass):
    def __init__(self):
        self._katcp_ip = Credentials.katcpip
        self._katcp_port = 7147
        self._timeout = 10
        self._username = Credentials.username
        self._password = Credentials.password
        self._setUp()
        super(AVN_Rx, self).__init__()

    def _setUp(self):
        self._dir_remote = "/home/{}/Data/RoachAcquisition/".format(self._username)
        assert os.path.exists(self._dir_remote)
        self._dir_local = '/home/{}/avn_data'.format(self._username)
        self._dir_local_dump = '/home/{}/avn_data/dump'.format(self._username)
        assert self._username, "Username is needed!!!!"
        if not os.path.exists(self._dir_local_dump):
            self.logger.debug('Created {} for storing images.'.format(self._dir_local_dump))
            os.makedirs(self._dir_local_dump)

    @retry
    def katcp_request(self, katcprequest='help', katcprequestArg=None):
        """
        Katcp requests

        Parameters
        =========
        katcprequest: str
            Katcp requests messages [Defaults: 'help']
        katcprequestArg: str
            katcp requests messages arguments [Defaults: None]

        Return
        ======
        reply, informs : tuple
            katcp request messages
        """
        client = katcp.BlockingClient(self._katcp_ip, self._katcp_port)
        client.setDaemon(True)
        client.start()
        time.sleep(.1)
        is_connected = client.wait_running(self._timeout)
        time.sleep(.1)
        if not is_connected:
            client.stop()
            return

        try:
            if katcprequestArg:
                reply, informs = client.blocking_request(katcp.Message.request(katcprequest, katcprequestArg),
                    timeout=self._timeout)
            else:
                reply, informs = client.blocking_request(katcp.Message.request(katcprequest),
                    timeout=self._timeout)

            assert reply.reply_ok()
            self.logger.debug("Successfully executed kcpcmd: Reply: {}".format(reply))
        except Exception as exc:
            self.logger.error("Failed to connect to katcp client({}): {}".format(self._katcp_ip, exc))
            return
        else:
            client.stop()
            client = None
            return reply, informs

    def sensor_request(self, *args):
        try:
            if not args > 1:
                raise AssertionError()
            _, informs = self.katcp_request('sensor-value', args[0])
            return informs[0].arguments
        except Exception as exc:
            self.logger.error("Failed to connect to katcp: {}".format(exc))

    def scp_hdf5(self, file_format='.h5', stopCapture=False):
        """
        scp latest HDF5 file over to x server from y server.
        """
        if stopCapture:
            self.stopCapture()
            time.sleep(3) # Give it a bit more time to finish writing out.

        latest = 0
        latestfile = None
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(AllowAnythingPolicy())
        try:
            client.connect(self._katcp_ip, username=self._username, password=self._password)
            self.logger.debug("SSHing to {}".format(self._katcp_ip))
        except Exception as exc:
            self.logger.error("Could not connect to {} due to {}".format(self._katcp_ip, exc))
            client.close()
            raise

        sftp = client.open_sftp()
        sftp.chdir(self._dir_remote)

        for fileattr in sftp.listdir_attr():
            if fileattr.filename.endswith(file_format) and fileattr.st_mtime > latest:
                latest = fileattr.st_mtime
                latestfile = fileattr.filename
        # TODO
        # Add a time.ctime check, to see if the file is new or not
        if latestfile is not None:
            new_file = '/'.join([self._dir_local, latestfile])
            self.logger.debug("Copying {} to {}".format(latestfile, new_file))
            sftp.get(latestfile, new_file)
        client.close()
        try:
            assert os.path.exists(new_file)
        except Exception as exc:
            errmsg = "HDF5 File does not exist: {}".format(exc)
            self.logger.error(errmsg)
            raise RuntimeError(errmsg)

        #import IPython;IPython.embed()
        try:
            self.logger.debug("Extracting data from HDF5 ({})".format(new_file))
            with h5py.File(new_file, 'r') as fin:
                data = fin['Data'].values()
                for element in data:
                    # /Data/Left Power time average
                    # /Data/Right Power time average
                    # /Data/Stokes Q time average
                    # /Data/Stokes U time average
                    # /Data/StokesData
                    # /Data/Timestamps
                    # /Data/VisData
                    # if element.name.find('StokesData') > -1:
                    #     data_stokes = np.array(element.value)
                    if element.name.find('VisData') > -1:
                        data_raw = np.array(element.value)
                    # elif element.name.find('Timestamps') > -1:
                    #     data_ts = np.array(element.value)
        except Exception as exc:
            self.logger.error("{} :FAILED TO READ HDF5 FILE({})".format(exc, new_file))
            return
        # Ready to play with the data
        # np.array(data_stokes).shape
        # Out[9]: (3, 1024, 2)
        try:
            self.logger.debug("Backup the latest HDF5 file to {}".format(self._dir_local_dump))
            os.rename(new_file, '/'.join([self._dir_local_dump, latestfile]))
            # return data_stokes
            return data_raw
        except Exception:
            self.logger.error("Issues with the file name")
            return

    def get_hdf5(self, file_format='.h5', stopCapture=False, timeout=10):
        """
        get local HDF5 file
        """
        settling_time = 0.5
        if stopCapture:
            self.stopCapture()
            while int(self.sensor_request("recordingStopTime")[-1]) != 0 and timeout:
                timeout -= 1
                time.sleep(settling_time)
            if not timeout:
                raise RuntimeError("Timed out while waiting for recording to stop!")

            time.sleep(settling_time)

        try:
            epoch_time = time.time()
            latestfile = None
            latestfile = max(glob.iglob('*'.join([self._dir_remote, file_format])),
                key=os.path.getctime)
            assert os.path.exists(latestfile)
            file_timestamp = latestfile.split('/')[-1].replace('.h5', '').replace("\\@ 0 0", '').replace('_','')
            pattern = "%Y-%m-%dT%H.%M.%S.%f"
            epoch_timestamp = int(time.mktime(time.strptime(file_timestamp, pattern)))
            comp_timestamp = abs(epoch_time - epoch_timestamp - settling_time)
            self.logger.info(
                "Data timestamps -> epoch_timestamp:{}, epoch_time:{}, comp_timestamp:{}".format(
                    epoch_timestamp, epoch_time, comp_timestamp))
            if comp_timestamp <= 20: # This used to be 10. Is there a specific reason?
                self.logger.debug("Extracting data from HDF5 ({})".format(latestfile))
                with h5py.File(latestfile, 'r') as fin:
                    data = fin['Data'].values()
                    for element in data:
                        if element.name.find('VisData') > -1:
                            data_raw = np.array(element.value)
                            self.logger.info("Max Data: {}, Min Data: {}".format(
                                np.max(data_raw), np.min(data_raw)))
                self.logger.info("Backup the latest HDF5 ({}) file to {}".format(self._dir_remote,
                    self._dir_local_dump))
                os.rename(latestfile, '/'.join([self._dir_local_dump, latestfile.split('/')[-1]]))
                self.logger.debug("get_hdf5() returning: {}".format(data_raw))
                return data_raw
            else:
                self.logger.warn("get_hdf5(): comp_timestamp > 10 - not returning data!")
        except AssertionError:
            raise RuntimeError
        except Exception as exc:
            self.logger.error("{} :FAILED TO READ HDF5 FILE({})".format(exc, latestfile))

    def startCapture(self, recording_duration=0):
        try:
            return self.katcp_request(katcprequest='startRecording', katcprequestArg='\@ 0 {}'.format(
                recording_duration)) if recording_duration else self.katcp_request(katcprequest='startRecording')
        except Exception as exc:
            self.logger.error("Failed to start recording: {}".format(exc))

    def stopCapture(self):
        try:
            reply, informs = self.katcp_request(katcprequest='stopRecording')
            assert reply.reply_ok()
            self.logger.debug("Stop data recording")
            return reply
        except Exception as exc:
            self.logger.error("Failed to stop recording: {}".format(exc))

    def setGain(self, gain=32):
        try:
            reply, informs = self.katcp_request(katcprequest="setRoachDspGain", katcprequestArg="{}".format(gain))
            assert reply.reply_ok()
            actual_gain = float(self.sensor_request("roachDspGain")[-1])
            assert np.floor(actual_gain) == np.floor(gain), "Gain not set."
            self.logger.debug(("Set digital gain to {}, actual {}".format(gain, actual_gain)))
            return reply
        except Exception as exc:
            self.logger.error("Failed to set DSP gain: {}".format(exc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy data from AVN system')
    parser.add_argument('--hostname', dest='hostname', action='store', default='10.8.5.8',
                        help='AVN Hostname')
    parser.add_argument('-u', '--username', dest='username', action='store', default='', required=True,
                        help='Username')
    parser.add_argument('-p', '--password', dest='password', action='store', default='', required=True,
                        help='Password')
    args = vars(parser.parse_args())

    # Lets play
    # import IPython; globals().update(locals()); IPython.embed(header='Python Debugger')
