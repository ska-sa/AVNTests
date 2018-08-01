#!/usr/bin/python

##Basic socket interface to the R&S signal generator used for CW test signal input

import logging
import os
import serial
import socket
import sys
import time


# This class could be imported from a utility module
class LoggingClass(object):
    @property
    def logger(self):
        name = '.'.join([os.path.basename(sys.argv[0]), self.__class__.__name__])
        return logging.getLogger(name)


class SCPI(LoggingClass):
    """Connect to the R&S signal generator"""

    def __init__(self, host=None, port=5025, device=None, baudrate=115200, timeout=5,
                connection='socket', display_info=False):
        try:
            self.host = host
            assert isinstance(self.host, str)
            self.port = port
            self._device = device
            self._connection = connection
            self._display_info = display_info
            self._baudrate = baudrate
            self._timeout = timeout
            self._enabled_output = False
            if self.host and self._device:
                raise RuntimeError(
                    'Only one connection can be initiated at a time.\n'
                    'Select socket or serial connection.\n')
        except Exception as exc:
            self.logger.error("{}: Error occurred.".format(exc))
        else:
            self._connect()

    def _connect(self):
        # Ethernet socket connection
        if self.host:
            try:
                self.logger.debug("Initialising socket connection")
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(self._timeout)
                status = self._sock.connect_ex((self.host, self.port))
                self.logger.debug("Connected to sigGen on {}:{}".format(self.host, self.port))
                assert status == 0
            except AssertionError:
                msg = "Failed to connect to the sigGen, Link is down!!"
                self._sock.close()
                self.logger.error(msg)
                raise RuntimeError(msg)
            except socket.error:
                msg = "Cannot connect to signal generator"
                self._sock.close()
                self.logger.error(msg)
                raise RuntimeError(msg)
        elif self._device:
            self._connection = 'serial'
            self._sock = serial.Serial(self._device, self.baudrate, self._timeout, rtscts=0)
        else:
            raise RuntimeError('No connections specified.\n')

        # Query instrument identification
        if self._display_info:
            self.write("*IDN?")
            self.logger.info("DEVICE ID: {}".format(self.read()))
        # Ensure that the RF output is active
        time.sleep(1)
        self.outputOn()

    # send query / command via relevant port comm
    def write(self, command):
        self.logger.debug("send query / command via relevant port comm")
        if self._connection == 'serial':
            self._sock.write(command + '\r\n')
        else:
            self._sock.send(command + '\n')
        time.sleep(1)

    def read(self):
        self.logger.debug("read query / command via relevant port comm")
        return self._sock.readline() if self._connection == 'serial' else self._sock.recv(128)

    # activates RF output
    def outputOn(self):
        try:
            self.logger.debug('Enable RF Output')
            self.write("OUTPut ON")
            self._enabled_output = True
        except Exception:
            self.logger.error("Failed to enable RF Output")

    # deactivates the RF output
    def outputOff(self):
        self.logger.debug('Disable RF Output')
        self.write("OUTPut OFF")

    # reset
    def reset(self):
        self.logger.debug('Signal Generator Reset')
        self.write("*RST")
        self.write("*CLS")

    # close the comms port to the R&S signal generator
    def close(self):
        self.logger.debug('Closing comms port to the signal generator')
        self.outputOff()
        self.reset()
        self._sock.close()

    # set requested frequency
    def setFrequency(self, freq):
        self.logger.debug("Setting frequency: {} Hz".format(freq))
        try:
            self.outputOn()
            self.write("FREQuency {:.2f}".format(freq))     # Hz
            return self.getFrequency()
        except Exception:
            self.logger.error("Failed to set frequency to {}".format(freq))

    # read signal generator frequency
    def getFrequency(self):
        try:
            self.write('FREQuency?')
            return_freq = float(self.read())
            self.logger.debug("Frequency read from sigGen: {} Hz".format(return_freq))
            return return_freq      # Hz
        except Exception as exc:
            self.logger.exception("{} :Failed to read the frequency".format(exc))
            # print return_freq.split('\n')

    # set requested power level
    def setPower(self, pwr):
        try:
            self.logger.debug("Set requested power level to {}dBm".format(pwr))
            self.write('POWer {}'.format(pwr))      # dBm
            return self.getPower()
        except Exception as exc:
            self.logger.error("{}: Failed to set power levels".format(exc))

    # read sig gen power level
    def getPower(self):
        self.logger.debug("Read sigGen power level")
        try:
            self.write('POWer?')
            returned_pwr = float(self.read())       # dBm
            self.logger.debug("Power read from sigGen: {} dBm".format(returned_pwr))
            return returned_pwr         # dBm
        except Exception as exc:
            self.logger.error("{}: Failed to read sigGen power levels".format(exc))

#     def setSquare(self):
#         #square function
#         self._sock.send("FUNCtion SQUare\n")
#
#     def setSin(self):
#         #sine function
#         self._sock.send("FUNCtion SIN\n")
#
#     def setVoltage(self, low, high):
#         #set initial voltage
#         self._sock.send("VOLTage:HIGH %.2f\n"%(high,))
#         self._sock.send("VOLTage:LOW %.2f\n"%(low,))
#
#     def setLinSweep(self, start, stop, time):
#         #set linear sweep with the specified start/stop freqs
#         self._sock.send("FREQ:STAR %.2f\n"%(start,))
#         self._sock.send("FREQ:STOP %.2f\n"%(stop,))
#         self._sock.send("SWE:SPAC LIN\n")
#         self._sock.send("SWE:TIME %.3f\n"%(time,))
#         self._sock.send("SWE:STAT ON\n")



if __name__ == '__main__':

    # SMB100A R&S Signal Generator IP address
    siggen_ip = '10.8.5.72'
    siggen_port = 5025

# ## Opening socket to signal generator for CW input signal
#   siggen_socket=socket.socket()
#   siggen_socket.connect((siggen_ip, siggen_port))
#   siggen_socket.send("OUTPut OFF\n")
#   print 'Closing all ports...'
#   try:
#     siggen_socket.close()
#   except:
#     pass # socket already closed
#
#   raw_input('Switching off RF output for testing. Press ENTER to switch RF ON and continue')

    # Using SCPI class for comms to signal generator for CW input signal
    sigme = SCPI(siggen_ip)
    set_freq = 154.4e6      # Hz
    print 'Frequency requested: %.2f Hz' % set_freq
    sigme.setFrequency(freq=set_freq)       # Hz
    if ('%.2f' % set_freq) != ('%.2f' % sigme.getFrequency()):
        print 'Incorrect frequency received from siggen: %.2f Hz' % sigme.getFrequency()
    else:
        print 'Frequency set to: %f MHz' % (sigme.getFrequency() / 1e6)
    set_pwr = -27       # dBm
    print 'Power level requested: %s dBm' % set_pwr
    sigme.setPower(pwr=set_pwr)
    if ('%.2f' % set_pwr) != ('%.2f' % sigme.getPower()):
        print 'Incorrect power received from siggen: %.2f dBm' % sigme.getPower()
    else:
        print 'Power set to: %f dBm' % sigme.getPower()
    sigme.outputOn()
    time.sleep(5)
    sigme.outputOff()

    print 'Closing all ports...'
    try:
        sigme.close()
    except:
        # socket already closed
        pass
