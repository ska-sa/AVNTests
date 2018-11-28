from __future__ import print_function
import atexit
# import pandas as pd
import sys
import time
# import h5py
# import os
# import glob
import numpy as np
import matplotlib.pyplot as plt

from avn_tests.utils import normalised_magnitude
from avn_tests.avn_rx import AVN_Rx

# avn_data_path = "/home/avnuser/avn_data/dump/"
avn_data_path = "/home/avnuser/Data/RoachAcquisition/"

class RealTimePlot(AVN_Rx):

    def __init__(self):
        super(RealTimePlot, self).__init__()

    def on_launch(self):
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([])

    def on_running(self, xdata):
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(np.arange(len(xdata)))
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def __call__(self):
        self.on_launch()
        try:
            while True:
                self.startCapture()
                time.sleep(1)
                raw_data = self.get_hdf5(stopCapture=True)
                xdata = normalised_magnitude(raw_data[1, :, :])[1:]
                self.on_running(xdata)
                print ("{}".format(xdata), end='\r')
        except Exception:
            time.sleep(0.1)
            sys.stdout.flush()
        else:
            return xdata

if __name__ == '__main__':

    receiver = AVN_Rx()
    atexit.register(receiver.stopCapture)

    try:
        realtime_plot = RealTimePlot()
        realtime_plot()
    except KeyboardInterrupt:
        import IPython; globals().update(locals()); IPython.embed(header='Python Debugger')
