#!/usr/bin/env python

# https://stackoverflow.com/a/44077346
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cbf@ska.ac.za                                                       #
# Maintainer: mmphego@ska.ac.za, alec@ska.ac.za                               #
# Copyright @ 2016 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

from __future__ import division, print_function

import logging
import os
import random
import time
import unittest

import numpy as np
from ast import literal_eval as evaluate
from nosekatreport import Aqf, aqf_requirements, aqf_vr, system

from avn_tests import signalGen
from avn_tests.aqf_utils import (aqf_plot_and_save, aqf_plot_channels,
                                 aqf_plot_phase_results, aqf_plot_xy,
                                 cls_end_aqf, test_heading)
from avn_tests.avn_rx import AVN_Rx
from avn_tests.utils import (Credentials, calc_freq_samples,
                             channel_center_freqs, executed_by,
                             loggerise, normalised_magnitude, wipd)
# from descriptions import TestProcedure

LOGGER = logging.getLogger(__file__)


@cls_end_aqf
@system('avn')
class test_AVN(unittest.TestCase):
    """ Unit-testing class for AVN tests"""

    def setUp(self):
        Aqf.step('Configuring HDF5 receiver.')
        self.avnControl = AVN_Rx()
        super(test_AVN, self).setUp()
        try:
            Aqf.step('Setting up Signal Generator.')
            self.signalGen = signalGen.SCPI(host=Credentials.hostip)
            self.signalGen.reset()
        except Exception:
            raise RuntimeError("Failed to connect to signal generator")

    def set_instrument(self, acc_time=0.5, **kwargs):

        try:
            self.errmsg = None
            self.n_chans = int(self.avnControl.sensor_request("roachNumFrequencyChannels")[-1])
            self.bandwidth = float(self.avnControl.sensor_request('roachFrequencyFs')[-1]) / 2
            self.Nfft = float(self.avnControl.sensor_request('roachSizeOfCoarseFFT')[-1])

            # This determines whether we're looking at the wideband or narrowband case.
            self.coarse_channel = None
            self.N_finefft = int(self.avnControl.sensor_request('roachSizeOfFineFFT')[-1])
            if self.N_finefft == 0:
                self.mode = "wideband"
                acc_len = 2*self.bandwidth/self.Nfft*acc_time
            else:
                self.mode = "narrowband"
                self.bandwidth /= (self.Nfft/2)
                self.coarse_channel = int(self.avnControl.sensor_request("roachCoarseChannelSelect")[-1])
                acc_len = int(self.bandwidth / self.N_finefft * acc_time)

            Aqf.step('Set and confirm accumulation period via CAM interface.')
            reply, _ = self.avnControl.katcp_request(
                katcprequest='setRoachAccumulationLength', katcprequestArg=acc_len)
            if not reply.reply_ok():
                raise AssertionError()
            actual_acc_len = int(self.avnControl.sensor_request('roachAccumulationLength')[-1])
            Aqf.equals(acc_len, actual_acc_len,
                       'Accumulation time set to {:.3f} seconds'.format(acc_time))
            Aqf.step("Enabled SigGen RF Output")
            self.signalGen.outputOn()
        except Exception as e:
            self.errmsg = ('Failed to set accumulation time due to :{}'.format(e))
            Aqf.failed(self.errmsg)
            LOGGER.exception(self.errmsg)
        else:
            self.addCleanup(self.avnControl.stopCapture)
            self.addCleanup(executed_by)
            self.addCleanup(self.signalGen.close)
            return True

    # @aqf_vr('CBF.V.3.30')
    # @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043",
    #                   "CBF-REQ-0053")
    def test_channelisation(self):
        #        Aqf.procedure(TestProcedure.Channelisation)
        try:
            if not evaluate(os.getenv('DRY_RUN', 'False')):
                raise AssertionError()
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                if self.n_chans == 1024:
                    _mode = "Wideband"
                else:
                    _mode = "NarrowBand"
                test_heading("CBF.AVN Channelisation {} Coarse L-band".format(_mode))
                self._test_channelisation()
            else:
                Aqf.failed(self.errmsg)

    # @aqf_vr('CBF.V.4.7')
    # @aqf_requirements("CBF-REQ-0096")
    # def test_accumulation_length(self):
    #     # The CBF shall set the Baseline Correlation Products accumulation interval to a fixed time
    #     # in the range $$500 +0 -20ms$$.
    #     Aqf.procedure(TestProcedure.VectorAcc)
    #     try:
    #         assert evaluate(os.getenv('DRY_RUN', 'False'))
    #     except AssertionError:
    #         instrument_success = self.set_instrument()
    #         if instrument_success:
    #             if '32k' in self.instrument:
    #                 Aqf.step('Testing maximum channels to %s due to quantiser snap-block and '
    #                          'system performance limitations.' % self.n_chans_selected)
    #             chan_index = self.n_chans_selected
    #             n_chans = self.cam_sensors.get_value('n_chans')
    #             test_chan = random.choice(range(n_chans)[:self.n_chans_selected])
    #             n_ants = self.cam_sensors.get_value('n_ants')
    #             if n_ants == 4:
    #                 acc_time = 0.998
    #             else:
    #                 acc_time = 2 * n_ants / 32.
    #             self._test_vacc(test_chan, chan_index, acc_time)
    #         else:
    #             Aqf.failed(self.errmsg)

    # def _test_channelisation(self, test_chan=212):
    def _test_channelisation(self, test_chan=2212):

        frequency_tweak = -1220

        req_chan_spacing = self.bandwidth / self.n_chans
        requested_test_freqs = calc_freq_samples(
            self, test_chan, samples_per_chan=101, chans_around=2)
        expected_fc = channel_center_freqs(self)[test_chan]
        #Aqf.note("Test channel given: {}, at frequency {}".format(test_chan, expected_fc))
        # Get baseline 0 data, i.e. auto-corr of m000h
        # test_baseline = 0
        # [CBF-REQ-0053]
        min_bandwithd_req = 1.5e6
        # [CBF-REQ-0126] CBF channel isolation
        cutoff = 53  # dB
        # Placeholder of actual frequencies that the signal generator produces
        actual_test_freqs = []
        # Channel magnitude responses for each frequency
        chan_responses = []
        last_source_freq = None
        # print_counts = 3

        cw_power = -49.0
        Aqf.step(
            'Configured signal generator to generate a continuous wave (cwg0), with {} dBm '.format(
                cw_power))
        try:
            _set_freq = self.signalGen.setFrequency(expected_fc)
            if not _set_freq == expected_fc:
                raise AssertionError()
            _set_pw = self.signalGen.setPower(cw_power)
            if not _set_pw == cw_power:
                raise AssertionError()
            Aqf.passed("Signal Generator set successfully.")
            self.avnControl.startCapture()
            time.sleep(1)
        except Exception:
            LOGGER.error("Failed to set Signal Generator parameters",
                exc_info=True)
            return False

        try:
            Aqf.step("Initialise a continuous data recording.")
            Aqf.step('Capture an initial HDF5 accumulation and determine the number of frequency '
                     'channels in the dump.')
            initial_dump = self.avnControl.get_hdf5(stopCapture=True)
            self.assertIsInstance(initial_dump, np.ndarray)
        except Exception:
            errmsg = 'Could not retrieve initial clean HDF5 accumulation.'
            LOGGER.error(errmsg)
            Aqf.failed(errmsg)
            return
        else:
            Aqf.equals(
                self.n_chans, initial_dump.shape[1],
                'Confirm that the number of channels in the SPEAD accumulation, is equal '
                'to the number of frequency channels as calculated: {}'.format(self.n_chans))
            Aqf.step(
                'The CBF, when configured to produce the Imaging data product set and Wideband '
                'Fine resolution channelisation, shall channelise a total bandwidth of >= {}'.
                format(min_bandwithd_req))
            Aqf.is_true(
                self.bandwidth >= min_bandwithd_req,
                'Channelise total bandwidth {}Hz shall be >= {}Hz.'.format(
                    self.bandwidth, min_bandwithd_req))
            chan_spacing = self.bandwidth / initial_dump.shape[1]
            chan_spacing_tol = [chan_spacing - (chan_spacing * 1 / 100),
                chan_spacing + (chan_spacing * 1 / 100)]
            Aqf.step(
                'Confirm that the number of calculated channel frequency step is within requirement.')
            msg = (
                'Verify that the calculated channel frequency (%s Hz)step size is between %s and '
                '%s Hz' % (chan_spacing, req_chan_spacing / 2, req_chan_spacing))
            Aqf.in_range(chan_spacing, req_chan_spacing / 2, req_chan_spacing, msg)

            Aqf.step('Confirm that the channelisation spacing and '
                     'confirm that it is within the maximum tolerance.')
            msg = ('Channelisation spacing is within maximum tolerance of 1% of the '
                   'channel spacing.')
            Aqf.in_range(chan_spacing, chan_spacing_tol[0], chan_spacing_tol[1], msg)

            initial_freq_response = normalised_magnitude((initial_dump[1, :, :])[1:])
            where_is_the_tone = np.argmax(initial_freq_response)
            max_tone_val = np.max(initial_freq_response)
            Aqf.note("Single peak found at channel %s, with max power of %.5f(%.5fdB)" %
                     (where_is_the_tone, max_tone_val, 10 * np.log10(max_tone_val)))

            plt_filename = '{}_overall_channel_resolution_Initial_capture.png'.format(
                self._testMethodName)
            plt_title = 'Initial Overall frequency response at %s' % test_chan
            caption = (
                'An overall frequency response at the centre frequency {}. Signal Generator is '
                'configured to generate a continuous wave, with power {}dBm'.format(test_chan,
                    cw_power))
            aqf_plot_channels(initial_freq_response, plt_filename, plt_title, caption=caption)

        Aqf.step('Sweep the digitiser simulator over the centre frequencies of at '
                 'least all the channels that fall within the complete L-band')
        failure_count = 0

        for i, freq in enumerate(requested_test_freqs):
            self.avnControl.startCapture()
            this_source_freq = self.signalGen.setFrequency(freq + frequency_tweak)
            _msg = ('Getting channel response for freq {} @ {}: {:.3f} MHz.'.format(
                i + 1, len(requested_test_freqs), freq / 1e6))
            Aqf.progress(_msg)
            # if i < print_counts:
            #     Aqf.progress(_msg)
            # elif i == print_counts:
            #     Aqf.progress('.' * print_counts)
            # elif i >= (len(requested_test_freqs) - print_counts):
            #     Aqf.progress(_msg)
            # else:
            #     LOGGER.debug(_msg)
            # import IPython; globals().update(locals()); IPython.embed(header='Python Debugger')
            time.sleep(1)
            if this_source_freq == last_source_freq:
                LOGGER.debug('Skipping channel response for freq {} @ {}: {} MHz.\n'
                             'Digitiser frequency is same as previous.'.format(i + 1,
                                len(requested_test_freqs), freq / 1e6))
                continue  # Already calculated this one
            else:
                last_source_freq = this_source_freq
            try:
                this_freq_dump = self.avnControl.get_hdf5(stopCapture=True)
                self.assertIsInstance(this_freq_dump, np.ndarray)
            except AssertionError:
                #ToDO add a retry here for freq dump
                failure_count += 1
                errmsg = ('Could not retrieve clean accumulation for freq (%s @ %s: %sMHz).' %
                          (i + 1, len(requested_test_freqs), freq / 1e6))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                if failure_count >= 5:
                    _errmsg = 'Cannot continue running the test, Not receiving clean accumulations.'
                    LOGGER.error(_errmsg)
                    Aqf.failed(_errmsg)
                    return False
            else:
                this_freq_response = normalised_magnitude((this_freq_dump[1, :, :])[1:])
                actual_test_freqs.append(this_source_freq)
                chan_responses.append(this_freq_response)

            # Plot an overall frequency response at the centre frequency just as
            # a sanity check

            if np.abs(freq - expected_fc) < 0.1:
                plt_filename = '{}_overall_channel_resolution.png'.format(self._testMethodName)
                plt_title = 'Overall frequency response at {} at {:.3f}MHz.'.format(
                    test_chan, this_source_freq / 1e6)
                max_peak = np.max(loggerise(this_freq_response))
                Aqf.note("Single peak found at channel %s, with max power of %s (%fdB) midway "
                         "channelisation, to confirm if there is no offset." %
                         (np.argmax(this_freq_response), np.max(this_freq_response), max_peak))
                new_cutoff = max_peak - cutoff
                # y_axis_limits = (-100, 1)
                # caption = ('An overall frequency response at the centre frequency, and ({:.3f}dB) '
                #            'and selected baseline {} / {} to test. CBF channel isolation [max channel'
                #            ' peak ({:.3f}dB) - ({}dB) cut-off] when '
                #            'digitiser simulator is configured to generate a continuous wave, with '
                #            'cw scale: {}, awgn scale: {}, Eq gain: {} and FFT shift: {}'.format(
                #                new_cutoff, test_baseline, bls_to_test, max_peak, cutoff, cw_scale,
                #                awgn_scale, gain, fft_shift))

                aqf_plot_channels(this_freq_response, plt_filename, plt_title, cutoff=new_cutoff)

        if not where_is_the_tone == test_chan:
            Aqf.note("We expect the channel response at %s, but in essence it is in channel %s, ie "
                     "There's a channel offset of %s" % (test_chan, where_is_the_tone,
                                                         np.abs(test_chan - where_is_the_tone)))
            test_chan += np.abs(test_chan - where_is_the_tone)

        # Convert the lists to numpy arrays for easier working
        actual_test_freqs = np.array(actual_test_freqs)
        chan_responses = np.array(chan_responses)
        df = self.bandwidth / (self.n_chans - 1)
        try:
            rand_chan_response = len(chan_responses[random.randrange(len(chan_responses))])
            # assert rand_chan_response == self.n_chans_selected
        except AssertionError:
            errmsg = (
                'Number of channels (%s) found on the spead data is inconsistent with the '
                'number of channels (%s) expected.' % (rand_chan_response, self.n_chans_selected))
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
        else:
            plt_filename = '{}_Channel_Response.png'.format(self._testMethodName)
            plot_data = loggerise(chan_responses[:, test_chan], dynamic_range=90, normalise=True)
            plt_caption = ('Frequency channel {} @ {}MHz response vs source frequency'.format(
                test_chan, expected_fc / 1e6))
            plt_title = 'Channel {} @ {:.3f}MHz response.'.format(test_chan, expected_fc / 1e6)

            # Plot channel response with -53dB cutoff horizontal line
            aqf_plot_and_save(
                freqs=actual_test_freqs[1:-1] - frequency_tweak,
                data=plot_data[1:-1],
                df=df,
                expected_fc=expected_fc,
                plot_filename=plt_filename,
                plt_title=plt_title,
                caption=plt_caption)
            # try:
            #     # CBF-REQ-0126
            #     pass_bw_min_max = np.argwhere(
            #         (np.abs(plot_data) >= 3.0) & (np.abs(plot_data) <= 3.3))
            #     pass_bw = float(np.abs(
            #         actual_test_freqs[pass_bw_min_max[0]] - actual_test_freqs[pass_bw_min_max[-1]]))

            #     att_bw_min_max = [np.argwhere(plot_data == i)[0][0] for i in plot_data
            #                       if (abs(i) >= (cutoff - 1)) and (abs(i) <= (cutoff + 1))]
            #     att_bw = actual_test_freqs[att_bw_min_max[-1]] - actual_test_freqs[att_bw_min_max[0]]

            # except Exception:
            #     msg = ('Could not compute if, CBF performs channelisation such that the 53dB '
            #            'attenuation bandwidth is less/equal to 2x the pass bandwidth')
            #     Aqf.failed(msg)
            #     LOGGER.exception(msg)
            # else:
            #     msg = ('The CBF shall perform channelisation such that the 53dB attenuation bandwidth(%s)'
            #            'is less/equal to 2x the pass bandwidth(%s)' % (att_bw, pass_bw))
            #     Aqf.is_true(att_bw >= pass_bw, msg)

            # Get responses for central 80% of channel
            central_indices = ((actual_test_freqs <= expected_fc + 0.4 * df) &
                               (actual_test_freqs >= expected_fc - 0.4 * df))
            central_chan_responses = chan_responses[central_indices]
            central_chan_test_freqs = actual_test_freqs[central_indices]

            # Plot channel response for central 80% of channel
            graph_name_central = '{}_central.png'.format(self._testMethodName)
            plot_data_central = loggerise(
                central_chan_responses[:, test_chan], dynamic_range=90, normalise=True)

            caption = ('Channel {} central response vs source frequency on max channels {}'.format(
                test_chan, self.n_chans))
            plt_title = 'Channel {} @ {:.3f} MHz response @ 80%'.format(
                test_chan, expected_fc / 1e6)

            aqf_plot_and_save(
                central_chan_test_freqs - frequency_tweak,
                plot_data_central,
                df,
                expected_fc,
                graph_name_central,
                plt_title,
                caption=caption)

            Aqf.step(
                'Test that the peak channeliser response to input frequencies in central 80% of '
                'the test channel frequency band are all in the test channel')
            fault_freqs = []
            fault_channels = []
            for i, freq in enumerate(central_chan_test_freqs):
                max_chan = np.argmax(np.abs(central_chan_responses[i]))
                if max_chan != test_chan:
                    fault_freqs.append(freq)
                    fault_channels.append(max_chan)
            if fault_freqs:
                Aqf.failed('The following input frequencies (first and last): {!r} '
                           'respectively had peak channeliser responses in channels '
                           '{!r}\n, and not test channel {} as expected.'.format(
                               fault_freqs[1::-1], set(sorted(fault_channels)), test_chan))

                LOGGER.error('The following input frequencies: %s respectively had '
                             'peak channeliser responses in channels %s, not '
                             'channel %s as expected.' % (fault_freqs, set(sorted(fault_channels)),
                                                          test_chan))

            Aqf.less(
                np.max(np.abs(central_chan_responses[:, test_chan])), 0.99,
                'Confirm that the VACC output is at < 99% of maximum value, if fails '
                'then it is probably over-ranging.')

            max_central_chan_response = np.max(10 * np.log10(central_chan_responses[:, test_chan]))
            min_central_chan_response = np.min(10 * np.log10(central_chan_responses[:, test_chan]))
            chan_ripple = max_central_chan_response - min_central_chan_response
            acceptable_ripple_lt = 1.5
            Aqf.hop('80% channel cut-off ripple at {:.2f} dB, should be less than {} dB'.format(
                chan_ripple, acceptable_ripple_lt))

            # Get frequency samples closest channel fc and crossover points
            co_low_freq = expected_fc - df / 2
            co_high_freq = expected_fc + df / 2

            def get_close_result(_freq):
                ind = np.argmin(np.abs(actual_test_freqs - _freq))
                source_freq = actual_test_freqs[ind]
                response = chan_responses[ind, test_chan]
                return ind, source_freq, response

            # fc_ind, fc_src_freq, fc_resp = get_close_result(expected_fc)
            _, fc_src_freq, fc_resp = get_close_result(expected_fc)
            # co_low_ind, co_low_src_freq, co_low_resp = get_close_result(co_low_freq)
            _, co_low_src_freq, co_low_resp = get_close_result(co_low_freq)
            # co_high_ind, co_high_src_freq, co_high_resp = get_close_result(co_high_freq)
            _, co_high_src_freq, co_high_resp = get_close_result(co_high_freq)
            # [CBF-REQ-0047] CBF channelisation frequency resolution requirement
            Aqf.step('Confirm that the response at channel-edges are -3 dB '
                     'relative to the channel centre at {:.3f} Hz, actual source freq '
                     '{:.3f} Hz'.format(expected_fc, fc_src_freq))

            desired_cutoff_resp = -6  # dB
            acceptable_co_var = 0.1  # dB, TODO 2015-12-09 NM: thumbsuck number
            co_mid_rel_resp = 10 * np.log10(fc_resp)
            co_low_rel_resp = 10 * np.log10(co_low_resp)
            co_high_rel_resp = 10 * np.log10(co_high_resp)

            co_lo_band_edge_rel_resp = co_mid_rel_resp - co_low_rel_resp
            co_hi_band_edge_rel_resp = co_mid_rel_resp - co_high_rel_resp

            low_rel_resp_accept = np.abs(desired_cutoff_resp + acceptable_co_var)
            hi_rel_resp_accept = np.abs(desired_cutoff_resp - acceptable_co_var)

            # cutoff_edge = np.abs((co_lo_band_edge_rel_resp + co_hi_band_edge_rel_resp) / 2)

            no_of_responses = 3
            center_bin = [150, 250, 350]
            y_axis_limits = (-90, 1)

            legends = [
                'Channel {} / Sample {} \n@ {:.3f} MHz'.format(
                    ((test_chan + i) - 1), v,
                    channel_center_freqs(self)[test_chan + i] / 1e6)
                for i, v in zip(range(no_of_responses), center_bin)
            ]
            #center_bin.append('Channel spacing: {:.3f}kHz'.format(856e6 / self.n_chans_selected / 1e3))
            center_bin.append('Channel spacing: {:.3f}kHz'.format(chan_spacing / 1e3))

            channel_response_list = [
                chan_responses[:, test_chan + i - 1] for i in range(no_of_responses)
            ]
            plot_title = 'PFB Channel Response'
            plot_filename = '{}_adjacent_channels.png'.format(self._testMethodName)

            caption = (
                'Sample PFB central channel response between channel {}, with channelisation '
                'spacing of {}kHz within tolerance of 1%, with '
                'the digitiser simulator configured to generate a continuous wave, '
                'with cw scale:'.format(test_chan, chan_spacing / 1e3))

            aqf_plot_channels(
                zip(channel_response_list, legends),
                plot_filename,
                plot_title,
                normalise=True,
                caption=caption,
                vlines=center_bin,
                # normalise=True, caption=caption, cutoff=-cutoff_edge, vlines=center_bin,
                # xlabel='Sample Steps', ylimits=y_axis_limits)
                xlabel='Sample Steps')

            Aqf.step(
                "Measure the power difference between the middle of the center and the middle of "
                "the next adjacent bins and confirm that is > -%sdB" % cutoff)
            for bin_num, chan_resp in enumerate(channel_response_list, 1):
                power_diff = np.max(loggerise(chan_resp)) - cutoff
                msg = "Confirm that the power difference (%.2fdB) in bin %s is more than %sdB" % (
                    power_diff, bin_num, -cutoff)
                Aqf.less(power_diff, -cutoff, msg)

            # Plot Central PFB channel response with ylimit 0 to -6dB
            y_axis_limits = (-7, 1)
            plot_filename = '{}_central_adjacent_channels.png'.format(self._testMethodName)
            plot_title = 'PFB Central Channel Response'
            caption = (
                'Sample PFB central channel response between channel {} , with the digitiser '
                'simulator configured to generate a continuous wave, '.format(test_chan))

            aqf_plot_channels(
                zip(channel_response_list, legends),
                plot_filename,
                plot_title,
                normalise=True,
                caption=caption,
                cutoff=-1.5,
                xlabel='Sample Steps',
                ylimits=y_axis_limits)

            Aqf.is_true(
                low_rel_resp_accept <= co_lo_band_edge_rel_resp <= hi_rel_resp_accept,
                'Confirm that the relative response at the low band-edge '
                '(-{co_lo_band_edge_rel_resp} dB @ {co_low_freq} Hz, actual source freq '
                '{co_low_src_freq}) is within the range of {desired_cutoff_resp} +- 1% '
                'relative to channel centre response.'.format(**locals()))

            Aqf.is_true(
                low_rel_resp_accept <= co_hi_band_edge_rel_resp <= hi_rel_resp_accept,
                'Confirm that the relative response at the high band-edge '
                '(-{co_hi_band_edge_rel_resp} dB @ {co_high_freq} Hz, actual source freq '
                '{co_high_src_freq}) is within the range of {desired_cutoff_resp} +- 1% '
                'relative to channel centre response.'.format(**locals()))

    # def _test_vacc(self, test_chan, chan_index=None, acc_time=0.998):
    #     """Test vector accumulator"""
    #     # Choose a test frequency around the centre of the band.
    #     test_freq = self.bandwidth / 2.
    #
    #     test_input = self.cam_sensors.input_labels[0]
    #     eq_scaling = 30
    #     acc_times = [acc_time / 2, acc_time]
    #     #acc_times = [acc_time/2, acc_time, acc_time*2]
    #     n_chans = self.cam_sensors.get_value("n_chans")
    #     try:
    #         internal_accumulations = int(self.cam_sensors.get_value('xeng_acc_len'))
    #     except Exception as e:
    #         errmsg = 'Failed to retrieve X-engine accumulation length: %s.' % str(e)
    #         LOGGER.exception(errmsg)
    #         Aqf.failed(errmsg)
    #     try:
    #         initial_dump = get_clean_dump(self)
    #         assert isinstance(initial_dump, dict)
    #     except Exception:
    #         errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
    #         Aqf.failed(errmsg)
    #         LOGGER.exception(errmsg)
    #         return
    #
    #     delta_acc_t = self.cam_sensors.fft_period * internal_accumulations
    #     test_acc_lens = [np.ceil(t / delta_acc_t) for t in acc_times]
    #     test_freq_channel = abs(
    #         np.argmin(np.abs(self.cam_sensors.ch_center_freqs[:chan_index] - test_freq)) -
    #         test_chan)
    #     eqs = np.zeros(n_chans, dtype=np.complex)
    #     eqs[test_freq_channel] = eq_scaling
    #     get_and_restore_initial_eqs(self)
    #     try:
    #         reply, _informs = self.avnControl.katcp_rct.req.gain(test_input, *list(eqs))
    # if not reply.reply_ok():
    #     raise AssertionError()
    #         Aqf.hop('Gain successfully set on input %s via CAM interface.' % test_input)
    #     except Exception:
    #         errmsg = 'Gains/Eq could not be set on input %s via CAM interface' % test_input
    #         Aqf.failed(errmsg)
    #         LOGGER.exception(errmsg)
    #
    #     Aqf.step('Configured Digitiser simulator output(cw0 @ {:.3f}MHz) to be periodic in '
    #              'FFT-length: {} in order for each FFT to be identical'.format(
    #                  test_freq / 1e6, n_chans * 2))
    #
    #     cw_scale = 0.125
    #     # The re-quantiser outputs signed int (8bit), but the snapshot code
    #     # normalises it to floats between -1:1. Since we want to calculate the
    #     # output of the vacc which sums integers, denormalise the snapshot
    #     # output back to ints.
    #     # q_denorm = 128
    #     # quantiser_spectrum = get_quant_snapshot(self, test_input) * q_denorm
    #     try:
    #         # Make dsim output periodic in FFT-length so that each FFT is identical
    #         self.signalGen.sine_sources.sin_0.set(
    #             frequency=test_freq, scale=cw_scale, repeat_n=n_chans * 2)
    #         self.signalGen.sine_sources.sin_1.set(
    #             frequency=test_freq, scale=cw_scale, repeat_n=n_chans * 2)
    #         assert self.signalGen.sine_sources.sin_0.repeat == n_chans * 2
    #     except AssertionError:
    #         errmsg = 'Failed to make the DEng output periodic in FFT-length so that each FFT is identical'
    #         Aqf.failed(errmsg)
    #         LOGGER.exception(errmsg)
    #     try:
    #         reply, informs = self.avnControl.katcp_rct.req.quantiser_snapshot(test_input)
    # if not reply.reply_ok():
    #     raise AssertionError()
    #         informs = informs[0]
    #     except Exception:
    #         errmsg = ('Failed to retrieve quantiser snapshot of input %s via '
    #                   'CAM Interface: \nReply %s' % (test_input, str(reply).replace('_', ' ')))
    #         Aqf.failed(errmsg)
    #         LOGGER.exception(errmsg)
    #         return
    #     else:
    #         quantiser_spectrum = np.array(evaluate(informs.arguments[-1]))
    #         if chan_index:
    #             quantiser_spectrum = quantiser_spectrum[:chan_index]
    #         # Check that the spectrum is not zero in the test channel
    #         # Aqf.is_true(quantiser_spectrum[test_freq_channel] != 0,
    #         # 'Check that the spectrum is not zero in the test channel')
    #         # Check that the spectrum is zero except in the test channel
    #         Aqf.is_true(
    #             np.all(quantiser_spectrum[0:test_freq_channel] == 0),
    #             'Confirm that the spectrum is zero except in the test channel:'
    #             ' [0:test_freq_channel]')
    #         Aqf.is_true(
    #             np.all(quantiser_spectrum[test_freq_channel + 1:] == 0),
    #             'Confirm that the spectrum is zero except in the test channel:'
    #             ' [test_freq_channel+1:]')
    #         Aqf.step('FFT Window [{} samples] = {:.3f} micro seconds, Internal Accumulations = {}, '
    #                  'One VACC accumulation = {}s'.format(n_chans * 2,
    #                                                       self.cam_sensors.fft_period * 1e6,
    #                                                       internal_accumulations, delta_acc_t))
    #
    #         chan_response = []
    #         for vacc_accumulations, acc_time in zip(test_acc_lens, acc_times):
    #             try:
    #                 reply = self.avnControl.katcp_rct.req.accumulation_length(acc_time, timeout=60)
    #                 assert reply.succeeded
    #             except Exception:
    #                 Aqf.failed('Failed to set accumulation length of {} after maximum vacc '
    #                            'sync attempts.'.format(vacc_accumulations))
    #             else:
    #                 internal_acc = (2 * internal_accumulations * n_chans)
    #                 accum_len = int(
    #                     np.ceil((acc_time * self.cam_sensors.get_value('sample')) / internal_acc))
    #                 Aqf.almost_equals(
    #                     vacc_accumulations, accum_len, 1,
    #                     'Confirm that vacc length was set successfully with'
    #                     ' {}, which equates to an accumulation time of {:.6f}s'.format(
    #                         vacc_accumulations, vacc_accumulations * delta_acc_t))
    #                 no_accs = internal_accumulations * vacc_accumulations
    #                 expected_response = np.abs(quantiser_spectrum)**2 * no_accs
    #                 try:
    #                     dump = get_clean_dump(self)
    #                     assert isinstance(dump, dict)
    #                 except Exception:
    #                     errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
    #                     Aqf.failed(errmsg)
    #                     LOGGER.exception(errmsg)
    #                 else:
    #                     actual_response_mag = normalised_magnitude(dump['xeng_raw'][:, 0, :])
    #                     chan_response.append(actual_response_mag)
    #                     # Check that the accumulator response is equal to the expected response
    #                     caption = (
    #                         'Accumulators actual response is equal to the expected response for {} '
    #                         'accumulation length with a periodic cw tone every {} samples'
    #                         ' at frequency of {:.3f} MHz with scale {}.'.format(
    #                             test_acc_lens, n_chans * 2, test_freq / 1e6, cw_scale))
    #
    #                     plot_filename = ('{}/{}_chan_resp_{}_vacc.png'.format(
    #                         self.logs_path, self._testMethodName, int(vacc_accumulations)))
    #                     plot_title = ('Vector Accumulation Length: channel %s' % test_freq_channel)
    #                     msg = ('Confirm that the accumulator actual response is '
    #                            'equal to the expected response for {} accumulation length'.format(
    #                                vacc_accumulations))
    #
    #                     if not Aqf.array_abs_error(expected_response[:chan_index],
    #                                                actual_response_mag[:chan_index], msg):
    #                         aqf_plot_channels(
    #                             actual_response_mag,
    #                             plot_filename,
    #                             plot_title,
    #                             log_normalise_to=0,
    #                             normalise=0,
    #                             caption=caption)

    @aqf_vr('TBD')
    @aqf_requirements("TBD")
    def test_linearity(self):
        #Aqf.procedure(TestProcedure.LBandEfficiency)
        try:
            if not evaluate(os.getenv('DRY_RUN', 'False')):
                raise AssertionError()
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_linearity(test_channel=100,
                                cw_start_scale = -8.0,
                                gain = 32,
                                fft_shift = 2047,
                                max_steps = 45)
            else:
                Aqf.failed(self.errmsg)


    def _test_linearity(self, test_channel, cw_start_scale, gain, fft_shift, max_steps):

        ch_bandwidth = self.bandwidth / self.n_chans
        f_start = 400000000.
        f_offset = 50000
        ch_list = f_start + np.arange(self.n_chans) * ch_bandwidth
        freq = ch_list[self.n_chans-test_channel] + f_offset
        try:
            Aqf.step('Setting signal generator frequency to: {:.6f} MHz'.format(freq / 1000000.))
            _set_freq = self.signalGen.setFrequency(freq)
            if not _set_freq == freq:
                raise AssertionError()
            #Aqf.passed("Signal Generator set successfully.")
        except Exception:
            LOGGER.error("Failed to set Signal Generator parameters", exc_info=True)
            return False

        def get_cw_val(cw_scale,gain,fft_shift,test_channel):
            local_freq = ch_list[self.n_chans-test_channel] + f_offset
            #Aqf.step('Signal generator configured to generate a continuous wave at: '
            #        '{:.6f} MHz, cw scale: {} dBm, eq gain: {}, fft shift: {}'.format(
            #                                                                freq / 1000000.,
            #                                                                cw_scale,
            #                                                                gain,
            #                                                                fft_shift))


            try:
                if local_freq != freq:
                    Aqf.step('Setting signal generator frequency to: {:.6f} MHz'.format(freq / 1000000.))
                    _set_freq = self.signalGen.setFrequency(local_freq)
                    if not _set_freq == local_freq:
                        raise AssertionError()
                Aqf.step('Setting signal generator level to: {} dBm'.format(cw_scale))
                _set_pw = self.signalGen.setPower(cw_scale)
                if not _set_pw == cw_scale:
                    raise AssertionError()
                #Aqf.passed("Signal Generator set successfully.")
                self.avnControl.startCapture()
                time.sleep(3)
            except Exception as exc:
                LOGGER.error("Failed to set Signal Generator parameters")
                return False

            try:
                LOGGER.info('Capture a dump via HDF5 file.')
                dump = self.avnControl.get_hdf5(stopCapture=True)
                self.assertIsInstance(dump, np.ndarray)
            except Exception:
                errmsg = 'Could not retrieve clean HDF5 accumulation.'
                LOGGER.error(errmsg)
                Aqf.failed(errmsg)
                return
            # Dump shape = time, channels, left and right values]
            # Use left
            channel_resp = dump[:-1, test_channel, 0]
            channel_resp = channel_resp.sum(axis=0)/channel_resp.shape[0]
            return 10*np.log10(np.abs(channel_resp))

        cw_scale = cw_start_scale
        cw_delta = 1.0
        fullscale = 10*np.log10(pow(2,32))
        curr_val = fullscale
        Aqf.hop('Finding starting cw input scale...')
        max_cnt = max_steps
        while (curr_val >= fullscale) and max_cnt:
            prev_val = curr_val
            curr_val = get_cw_val(cw_scale,gain,fft_shift,test_channel)
            Aqf.hop('curr_val = {}'.format(curr_val))
            cw_scale -= cw_delta
            max_cnt -= 1
        cw_start_scale = cw_scale + 4*cw_delta
        Aqf.hop('Starting cw input scale set to {}'.format(cw_start_scale))
        cw_scale = cw_start_scale
        output_power = []
        x_val_array = []
        # Find closes point to this power to place linear expected line.
        #exp_step = 6
        #exp_y_lvl = 70
        #exp_y_dlt = exp_step/2
        #exp_y_lvl_lwr = exp_y_lvl-exp_y_dlt
        #exp_y_lvl_upr = exp_y_lvl+exp_y_dlt
        #exp_y_val = 0
        #exp_x_val = 0
        min_cnt_val = 3
        min_cnt = min_cnt_val
        max_cnt = max_steps
        while min_cnt and max_cnt:
            curr_val = get_cw_val(cw_scale,gain,fft_shift,test_channel)
            #if exp_y_lvl_lwr < curr_val < exp_y_lvl_upr:
            #    exp_y_val = curr_val
            #    exp_x_val = cw_scale
            step = curr_val-prev_val
            if np.abs(step) < 0.2 or curr_val < 0:
                min_cnt -= 1
            else:
                min_cnt = min_cnt_val
            x_val_array.append(cw_scale)
            Aqf.step('CW power = {}dB, Step = {}dB, channel = {}'.format(curr_val, step, test_channel))
            prev_val=curr_val
            output_power.append(curr_val)
            cw_scale -= cw_delta
            max_cnt -= 1
        output_power = np.array(output_power)
        output_power = output_power - output_power.max()

        plt_filename = '{}_cbf_response_{}_{}.png'.format(self._testMethodName,gain,cw_start_scale)
        plt_title = 'Response (Linearity Test)'
        caption = ('Signal generator start level: {} dBm, end level: {} dBm. '
                   'FFT Shift: {}, Quantiser Gain: {}'
                   ''.format(cw_start_scale, cw_scale, fft_shift,
                                            gain))
        exp_idx = int(len(x_val_array)/3)
        m = 1
        #c = exp_y_val - m*exp_x_val
        c = output_power[exp_idx] - m*x_val_array[exp_idx]
        y_exp = []
        for x in x_val_array:
            y_exp.append(m*x + c)
        aqf_plot_xy(zip(([x_val_array,output_power],[x_val_array,y_exp]),['Response','Expected']),
                     plt_filename, plt_title, caption,
                     xlabel='Input Power [dBm]',
                     ylabel='Integrated Output Power [dBfs]')
        Aqf.end(passed=True, message='TBD')


    @aqf_vr('TBD')
    @aqf_requirements("TBD")
    def test_digital_gain(self):
        #Aqf.procedure(TestProcedure.LBandEfficiency)
        try:
            if not evaluate(os.getenv('DRY_RUN', 'False')):
                raise AssertionError()
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_digital_gain(test_channel=100,
                                cw_scale = -15.0,
                                gain_start = 75,
                                fft_shift = 2047,
                                max_steps = 120)
            else:
                Aqf.failed(self.errmsg)


    def _test_digital_gain(self, test_channel, cw_scale, gain_start, fft_shift, max_steps):
        # This works out what frequency to set the cw source at.
        ch_bandwidth = self.bandwidth / self.n_chans
        f_start = 400000000.
        f_offset = 50000
        ch_list = f_start + np.arange(self.n_chans) * ch_bandwidth
        freq = ch_list[self.n_chans-test_channel] + f_offset
        try:
            Aqf.step('Setting signal generator frequency to: {:.6f} MHz'.format(freq / 1000000.))
            _set_freq = self.signalGen.setFrequency(freq)
            if not _set_freq == freq:
                raise AssertionError()
            #Aqf.passed("Signal Generator set successfully.")
        except Exception as exc:
            LOGGER.error("Failed to set Signal Generator parameters")
            return False

        def get_cw_val(cw_scale,gain,fft_shift,test_channel):
            """Get the CW power value from the given channel."""
            local_freq = ch_list[self.n_chans-test_channel] + f_offset

            try:
                Aqf.step("Setting digital gain to: {}".format(gain))
                self.avnControl.setGain(gain)
                if local_freq != freq:
                    Aqf.step('Setting signal generator frequency to: {:.6f} MHz'.format(freq / 1000000.))
                    _set_freq = self.signalGen.setFrequency(local_freq)
                    if not _set_freq == local_freq:
                        raise AssertionError()
                Aqf.step('Setting signal generator level to: {} dBm'.format(cw_scale))
                _set_pw = self.signalGen.setPower(cw_scale)
                if not _set_pw == cw_scale:
                    raise AssertionError()
                #Aqf.passed("Signal Generator set successfully.")
                self.avnControl.startCapture()
                time.sleep(1)
            except Exception as exc:
                LOGGER.error("Failed to set Signal Generator parameters")
                return False

            try:
                LOGGER.info('Capture a dump via HDF5 file.')
                dump = self.avnControl.get_hdf5(stopCapture=True)
                self.assertIsInstance(dump, np.ndarray)
            except Exception: # TODO - this is bad.
                errmsg = 'Could not retrieve clean HDF5 accumulation.'
                LOGGER.error(errmsg)
                Aqf.failed(errmsg)
                return
            # Dump shape = time, channels, left and right values]
            # Use left
            channel_resp = dump[:-1, test_channel, 0]
            channel_resp = channel_resp.sum(axis=0)/channel_resp.shape[0]
            return np.sqrt(channel_resp)

        # Determine the start of the range, find out where it stops saturating.
        gain = gain_start
        gain_delta = 2.0
        fullscale = pow(2,32)
        curr_val = fullscale
        Aqf.hop('Finding starting gain...')
        max_cnt = max_steps
        while (curr_val >= fullscale) and max_cnt:
            prev_val = curr_val
            curr_val = get_cw_val(cw_scale,gain,fft_shift,test_channel)
            Aqf.hop('curr_val = {}'.format(curr_val))
            gain -= gain_delta
            max_cnt -= 1
        gain_start = gain + 4*gain_delta
        Aqf.hop('Starting gain set to {}'.format(gain_start))

        gain = gain_start
        output_power = []
        x_val_array = []
        # Find closes point to this power to place linear expected line.
        #exp_step = 6
        #exp_y_lvl = 70
        #exp_y_dlt = exp_step/2
        #exp_y_lvl_lwr = exp_y_lvl-exp_y_dlt
        #exp_y_lvl_upr = exp_y_lvl+exp_y_dlt
        #exp_y_val = 0
        #exp_x_val = 0
        min_cnt_val = 3
        min_cnt = min_cnt_val
        max_cnt = max_steps
        while min_cnt and max_cnt:
            curr_val = get_cw_val(cw_scale,gain,fft_shift,test_channel)
            #if exp_y_lvl_lwr < curr_val < exp_y_lvl_upr:
            #    exp_y_val = curr_val
            #    exp_x_val = cw_scale
            step = prev_val/curr_val
            if np.abs(step) < 0.2 or curr_val < 0:
                min_cnt -= 1
            else:
                min_cnt = min_cnt_val
            x_val_array.append(gain)
            Aqf.step('CW power = {}dB, Step = {}dB, channel = {}'.format(curr_val, step, test_channel))
            prev_val=curr_val
            output_power.append(curr_val)
            gain -= gain_delta
            if gain <= 0:
                break
            max_cnt -= 1
        output_power = np.array(output_power)
        #output_power = output_power - output_power.max()

        plt_filename = '{}_cbf_response_{}_{}.png'.format(self._testMethodName,cw_scale,gain_start)
        plt_title = 'Response (Gain Test)'
        caption = ('Gain start level: {}, end level: {}. '
                   'Signal generator level: {}, FFT Shift: {}, Quantiser Gain: {}'
                   ''.format(gain_start, gain, cw_scale, fft_shift,
                                            gain))
        exp_idx = int(len(x_val_array)/3)
        m = np.average(np.diff(output_power[exp_idx:])/np.diff(x_val_array[exp_idx:]))
        #c = exp_y_val - m*exp_x_val
        c = output_power[exp_idx] - m*x_val_array[exp_idx]
        y_exp = []
        for x in x_val_array:
            y_exp.append(m*x + c)
        aqf_plot_xy(zip(([x_val_array,output_power],[x_val_array,y_exp]),['Response','Expected']),
                     plt_filename, plt_title, caption,
                     xlabel='Digital gain',
                     ylabel='Integrated Output Power [raw]')
        Aqf.end(passed=True, message='TBD')


    @aqf_vr('TBD')
    @aqf_requirements("TBD")
    def test_accumulation_length(self):
        #Aqf.procedure(TestProcedure.LBandEfficiency)
        try:
            if not evaluate(os.getenv('DRY_RUN', 'False')):
                raise AssertionError()
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_accumulation_length(test_channel=100,
                                cw_scale = -15.0,
                                gain = 32.0,
                                fft_shift = 2047,
                                accum_length_start=2.0,
                                max_steps = 100)
            else:
                Aqf.failed(self.errmsg)


    def _test_accumulation_length(self, test_channel, cw_scale, gain, fft_shift, accum_length_start, max_steps):
        # This works out what frequency to set the cw source at.
        ch_bandwidth = self.bandwidth / self.n_chans
        f_start = 400000000.
        f_offset = 50000
        ch_list = f_start + np.arange(self.n_chans) * ch_bandwidth
        freq = ch_list[self.n_chans-test_channel] + f_offset
        try:
            Aqf.step('Setting signal generator frequency to: {:.6f} MHz'.format(freq / 1000000.))
            _set_freq = self.signalGen.setFrequency(freq)
            if not _set_freq == freq:
                raise AssertionError()
            Aqf.step('Setting signal generator level to: {} dBm'.format(cw_scale))
            _set_pw = self.signalGen.setPower(cw_scale)
            if not _set_pw == cw_scale:
                raise AssertionError()
            #Aqf.passed("Signal Generator set successfully.")
        except Exception:
            LOGGER.error("Failed to set Signal Generator parameters", exc_info=True)
            return False

        def get_cw_val(acc_len,test_channel):
            """Get the CW power value from the given channel."""
            # local_freq = ch_list[self.n_chans-test_channel] + f_offset

            try:

                Aqf.step("Setting accumulation length to: {}".format(acc_len))
                reply, _ = self.avnControl.katcp_request(
                    katcprequest='setRoachAccumulationLength', katcprequestArg=int(acc_len*390625)) # Because it counts in spectra.
                if not reply.reply_ok():
                    raise AssertionError()
                actual_acc_len = int(self.avnControl.sensor_request('roachAccumulationLength')[-1])
                Aqf.equals(int(acc_len*390625), actual_acc_len,
                           "Accumulation length set to {} frames".format(actual_acc_len))


                time.sleep(acc_len)
                self.avnControl.startCapture()
                time.sleep(2*acc_len + 1) # Just to make sure.
            except Exception as exc:
                LOGGER.error("Failed to set Signal Generator parameters")
                return False

            try:
                LOGGER.info('Capture a dump via HDF5 file.')
                dump = self.avnControl.get_hdf5(stopCapture=True)
                self.assertIsInstance(dump, np.ndarray)
            except Exception as exc:
                errmsg = 'Could not retrieve clean HDF5 accumulation, error reported: {}'.format(exc)
                LOGGER.error(errmsg)
                Aqf.failed(errmsg)
                return
            # Dump shape = time, channels, left and right values]
            # Use left
            channel_resp = dump[:-1, test_channel, 0]
            channel_resp = channel_resp.sum(axis=0)/channel_resp.shape[0]
            return channel_resp

        # Determine the start of the range, find out where it stops saturating.
        accum_length = accum_length_start
        accum_length_delta = 0.125
        fullscale = pow(2,32)
        curr_val = fullscale
        Aqf.hop('Finding starting gain...')
        max_cnt = max_steps
        while (curr_val >= fullscale) and max_cnt:
            prev_val = curr_val
            curr_val = get_cw_val(accum_length,test_channel)
            Aqf.hop('curr_val = {}'.format(curr_val))
            accum_length -= accum_length_delta
            max_cnt -= 1
        accum_length_start = accum_length + 4*accum_length_delta
        Aqf.hop('Starting accumulation length set to {}'.format(accum_length_start))

        accum_length = accum_length_start
        output_power = []
        x_val_array = []
        # Find closes point to this power to place linear expected line.
        #exp_step = 6
        #exp_y_lvl = 70
        #exp_y_dlt = exp_step/2
        #exp_y_lvl_lwr = exp_y_lvl-exp_y_dlt
        #exp_y_lvl_upr = exp_y_lvl+exp_y_dlt
        #exp_y_val = 0
        #exp_x_val = 0
        min_cnt_val = 3
        min_cnt = min_cnt_val
        max_cnt = max_steps
        while min_cnt and max_cnt:
            curr_val = get_cw_val(accum_length,test_channel)
            #if exp_y_lvl_lwr < curr_val < exp_y_lvl_upr:
            #    exp_y_val = curr_val
            #    exp_x_val = cw_scale
            try:
                step = prev_val/curr_val
            except ZeroDivisionError:
                step = 0 # Let's just fix it. Maybe this is bad, but I don't think so.
            if np.abs(step) < 0.2 or curr_val < 0:
                min_cnt -= 1
            else:
                min_cnt = min_cnt_val
            x_val_array.append(accum_length)
            Aqf.step('Accum length = {}s, Step = {}, channel = {}'.format(curr_val, step, test_channel))
            prev_val=curr_val
            output_power.append(curr_val)
            accum_length -= accum_length_delta
            max_cnt -= 1
        output_power = np.array(output_power)
        #output_power = output_power - output_power.max()

        plt_filename = '{}_cbf_response_{}_{}.png'.format(self._testMethodName,cw_scale,accum_length_start)
        plt_title = 'Response (Gain Test)'
        caption = ('Accum length start level: {}, end level: {}. '
                   'Signal generator level: {}, FFT Shift: {}, Quantiser Gain: {}'
                   ''.format(accum_length_start, accum_length, cw_scale, fft_shift,
                                            gain))
        exp_idx = int(len(x_val_array)/3)
        m = 1
        #c = exp_y_val - m*exp_x_val
        c = output_power[exp_idx] - m*x_val_array[exp_idx]
        y_exp = []
        for x in x_val_array:
            y_exp.append(m*x + c)
        aqf_plot_xy(zip(([x_val_array,output_power],[x_val_array,y_exp]),['Response','Expected']),
                     plt_filename, plt_title, caption,
                     xlabel='Accumulation length [s]',
                     ylabel='Integrated Output Power [raw]')
        Aqf.end(passed=True, message='TBD')
