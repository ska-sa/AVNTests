from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

from avn_tests import signalGen
from avn_tests.utils import calc_freq_samples, Credentials
from avn_tests.avn_rx import AVN_Rx


# Variables
n_chans = 1024
bandwidth = 512e6


# Which tests to run
test_center_frequencies = False
test_channelisation = False
test_linearity_with_signal_gen = False
test_attenuators = False
test_gain = True
test_accumulation = True

# Derived values
ch_bandwidth = bandwidth / n_chans


def get_channel_center_freqs(n_chans, bandwidth):
    #TODO this is for the wideband case. Adapt to narrowband.
    #Also think about nyquist zoning... but meh, maybe not.
    f_start = 2*bandwidth;
    ch_bandwidth = bandwidth / n_chans
    return f_start - np.arange(n_chans) * ch_bandwidth

def channel_with_max_power(spectrum):
    return np.argmax(spectrum[1:]) + 1


def dump_to_spectrum(dump):
    spectrum = 10*np.log10(np.average(dump[:,:,0], axis=0))
    return spectrum

def normalised_spectrum(spectrum):
    return spectrum - spectrum[channel_with_max_power(spectrum)]


def plot_spectrum(freq, spectrum):
    plt.figure(figsize=(10,8))
    plt.plot(freq / 1e6, spectrum)
    plt.ylabel("Normalised power [dB]")
    plt.xlabel("Frequency [MHz]")
    plt.title("Spectrum")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    signal_gen = signalGen.SCPI(host=Credentials.hostip)
    print("Resetting signal generator.")
    signal_gen.reset()

    channel_center_freqs = get_channel_center_freqs(n_chans, bandwidth)

    receiver = AVN_Rx()


    if test_channelisation:
        start_time = datetime.datetime.now()
        print("Testing channel response.")

        test_channel = 500
        surrounding_channels = 2
        points_per_channel = 101
        test_attenuation = 63
        test_gain = 0.125
        cw_power = -22.0 # dBm

        test_frequency = channel_center_freqs[test_channel]
        channels_to_watch = slice(test_channel - surrounding_channels, test_channel + surrounding_channels + 1)
        channel_responses = []
        frequencies = np.linspace(channel_center_freqs[test_channel - surrounding_channels] + ch_bandwidth/2,
                                  channel_center_freqs[test_channel + surrounding_channels] - ch_bandwidth/2,
                                  (points_per_channel - 1)*(2*surrounding_channels + 1) + 1)

        print("Setting ADC attenuation to {} dB for this test.".format(atten / 2.0))
        receiver.katcp_request(katcprequest="setRoachADC0Attenuation", katcprequestArg="{:d}".format(test_atten))
        receiver.katcp_request(katcprequest="setRoachADC1Attenuation", katcprequestArg="{:d}".format(test_atten) )

        print("Setting signal generator output power to {} dBm for this test.".format(cw_power))
        signal_gen.setPower(cw_power)

        print("Setting DSP gain to {} for this test.".format(test_gain))
        receiver.katcp_request(katcprequest="setRoachDspGain", katcprequestArg="{:d}".format(test_gain))

        for i, freq in enumerate(frequencies):
            print("Setting signal gen to {} MHz ({} of {})".format(signal_gen.setFrequency(freq)/1e6, i+1, len(frequencies)))

            receiver.startCapture()
            time.sleep(1)
            dump = receiver.get_hdf5(stopCapture=True)
            spectrum = dump_to_spectrum(dump)
            channel_responses.append(spectrum[channels_to_watch])

        channel_responses = np.array(channel_responses)


        # Clamshell response of all N channels.
        plt.figure(figsize=(12,10))
        for i in range(channel_responses.shape[1]):
            plt.plot(frequencies/1e6, channel_responses[:,i] - np.max(channel_responses), label="Channel {} ".format(test_channel - surrounding_channels + i))

        plt.grid()
        plt.legend()
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Channel response [dB]")
        plt.savefig("clamshell_response.png")
        plt.close()


        # Test channel response.
        norm_channel_response = channel_responses[:,channel_responses.shape[1]/2] \
                                - np.max(channel_responses[:,channel_responses.shape[1]/2])
        plt.figure(figsize=(12,10))
        plt.plot(frequencies/1e6, norm_channel_response,
                 label="Channel {}".format(test_channel))
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Channel response [dB]")
        plt.title("Channel {} response".format(test_channel))
        plt.grid()
        plt.plot([(test_frequency - ch_bandwidth/2)/1e6, (test_frequency - ch_bandwidth/2)/1e6], [np.max(norm_channel_response), np.min(norm_channel_response)], dashes=[2, 2], color="black")
        plt.plot([(test_frequency + ch_bandwidth/2)/1e6, (test_frequency + ch_bandwidth/2)/1e6], [np.max(norm_channel_response), np.min(norm_channel_response)], dashes=[2, 2], color="black")
        plt.plot(frequencies/1e6, np.ones(len(frequencies))*-3, dashes=[2,2], color="red")
        plt.plot(frequencies/1e6, np.ones(len(frequencies))*-6, dashes=[6,1], color="red")
        plt.savefig("test_channel_response.png")

        plt.xlim((test_frequency - 3.0/4*ch_bandwidth)/1e6, (test_frequency + 3.0/4*ch_bandwidth)/1e6)
        plt.ylim(-10, 2)
        plt.title("Channel {} response (zoomed)".format(test_channel))
        plt.savefig("test_channel_zoomed.png")

        end_time = datetime.datetime.now()
        print("Channelisation test took {}.".format(end_time - start_time))


    if test_linearity_with_signal_gen:
        print("Testing linearity with signal generator.")
        start_time = datetime.datetime.now()
        test_channel = 500
        test_gain = 0.125
        input_power_range = np.arange(-80.0, -5.0, 0.25)

        test_frequency = channel_center_freqs[test_channel] - ch_bandwidth / 10.0 # Just so that I'm not in the centre of the bin
        print("Setting signal gen to {} MHz for the test.".format(signal_gen.setFrequency(test_frequency)/1e6))

        print("Setting DSP gain to {} for this test.".format(test_gain))
        receiver.katcp_request(katcprequest="setRoachDspGain", katcprequestArg="{:d}".format(test_gain))

        output_powers = []

        for i, pwr in enumerate(input_power_range):
            print("Setting output power to {} dBm. ({} of {})".format(signal_gen.setPower(pwr), i, len(input_power_range)))
            receiver.startCapture()
            time.sleep(1)
            dump = receiver.get_hdf5(stopCapture=True)
            spectrum = dump_to_spectrum(dump)
            output_powers.append(spectrum[test_channel])

        output_powers = np.array(output_powers)

        plt.figure(figsize=(12,10))
        plt.plot(input_power_range, output_powers)
        plt.xlabel("Input power [dBm]")
        plt.ylabel("Output power [dBa]")
        plt.title("Linearity tested with signal generator at {} MHz.".format(test_frequency/1e6))
        plt.grid()
        plt.savefig("linearity.png")

        end_time = datetime.datetime.now()
        print("Linearity with signal gen test took {}.".format(end_time - start_time))

    if test_attenuators:
        print("Testing built-in attenuators.")
        start_time = datetime.datetime.now()
        test_channel = 500
        test_power = -40.0
        test_gain = 0.125
        attenuator_range = np.arange(0, 63, 6)

        test_frequency = channel_center_freqs[test_channel] - ch_bandwidth / 10.0 # Just so that I'm not in the centre of the bin
        print("Setting signal gen to {} MHz for the test.".format(signal_gen.setFrequency(test_frequency)/1e6))
        print("Setting signal gen to {} dBm for the test.".format(signal_gen.setPower(test_power)))

        print("Setting DSP gain to {} for this test.".format(test_gain))
        receiver.katcp_request(katcprequest="setRoachDspGain", katcprequestArg="{:d}".format(test_gain))


        output_powers = []

        for i, atten in enumerate(attenuator_range):
            print("Setting attenuator to {} dB. ({} of {})".format(atten / 2.0, i + 1, len(attenuator_range)))
            receiver.katcp_request(katcprequest="setRoachADC0Attenuation", katcprequestArg="{:d}".format(atten))
            receiver.katcp_request(katcprequest="setRoachADC1Attenuation", katcprequestArg="{:d}".format(atten) )
            receiver.startCapture()
            time.sleep(1)
            dump = receiver.get_hdf5(stopCapture=True)
            spectrum = dump_to_spectrum(dump)
            output_powers.append(spectrum[test_channel])

        output_powers = np.array(output_powers)

        plt.figure(figsize=(12,10))
        plt.plot(attenuator_range / 2.0, output_powers)
        plt.xlabel("Attenuator setting [dB]")
        plt.ylabel("Output power [dBa]")
        plt.title("Variable attenuator tested at {} MHz.".format(test_frequency/1e6))
        plt.grid()
        plt.savefig("attenuators.png")

        end_time = datetime.datetime.now()
        print("Attenuator test took {}.".format(end_time - start_time))



    if test_gain:
        print("Testing digital gain.")
        start_time = datetime.datetime.now()
        test_channel = 500
        test_power = -40.0
        test_attenuation = 63
        gain_range = np.arange(0.15625, 16.0, 1.0)

        test_frequency = channel_center_freqs[test_channel] - ch_bandwidth / 10.0 # Just so that I'm not in the centre of the bin
        print("Setting signal gen to {} MHz for the test.".format(signal_gen.setFrequency(test_frequency)/1e6))
        print("Setting signal gen to {} dBm for the test.".format(signal_gen.setPower(test_power)))

        print("Setting attenuation to {} dB for this test.".format(test_attenuation / 2.0))
        receiver.katcp_request(katcprequest="setRoachADC0Attenuation", katcprequestArg="{:d}".format(test_attenuation))
        receiver.katcp_request(katcprequest="setRoachADC1Attenuation", katcprequestArg="{:d}".format(test_attenuation) )

        output_powers = []

        for i, gain in enumerate(gain_range):
            print("Setting digital gain to {}. ({} of {})".format(gain, i + 1, len(gain_range)))
            receiver.katcp_request(katcprequest="setRoachDspGain", katcprequestArg="{:d}".format(gain))
            receiver.startCapture()
            time.sleep(1)
            dump = receiver.get_hdf5(stopCapture=True)
            spectrum = dump_to_spectrum(dump)
            output_powers.append(spectrum[test_channel])

        output_powers = np.array(output_powers)

        plt.figure(figsize=(12,10))
        plt.plot(attenuator_range / 2.0, output_powers)
        plt.xlabel("Gain setting")
        plt.ylabel("Output power [dBa]")
        plt.title("Digital gain tested at {} MHz.".format(test_frequency/1e6))
        plt.grid()
        plt.savefig("gain.png")

        end_time = datetime.datetime.now()
        print("Digital gain test took {}.".format(end_time - start_time))



    #cleanup
    signal_gen.reset()

