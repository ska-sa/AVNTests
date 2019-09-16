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
channelisation_test = True
linearity_test = True
attenuator_test = True
gain_test = True
accumulation_test = True

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


    #default parameters when these are not the things being tested
    test_channel = 500
    test_attenuation = 63
    test_gain = 0.125
    test_power = -40
    test_accumulation = 0.5
    test_name = ""

    if channelisation_test:
        test_name = "Channelisation"
        print("Testing channel response.")
        start_time = datetime.datetime.now()

        #Setup specific to this test
        surrounding_channels = 2
        points_per_channel = 101
        cw_power = -22.0  # dBm

        print("Setting ADC attenuation to {} dB for {} test.".format(test_attenuation / 2.0, test_name))
        receiver.katcp_request(katcprequest="setRoachADC0Attenuation", katcprequestArg="{:d}".format(test_attenuation))
        receiver.katcp_request(katcprequest="setRoachADC1Attenuation", katcprequestArg="{:d}".format(test_attenuation))

        print("Setting DSP gain to {} for {} test.".format(test_gain, test_name))
        receiver.katcp_request(katcprequest="setRoachDspGain", katcprequestArg="{:d}".format(test_gain))

        print("Setting accumulation time to {} s for {} test.".format(test_accumulation, test_name))
        receiver.katcp_request(katcprequest="setRoachAccumulationLength", katcprequestArg="{:d}".format(test_accumulation*500000))

        print("Setting signal generator output power to {} dBm for {} test.".format(cw_power, test_name))
        signal_gen.setPower(cw_power)

        #Frequency range
        test_frequency = channel_center_freqs[test_channel]
        channels_to_watch = slice(test_channel - surrounding_channels, test_channel + surrounding_channels + 1)
        channel_responses = []
        frequencies = np.linspace(channel_center_freqs[test_channel - surrounding_channels] + ch_bandwidth/2,
                                  channel_center_freqs[test_channel + surrounding_channels] - ch_bandwidth/2,
                                  (points_per_channel - 1)*(2*surrounding_channels + 1) + 1)

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


    if linearity_test:
        test_name = "Linearity"
        print("Testing linearity with signal generator.")
        start_time = datetime.datetime.now()

        input_power_range = np.arange(-80.0, -5.0, 0.25)

        test_frequency = channel_center_freqs[test_channel] - ch_bandwidth / 10.0 # Just so that I'm not in the centre of the bin
        print("Setting signal gen freq output to {} MHz for {} test.".format(signal_gen.setFrequency(test_frequency)/1e6, test_name))

        print("Setting ADC attenuation to {} dB for {} test.".format(test_attenuation / 2.0, test_name))
        receiver.katcp_request(katcprequest="setRoachADC0Attenuation", katcprequestArg="{:d}".format(test_attenuation))
        receiver.katcp_request(katcprequest="setRoachADC1Attenuation", katcprequestArg="{:d}".format(test_attenuation))

        print("Setting DSP gain to {} for this test.".format(test_gain))
        receiver.katcp_request(katcprequest="setRoachDspGain", katcprequestArg="{:d}".format(test_gain))

        print("Setting accumulation time to {} s for {} test.".format(test_accumulation, test_name))
        receiver.katcp_request(katcprequest="setRoachAccumulationLength", katcprequestArg="{:d}".format(test_accumulation*500000))

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
        print("Linearity test took {}.".format(end_time - start_time))

    if attenuator_test:
        test_name = "Attenuator"
        print("Testing built-in attenuators.")
        start_time = datetime.datetime.now()
        attenuator_range = np.arange(0, 63, 1)

        test_frequency = channel_center_freqs[test_channel] - ch_bandwidth / 10.0 # Just so that I'm not in the centre of the bin
        print("Setting signal gen freq output to {} MHz for the test.".format(signal_gen.setFrequency(test_frequency)/1e6))
        print("Setting signal gen power output to {} dBm for the test.".format(signal_gen.setPower(test_power)))

        print("Setting DSP gain to {} for this test.".format(test_gain))
        receiver.katcp_request(katcprequest="setRoachDspGain", katcprequestArg="{:d}".format(test_gain))

        print("Setting accumulation time to {} s for {} test.".format(test_accumulation, test_name))
        receiver.katcp_request(katcprequest="setRoachAccumulationLength", katcprequestArg="{:d}".format(test_accumulation*500000))

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

    if gain_test:
        test_name = "Gain"
        print("Testing digital gain.")
        start_time = datetime.datetime.now()
        test_channel = 500
        test_attenuation = 63
        gain_range = np.arange(0.15625, 4.0, 0.15625)

        test_frequency = channel_center_freqs[test_channel] - ch_bandwidth / 10.0 # Just so that I'm not in the centre of the bin
        print("Setting signal gen to {} MHz for {} test.".format(signal_gen.setFrequency(test_frequency)/1e6, test_name))
        print("Setting signal gen to {} dBm for {} test.".format(signal_gen.setPower(test_power, test_name)))

        print("Setting attenuation to {} dB for {} test.".format(test_attenuation / 2.0), test_name)
        receiver.katcp_request(katcprequest="setRoachADC0Attenuation", katcprequestArg="{:d}".format(test_attenuation))
        receiver.katcp_request(katcprequest="setRoachADC1Attenuation", katcprequestArg="{:d}".format(test_attenuation))

        print("Setting accumulation time to {} s for {} test.".format(test_accumulation, test_name))
        receiver.katcp_request(katcprequest="setRoachAccumulationLength", katcprequestArg="{:d}".format(test_accumulation*500000))

        output_powers = []

        for i, gain in enumerate(gain_range):
            print("Setting digital gain to {}. ({} of {})".format(gain, i + 1, len(gain_range)))
            receiver.katcp_request(katcprequest="setRoachDspGain", katcprequestArg="{:f}".format(gain))
            receiver.startCapture()
            time.sleep(1)
            dump = receiver.get_hdf5(stopCapture=True)
            spectrum = dump_to_spectrum(dump)
            output_powers.append(spectrum[test_channel])

        output_powers = np.array(output_powers)

        plt.figure(figsize=(12,10))
        plt.plot(gain_range, output_powers)
        plt.xlabel("Gain setting")
        plt.ylabel("Output power [dBa]")
        plt.title("Digital gain tested at {} MHz.".format(test_frequency/1e6))
        plt.grid()
        plt.savefig("gain.png")

        end_time = datetime.datetime.now()
        print("Digital gain test took {}.".format(end_time - start_time))

    if accumulation_test:
        test_name = "Accumulation"
        print("Testing Accumulation.")
        start_time = datetime.datetime.now()

        local_attenuation = 2

        #Signal gen not used for this test.
        #test_frequency = channel_center_freqs[test_channel] - ch_bandwidth / 10.0 # Just so that I'm not in the centre of the bin
        #print("Setting signal gen to {} MHz for {} test.".format(signal_gen.setFrequency(test_frequency)/1e6, test_name))
        #print("Setting signal gen to {} dBm for {} test.".format(signal_gen.setPower(test_power), test_name))

        print("Setting attenuation to {} dB for {} test.".format(local_attenuation / 2.0, test_name))
        receiver.katcp_request(katcprequest="setRoachADC0Attenuation", katcprequestArg="{:d}".format(local_attenuation))
        receiver.katcp_request(katcprequest="setRoachADC1Attenuation", katcprequestArg="{:d}".format(local_attenuation) )

        print("Setting DSP gain to {} for {} test.".format(test_gain, test_name))
        receiver.katcp_request(katcprequest="setRoachDspGain", katcprequestArg="{:f}".format(test_gain))

        output_powers = []
        accum_range = np.arange(0.08, 5.0, 0.08)

        for i, accum in enumerate(accum_range):
            print("Setting accumulation length to {}. ({} of {})".format(accum, i + 1, len(accum_range)))
            receiver.katcp_request(katcprequest="setRoachAccumulationLength", katcprequestArg="{:d}".format(int(accum*500000)))
            receiver.startCapture()
            time.sleep(1)
            dump = receiver.get_hdf5(stopCapture=True)
            spectrum = dump_to_spectrum(dump)
            output_powers.append(spectrum[test_channel])

        output_powers = np.array(output_powers)

        plt.figure(figsize=(12,10))
        plt.plot(accum_range, output_powers)
        plt.xlabel("Accumulation length setting [seconds]")
        plt.ylabel("Output power [dBa]")
        plt.title("Digital gain tested at channel {}.".format(test_channel))
        plt.grid()
        plt.savefig("accumulation.png")

        end_time = datetime.datetime.now()
        print("Digital gain test took {}.".format(end_time - start_time))




    #cleanup
    signal_gen.reset()

