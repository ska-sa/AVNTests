class TestProcedure:
    """Test Procedures"""

    @property
    def TBD(self):
        _description = """
        **TBD**
        """
        return _description

    @property
    def GainCorr(self):
        _description = """
        **Gain Correction**

        1. Configure a digitiser simulator to generate correlated input noise signal.
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Randomly select an input to test.
            - Note: Gains are relative to reference channels, and are increased iteratively until output power is increased by more than 6dB.
        5. Set gain correction on selected input to default
            - Confirm the gain has been set
        6. Iteratively request gain correction on one input on a single channel, single polarisation.
            - Confirm output power increased by less than 1 dB with a random gain increment [Dependent on mode].
            - Until the output power is increased by more than 6 dB
        """
        return _description


    @property
    def VectorAcc(self):
        _description = """
        **Vector Accumulator**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate generate continuous wave
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Select a test input and frequency channel
        6. Compile a list of accumulation periods to test
        7. Set gain correction on selected input via CAM interface.
        8. Configure a digitiser simulator to generate periodic wave in order for each FFT to be identical.
            - Check that the spectrum is zero except in the test channel
            - Confirm FFT Window samples, Internal Accumulations, VACC accumulation
        9. Retrieve quantiser snapshot of the selected input via CAM Interface
        10. Iteratively set accumulation length and confirm if the right accumulation is set on the SPEAD accumulation,

            - Confirm that vacc length was set successfully, and equates to a specific accumulation time as per calculation
            - Check that the accumulator actual response is equal to the expected response for the accumulation length
        """
        return _description

    @property
    def Channelisation(self):
        _description = """
        **Channelisation Wideband Coarse/Fine L-band**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate continuous wave
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Calculate number of frequencies to iterate on
        6. Randomly select a frequency channel to test.
        7. Capture an initial correlator SPEAD accumulation and,
            - Determine the number of frequency channels
            - Confirm that the number of channels in the SPEAD accumulation, is equal to the number of frequency channels as calculated
            - Confirm that the Channelise total bandwidth is >= 770000000.0Hz.
            - Confirm the number of calculated channel frequency step is within requirement.
            - Verify that the calculated channel frequency step size is within requirement
            - Confirm the channelisation spacing and confirm that it is within the maximum tolerance.
        8. Sweep the digitiser simulator over the centre frequencies of at least all the channels that fall within the complete L-band
            - Capture channel response for every frequency channel in the selected frequencies calculated
        9. Check FFT overflow and QDR errors after channelisation.
        10. Check that the peak channeliser response to input frequencies in central 80% of the test
            channel frequency band are all in the test channel.
        11. Check that VACC output is at < 99% of maximum value, if fails then it is probably over-ranging.
        12. Check that ripple within 80% of cut-off frequency channel is < 1.5 dB
        13. Measure the power difference between the middle of the center and the middle of the next adjacent bins and confirm that is > -53dB
        14. Check that response at channel-edges are -3 dB relative to the channel centre at selected freq, actual source frequency
        15. Check that relative response at the low band-edge is within the range of -6 +- 1% relative to channel centre response.
        16. Check that relative response at the high band-edge is within the range of -6 +- 1% relative to channel centre response.
        """
        return _description


TestProcedure = TestProcedure()
