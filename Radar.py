import numpy as np
from scipy import signal

class Radar:
    
    @staticmethod
    def eigenwaveform(t, fc):
        return np.exp(1j*2*np.pi*fc*t)

    @staticmethod
    def chirp_waveform(t, f0, f1, T):
        """
        Generate a linear chirp signal starting from frequency f0 and ending at f1 over duration T.
        """
        return signal.chirp(t, f0, T, f1, method='linear')

    @staticmethod
    def quadratic_chirp(t, f0, f1, T):
        """
        Generate a quadratic chirp signal starting from frequency f0 and ending at f1 over duration T.
        """
        # beta is the chirpiness parameter
        beta = (f1 - f0) / (T**2)
        
        # Create the quadratic chirp signal
        phi_t = 2 * np.pi * (f0 * t + beta * (t**3) / 3)
        return np.cos(phi_t)

    @staticmethod
    def exponential_chirp(t, f0, f1, T):
        """
        Generate an exponential chirp signal starting from frequency f0 and ending at f1 over duration T.
        """
        # Create the exponential chirp signal
        return signal.chirp(t, f0, T, f1, method='quadratic', phi=-90)

    @staticmethod
    def bpsk_modulate(data, t, carrier_frequency, modulation_index):
        carrier_wave = np.cos(2 * np.pi * carrier_frequency * t)
        bpsk_waveform = np.cos(2 * np.pi * carrier_frequency * t + np.pi * data * modulation_index)
        return bpsk_waveform

    @staticmethod
    def generate_prn_code(length):
        return np.random.choice([0, 1], size=(length,), p=[0.5, 0.5])

    @staticmethod
    def gps_signal(t, carrier_frequency):
        # Generating a simple PRN code; real GPS uses specific sequences for each satellite
        prn_code = Radar.generate_prn_code(len(t))
        
        # Convert binary sequence to bipolar (-1 and 1) for BPSK modulation
        bpsk_data = 2*prn_code - 1
        
        # BPSK modulate the PRN code onto the carrier wave
        carrier_wave = np.cos(2 * np.pi * carrier_frequency * t)
        gps_waveform = carrier_wave * bpsk_data

        return gps_waveform

    @staticmethod
    def tone_ranger(t, carrier_frequency, major_tone_frequency, minor_tone_frequencies):
        # BPSK modulation function
        def bpsk_modulated_tone(t, frequency, data):
            return (2*data - 1) * np.cos(2 * np.pi * (carrier_frequency + frequency) * t)
        
        # Generate BPSK data
        data = np.random.choice([0, 1], size=(len(t),))
        """TODO: Use pure single tones with symmetric freq-spacing and
         decreasing amplitudes. Center the major tone at the center frequency."""
        # Modulate the major tone
        # major_tone = bpsk_modulated_tone(t, major_tone_frequency, data)
        
        # Modulate the minor tones
        minor_tones = sum(bpsk_modulated_tone(t, freq, data) for freq in minor_tone_frequencies)
        
        # Combine the major and minor tones
        composite_signal = major_tone + minor_tones
        
        return composite_signal