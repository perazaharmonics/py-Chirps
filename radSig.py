import numpy as np
from scipy import signal

class Pulses:
    @staticmethod
    def eigenwaveform(t, fc):
        return np.exp(1j*2*np.pi*fc*t)
        
    @staticmethod
    def chirp(t, f0, f1, T):
        return signal.chirp(t,f0,T,f1, method = 'linear')
        
    @staticmethod
    def quad_chirp(t, f0, f1, T):
        # chirpness parameter
        beta = (f1 - f0) / (T**2)
        # the quadratic chirp signal
        phi_t = 2 * np.pi * (f0 * t + beta * (t**3) / 3)
        return np.cos(phi_t)

    @staticmethod
    def exp_chirp(t, f0, f1, T):
        return signal.chirp(t, f0, T, f1, method = 'quadratic', phi = -90)

    @staticmethod
    def hyp_chirp(t, f0, f1, T):
        return signal.chirp(t,f0,T,f1, method = 'hyperbolic')
