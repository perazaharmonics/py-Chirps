import numpy as np
import time
from scipy import signal
from computeAF import ParallelAmbiguity
from plotAF import plotAS
from radsig import Pulses

    # iterate through the set of waveform indexes
def pulse_select():
    print("Please select the pulse type:", "/n")
    pulses = ["Eigenwaveform", "Linear Chirp", "Quadratic Chirp", "Exponential Chirp", "Hyperbolic Chirp"]
    for i, pt in enumerate(pulses, start=1):
        print(f"{i}: {pt}")
     # Mediate user input

    try:
        choice = int(input("Enter the number corresponding to your choice: "))
        assert 1 <= choice <= 6
        return choice
    # Catch the error if the user enters a non-integer value
    except (ValueError, AssertionError):
        print("\nInvalid choice. Defaulting to 2: Linear Chirp.\n")
        return 2

def testing():
    # Define the parameters
    fc = 65535
    fs = 4*fc
    t = np.arange(-0.5, 0.5, 1/fs)  # Time vector
    tau_range = np.linspace(-0.2, 0.2, 100)  # Delay range
    fd_range = np.linspace(-200, 200, 100)  # Doppler range

    # Get User input
    choice = pulse_select()

    # Init Objects
    AF_compute = ParallelAmbiguity()
    start = time.time()

    AF_plot = plotAS()

    signals = {
        1: Pulses.eigenwaveform(t, fc),
        2: Pulses.chirp(t, fc-50, fc+50, t[-1]-t[0]),
        3: Pulses.quad_chirp(t, fc-50, fc+50, t[-1]-t[0]),
        4: Pulses.exp_chirp(t, fc-50, fc+50, t[-1]-t[0]),
        5: Pulses.hyp_chirp(t, fc-50, fc+50, t[-1]-t[0])
        
    }

    s = signals.get(choice, signals[2]) # Default to linear chirp if None chosen

    start_time = time.time()
    AS = AF_compute.compute_AS_parallel(s, t, tau_range, fd_range)
    AF_plot.plot_AS(AS, t, fs, tau_range, fd_range)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    testing()
