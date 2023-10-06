import numpy as np
import time
from scipy import signal
from ParallelAmbiguity import ParallelAmbiguity as paramb
from PlotAmbiguity import PlotAmbiguity as pa
from Radar import Radar as rd

    # iterate through the set of waveform indexes
def select_pulse():
    print("Please select the pulse type:")
    pulse_types = ["Eigenwaveform", "Linear Chirp", "Quadratic Chirp", "Exponential Chirp", "BPSK", "Tone Ranger"]
    for i, pt in enumerate(pulse_types, start=1):
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

def test_functions():
    # Define the parameters
    fc = 1e6
    fs = 4*fc
    t = np.arange(-0.5, 0.5, 1/fs)  # Time vector
    tau_range = np.linspace(-0.2, 0.2, 100)  # Delay range
    fd_range = np.linspace(-200, 200, 100)  # Doppler range
    
    choice = select_pulse()

    # Initialize objects
    paramba = paramb()
    plotamb = pa()
    
    major_tone_frequency = 50e3  # 500 KHz major tone
    minor_tone_frequencies = [10e3, 20e3, 30e3, 40e3]  # Example minor tone frequencies
    
    # Signal generation based on user's choice
    signals = {
        1: rd.eigenwaveform(t, fc),
        2: rd.chirp_waveform(t, fc-50, fc+50, t[-1]-t[0]),
        3: rd.quadratic_chirp(t, fc-50, fc+50, t[-1]-t[0]),
        4: rd.exponential_chirp(t, fc-50, fc+50, t[-1]-t[0]),
        5: rd.bpsk_modulate(np.random.randint(0, 2, len(t)), t, fc, .01),
        6: rd.tone_ranger(t, fc, major_tone_frequency, minor_tone_frequencies)
    }

    start_time = time.time()
    s = signals.get(choice, signals[2])  # Default to Linear Chirp if invalid choice

    ambiguity_surface = paramba.compute_ambiguity_surface_parallel(s, t, tau_range, fd_range)
    plotamb.plot_ambiguity_function_3d(ambiguity_surface, t, fs, tau_range, fd_range)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    test_functions()