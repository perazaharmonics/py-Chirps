import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy import fftpack
from scipy.signal import chirp

import matplotlib
matplotlib.get_backend()
matplotlib.use('TkAgg')  # You can try different backends like 'Qt5Agg', 'GTK3Agg', etc.

def eigenwaveform(t, fc):
    return np.exp(1j*2*np.pi*fc*t)

def chirp_waveform(t, f0, f1, T):
    """
    Generate a linear chirp signal starting from frequency f0 and ending at f1 over duration T.
    """
    return signal.chirp(t, f0, T, f1, method='linear')

def quadratic_chirp(t, f0, f1, T):
    """
    Generate a quadratic chirp signal starting from frequency f0 and ending at f1 over duration T.
    """
    # beta is the chirpiness parameter
    beta = (f1 - f0) / (T**2)
    
    # Create the quadratic chirp signal
    phi_t = 2 * np.pi * (f0 * t + beta * (t**3) / 3)
    return np.cos(phi_t)

def exponential_chirp(t, f0, f1, T):
    """
    Generate an exponential chirp signal starting from frequency f0 and ending at f1 over duration T.
    """
    # Create the exponential chirp signal
    return chirp(t, f0, T, f1, method='quadratic', phi=-90)

def bpsk_modulate(data, t, carrier_frequency, modulation_index):
    carrier_wave = np.cos(2 * np.pi * carrier_frequency * t)
    bpsk_waveform = np.cos(2 * np.pi * carrier_frequency * t + np.pi * data * modulation_index)
    return bpsk_waveform
# Generate PRN code (for simplicity, we're using a random binary sequence)
def generate_prn_code(length):
    return np.random.choice([0, 1], size=(length,), p=[0.5, 0.5])

# Generate GPS-like signal
def gps_signal(t, carrier_frequency):
    # Generating a simple PRN code; real GPS uses specific sequences for each satellite
    prn_code = generate_prn_code(len(t))
    
    # Convert binary sequence to bipolar (-1 and 1) for BPSK modulation
    bpsk_data = 2*prn_code - 1
    
    # BPSK modulate the PRN code onto the carrier wave
    carrier_wave = np.cos(2 * np.pi * carrier_frequency * t)
    gps_waveform = carrier_wave * bpsk_data

    return gps_waveform

'''
def chirp_waveform_with_doppler(t, f0, f1, T, fd_std):
    """
    Generate a linear chirp signal starting from frequency f0 and ending at f1 over duration T,
    with a random frequency shift at each time step drawn from a normal distribution with standard deviation fd_std.
    """
    instantaneous_freq = np.linspace(f0, f1, len(t))
    freq_shift = np.random.normal(loc=0, scale=fd_std, size=len(t))
    freq = instantaneous_freq + freq_shift
    return signal.chirp(t, f0, T, f1, method='linear') * np.exp(1j * 2 * np.pi * freq * t) 
    '''


def compute_chunk(tau_range, s, t, fd):
    print("Computing chunk for fd = {}".format(fd))
    ambiguity_slice = np.zeros(len(tau_range))

    for i, tau in enumerate(tau_range):
        # Time shift
        shifted_signal = np.roll(s, int(tau * len(t) / t[-1]))  
        
        # Frequency shift
        doppler_signal = np.exp(-1j * 2 * np.pi * fd * t) * shifted_signal 
        
        # Compute the ambiguity surface
        correlation = np.sum(np.conj(s) * doppler_signal)
        ambiguity_slice[i] = np.abs(correlation)

    print("Done computing chunk for fd = {}".format(fd))
    return fd, ambiguity_slice

def compute_ambiguity_surface_parallel(s, t, tau_range, fd_range):
    num_processes = mp.cpu_count()

    with mp.Pool(processes=num_processes) as pool:
        # Map the compute_chunk function to the fd_range with pool.map or pool.starmap
        results = pool.starmap(compute_chunk, [(tau_range, s, t, fd) for fd in fd_range])

    # Create ambiguity_surface array
    ambiguity_surface = np.zeros((len(fd_range), len(tau_range)))

    # Populate ambiguity_surface with results
    for fd, ambiguity_slice in results:
        idx = np.where(fd_range == fd)[0][0]  # Get the index of the current Doppler frequency in fd_range
        ambiguity_surface[idx, :] = ambiguity_slice

    return ambiguity_surface

def plot_ambiguity_function_3d(s, t, fc, tau_range, fd_range):
    #s = eigenwaveform(t, fc)

    # using a linear chirp to see phase difference between the two XMT and its echo (Doppler Shifter version)
    # s = chirp_waveform(t, fc-50, fc+50, t[-1]-t[0])

    ambiguity_surface = compute_ambiguity_surface_parallel(s, t, tau_range, fd_range)
    tau_grid, fd_grid = np.meshgrid(tau_range, fd_range)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(tau_grid, fd_grid, np.abs(ambiguity_surface), cmap='viridis')  # Plot the magnitude in 3D
    ax1.set_xlabel('Delay (s)')
    ax1.set_ylabel('Doppler (Hz)')
    ax1.set_zlabel('Ambiguity function')
    ax1.set_title('Ambiguity function for fc = '+str(fc)+' Hz')

    # Plot the magnitude of the ambiguity function
    ax2 = fig.add_subplot(132)
    mag = np.abs(ambiguity_surface)
    extent = [tau_range[0], tau_range[-1], fd_range[0], fd_range[-1]]
    ax2.plot(tau_range, 20*np.log10(mag[:, 50]), label='Doppler = '+str(fd_range[50])+' Hz')
    ax2.set_xlabel('Delay (s)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Magnitude of ambiguity function')
    ax2.legend()

    # Extract and filter phase
    threshold = 0.05 * np.max(np.abs(ambiguity_surface))
    phase = np.angle(ambiguity_surface * (np.abs(ambiguity_surface) > threshold))
    
    # Plot the phase of the ambiguity function
    ax3 = fig.add_subplot(133)
    ax3.plot(tau_range, phase[:, 50], label='Doppler = '+str(fd_range[50])+' Hz')
    ax3.set_xlabel('Delay (s)')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_title('Phase of ambiguity function')
    ax3.legend()

    plt.tight_layout()
    plt.show()


def select_pulse():
    print("Please select the pulse type:")
    pulse_types = ["Eigenwaveform", "Linear Chirp", "Quadratic Chirp", "Exponential Chirp", "BPSK"]
    for i, pt in enumerate(pulse_types, start=1):
        print(f"{i}: {pt}")

    try:
        choice = int(input("Enter the number corresponding to your choice: "))
        assert 1 <= choice <= 5
        return choice
    except (ValueError, AssertionError):
        print("\nInvalid choice. Defaulting to 2: Linear Chirp.\n")
        return 2

def test_functions():
    fc = 700  # Carrier frequency
    fs = 4*fc  # Sampling frequency
    t = np.arange(-0.5, 0.5, 1/fs)  # Time vector
    tau_range = np.linspace(-0.2, 0.2, 100)  # Delay range
    fd_range = np.linspace(-200, 200, 100)  # Doppler range

    choice = select_pulse()

    # Signal generation based on user's choice
    signals = {
        1: eigenwaveform(t, fc),
        2: chirp_waveform(t, fc-50, fc+50, t[-1]-t[0]),
        3: quadratic_chirp(t, fc-50, fc+50, t[-1]-t[0]),
        4: exponential_chirp(t, fc-50, fc+50, t[-1]-t[0]),
        5: bpsk_modulate(np.random.randint(0, 2, len(t)), t, fc, .01)
    }

    s = signals.get(choice, signals[2])  # Default to Linear Chirp if invalid choice

    
    plot_ambiguity_function_3d(s, t, fc, tau_range, fd_range)

if __name__ == "__main__":
    test_functions()