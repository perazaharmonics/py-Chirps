import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy import fftpack
from scipy.signal import chirp


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

def compute_ambiguity_surface(s, t, tau_range, fd_range):
    """
    Compute the ambiguity function for discrete signal s over
    a range of time delays tau_range and Doppler shifts fd_range.
    """
    ambiguity_surface = np.empty((len(fd_range), len(tau_range)))

    for i, tau in enumerate(tau_range):
        for j, fd in enumerate(fd_range):
            # Create time-shifted and frequency-shifted version of the signal
            shifted_signal = np.roll(s, int(tau*len(t)/t[-1]))  # Time shift
            doppler_signal = np.exp(-1j * 2 * np.pi * fd * t) * shifted_signal  # Frequency shift

            # Correlate (multiply and sum) the signals
            correlation = np.sum(np.conj(s) * doppler_signal)
            ambiguity_surface[j, i] = np.abs(correlation)

    return ambiguity_surface

def plot_ambiguity_function_3d(s, t, fc, tau_range, fd_range):
    #s = eigenwaveform(t, fc)

    # using a linear chirp to see phase difference between the two XMT and its echo (Doppler Shifter version)
    # s = chirp_waveform(t, fc-50, fc+50, t[-1]-t[0])

    ambiguity_surface = compute_ambiguity_surface(s, t, tau_range, fd_range)
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
    print("1: Eigenwaveform")
    print("2: Linear Chirp")
    print("3: Quadratic Chrip")
    print("4: Exponential Chirp")
    print("5: BPSK")
    choice = input("Enter the number corresponding to your choice: ")
    return choice


# Test the functions
if __name__ == "__main__":
# Define the parameters

    fc = 700 # Carrier frequency
    fs = 4*fc # Sampling frequency
    tau_range = np.linspace(-0.2, 0.2, 100) # Delay range
    fd_range = np.linspace(-200, 200, 100) # Doppler range
    t = np.arange(-0.5, 0.5, 1/fs) # Time vector



    # Ask the user to choose a waveform
    choice = select_pulse()

    # Generate the chosen waveform
    if choice == '1':
        s = eigenwaveform(t, fc)
    elif choice == '2':
        s = chirp_waveform(t, fc-50, fc+50, t[-1]-t[0])
    elif choice == '3':
        s = quadratic_chirp(t, fc-50, fc+50, t[-1]-t[0])
    elif choice == '4':
        s = exponential_chirp(t, fc-50, fc+50, t[-1]-t[0])
    elif choice == '5':
        data = np.random.randint(0, 2, len(t))  # Random BPSK data
        modulation_index = .01  # Modify this as needed
        s = bpsk_modulate(data, t, fc, modulation_index)
    else:
        print("\n Invalid choice. Linear chirp selected as default. \n")
        s = chirp_waveform(t, fc-50, fc+50, t[-1]-t[0])
        

    # Plot the ambiguity function in 3D, magnitude, and phase
    plot_ambiguity_function_3d(s, t, fc, tau_range, fd_range)


