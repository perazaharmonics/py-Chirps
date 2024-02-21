import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class plotAS:
    def __init__(self):
        pass
    def plot_AS(self, AS, t, fc, tau_range, fd_range):
        tau_grid, fd_grid = np.meshgrid(tau_range, fd_range)
        fig = plt.figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(tau_grid, fd_grid, np.abs(AS), cmap='viridis')
        ax1.set_xlabel('Delay (s)')
        ax1.set_ylabel('Doppler (Hz)')
        ax1.set_zlabel('Ambiguity function')
        ax1.set_title('Ambiguity function for fc = '+str(fc)+' Hz')

        ax2 = fig.add_subplot(132)
        mag = np.abs(AS)
        ax2.plot(tau_range, 20*np.log10(mag[:, 50]), label='Doppler = '+str(fd_range[50])+' Hz')
        ax2.set_xlabel('Delay (s)')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_title('Magnitude of ambiguity function')
        ax2.legend()

        threshold = 0.05 * np.max(np.abs(AS))
        phase = np.angle(AS * (np.abs(AS) > threshold))

        ax3 = fig.add_subplot(133)
        ax3.plot(tau_range, phase[:, 50], label='Doppler = '+str(fd_range[50])+' Hz')
        ax3.set_xlabel('Delay (s)')
        ax3.set_ylabel('Phase (rad)')
        ax3.set_title('Phase of ambiguity function')
        ax3.legend()

        plt.tight_layout()
        plt.show()
