import numpy as np
import multiprocessing as mp

class ParallelAmbiguity:
    
    def __init__(self):
        self.ambiguity_surface = None
    
    def compute_chunk(self, tau_range, s, t, fd):
        print(f"Computing chunk for fd = {fd}")
        ambiguity_slice = np.zeros(len(tau_range))

        for i, tau in enumerate(tau_range):
            # Time shift
            shifted_signal = np.roll(s, int(tau * len(t) / t[-1]))  
            
            # Frequency shift
            doppler_signal = np.exp(-1j * 2 * np.pi * fd * t) * shifted_signal 
            
            # Compute the ambiguity surface
            correlation = np.sum(np.conj(s) * doppler_signal)
            ambiguity_slice[i] = np.abs(correlation)

        print(f"Done computing chunk for fd = {fd}")
        return fd, ambiguity_slice

    def compute_ambiguity_surface_parallel(self, s, t, tau_range, fd_range):
        num_processes = mp.cpu_count()

        with mp.Pool(processes=num_processes) as pool:
            # Map the compute_chunk function to the fd_range with pool.map or pool.starmap
            results = pool.starmap(self.compute_chunk, [(tau_range, s, t, fd) for fd in fd_range])

        # Create ambiguity_surface array
        self.ambiguity_surface = np.zeros((len(fd_range), len(tau_range)))

        # Populate ambiguity_surface with results
        for fd, ambiguity_slice in results:
            idx = np.where(fd_range == fd)[0][0]  # Get the index of the current Doppler frequency in fd_range
            self.ambiguity_surface[idx, :] = ambiguity_slice

        return self.ambiguity_surface
