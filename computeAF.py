import numpy as np
import multiprocessing as mp

class ParallelAmbiguity:
    def __init__(self):
        self.AS = None

    def compute_chunk(self, tau_range, s, t, fd):
        AF_slice = np.zeros(len(tau_range))

        for i, tau in enumerate(tau_range):
            #Time shift
            sig_shift = np.roll(s, int(tau * len(t) / t[-1]))
            # Freq shift
            echo = np.exp(-1j * 2 * np.pi * fd * t) * sig_shift
            #Compute Ambiguity Surface
            correlation = np.sum(np.conj(s) * echo)
            AF_slice[i] = np.abs(correlation)
        print(f"Done computing chunk for fd = {fd}")
        return fd, AF_slice

    def compute_AS_parallel(self, s, t, tau_range, fd_range):
        num_process = mp.cpu_count()

        with mp.Pool(processes=num_process) as Pool:
            # Map the compute_chunk function to the fd_range with pool.map or pool.starmap
            results = Pool.starmap(self.compute_chunk, [(tau_range, s, t, fd) for fd in fd_range])

        # Create AS array
        self.AS = np.zeros((len(fd_range), len(tau_range)))

        
        # Populate ambiguity surface with data
        for fd, AF_slice in results:
            idx = np.where(fd_range==fd)[0][0] # Get the index of the current Doppler frequency in fd_range
            self.AS[idx, :] = AF_slice
            
        return self.AS
