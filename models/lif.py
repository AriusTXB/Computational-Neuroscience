import numpy as np

class LIFModel:
    def __init__(self, tau_m=10.0, R_m=1.0, V_thresh=-50.0, V_reset=-70.0, V_rest=-70.0):
        self.tau_m = tau_m
        self.R_m = R_m
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.V_rest = V_rest

    def simulate(self, time_vector, input_current_array):
        dt = time_vector[1] - time_vector[0]
        V = np.zeros_like(time_vector)
        V[0] = self.V_rest
        spikes = []

        for i in range(1, len(time_vector)):
            dV = (-(V[i-1] - self.V_rest) + self.R_m * input_current_array[i-1]) / self.tau_m
            V[i] = V[i-1] + dV * dt
            
            if V[i] >= self.V_thresh:
                V[i] = self.V_reset
                spikes.append(time_vector[i])
                
        return time_vector, V, spikes