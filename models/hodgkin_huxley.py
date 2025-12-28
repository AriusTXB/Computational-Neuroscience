import numpy as np
from scipy.integrate import odeint

class HodgkinHuxley:
    def __init__(self):
        # Parameters
        self.C_m = 1.0
        self.g_Na = 120.0
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0
        self.E_K = -77.0
        self.E_L = -54.387

    # Rate functions
    def alpha_n(self, V): return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    def beta_n(self, V):  return 0.125 * np.exp(-(V + 65.0) / 80.0)
    def alpha_m(self, V): return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    def beta_m(self, V):  return 4.0 * np.exp(-(V + 65.0) / 18.0)
    def alpha_h(self, V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
    def beta_h(self, V):  return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def derivatives(self, y, t, I_func):
        V, n, m, h = y
        I_x = I_func(t)
        
        I_Na = self.g_Na * (m**3) * h * (V - self.E_Na)
        I_K  = self.g_K  * (n**4) * (V - self.E_K)
        I_L  = self.g_L  * (V - self.E_L)
        
        dVdt = (I_x - I_Na - I_K - I_L) / self.C_m
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        
        return [dVdt, dndt, dmdt, dhdt]

    def simulate(self, time_vector, input_current_func, V0=-65.0):
        # Initial steady states
        n0 = self.alpha_n(V0) / (self.alpha_n(V0) + self.beta_n(V0))
        m0 = self.alpha_m(V0) / (self.alpha_m(V0) + self.beta_m(V0))
        h0 = self.alpha_h(V0) / (self.alpha_h(V0) + self.beta_h(V0))
        y0 = [V0, n0, m0, h0]

        # Solve ODE
        sol = odeint(self.derivatives, y0, time_vector, args=(input_current_func,))
        
        # Extract results
        results = {
            't': time_vector,
            'V': sol[:, 0],
            'n': sol[:, 1],
            'm': sol[:, 2],
            'h': sol[:, 3]
        }
        
        # Calculate currents post-simulation for plotting
        results['I_Na'] = self.g_Na * (results['m']**3) * results['h'] * (results['V'] - self.E_Na)
        results['I_K']  = self.g_K  * (results['n']**4) * (results['V'] - self.E_K)
        results['I_L']  = self.g_L  * (results['V'] - self.E_L)
        results['I_ext'] = np.array([input_current_func(t) for t in time_vector])
        
        return results