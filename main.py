import os
import numpy as np
from models.hodgkin_huxley import HodgkinHuxley
from models.lif import LIFModel
from models.reservoir import ReservoirNetwork
import plotting.visualizer as viz

# --- CONFIGURATION ---
OUTPUT_DIR = "output"

def ensure_output_folder():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}/")

def run_hh_simulation():
    print("\n[1/3] Running Hodgkin-Huxley Simulation...")
    hh = HodgkinHuxley()
    t = np.linspace(0, 150, 3000) 
    
    def input_current(time):
        return 20.0 if 10.0 <= time <= 110.0 else 0.0

    results = hh.simulate(t, input_current)
    
    save_path = os.path.join(OUTPUT_DIR, "1_hodgkin_huxley_dashboard.png")
    viz.plot_hh_dashboard(results, stim_start=10, stim_end=110, save_path=save_path)

def run_lif_simulation():
    print("\n[2/3] Running LIF Simulation...")
    lif = LIFModel()
    t = np.arange(0, 100, 0.1)
    
    I = np.zeros_like(t)
    I[(t > 20) & (t < 40)] = 25
    I[(t > 60) & (t < 80)] = 25
    
    _, V, _ = lif.simulate(t, I)
    
    save_path = os.path.join(OUTPUT_DIR, "2_lif_simulation.png")
    viz.plot_lif_simulation(t, V, I, save_path=save_path)

def run_reservoir_simulation():
    print("\n[3/3] Running Reservoir Simulation (Mackey-Glass)...")
    # Generate Data
    def mackey_glass(n_samples=2000):
        x = np.zeros(n_samples)
        x[:20] = 1.2
        for t in range(20, n_samples-1):
            x[t+1] = x[t] + (0.2 * x[t-17]/(1 + x[t-17]**10) - 0.1*x[t])
        return (x - np.mean(x))/np.std(x)

    data = mackey_glass()
    train_len = 1500
    test_len = 400
    
    # Init Network
    esn = ReservoirNetwork(n_res=300)
    esn.initialize()
    
    # Train for Near Future (t+10)
    u_train = data[:train_len]
    target_near = data[10:train_len+10]
    esn.train(u_train, target_near)
    
    u_test = data[train_len:train_len+test_len]
    pred_near = esn.predict(u_test)
    truth_near = data[train_len+10 : train_len+test_len+10]
    
    # Re-Train for Far Future (t+100) (using same reservoir state for simplicity)
    target_far = data[100:train_len+100]
    esn.train(u_train, target_far)
    pred_far = esn.predict(u_test)
    truth_far = data[train_len+100 : train_len+test_len+100]

    save_path = os.path.join(OUTPUT_DIR, "3_reservoir_prediction.png")
    viz.plot_reservoir_prediction(truth_near, pred_near, truth_far, pred_far, save_path=save_path)

if __name__ == "__main__":
    ensure_output_folder()
    run_hh_simulation()
    run_lif_simulation()
    run_reservoir_simulation()
    print("\nAll simulations complete. Check the 'output/' folder for graphs.")