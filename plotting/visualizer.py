import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})

def save_and_show(fig, save_path):
    if save_path:
        print(f"Saving plot to {save_path}...")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_hh_dashboard(res, stim_start, stim_end, save_path=None):
    setup_style()
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)

    # 1. Membrane Potential
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(res['t'], res['V'], 'b-', linewidth=2, label='Membrane Potential')
    ax1.axvline(stim_start, color='r', linestyle='--', alpha=0.5, label='Stimulus On')
    ax1.axvline(stim_end, color='r', linestyle='--', alpha=0.5, label='Stimulus Off')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Hodgkin-Huxley: Action Potential', fontweight='bold')
    ax1.set_ylabel('Potential (mV)')
    ax1.legend(loc='upper right')

    # 2. Sodium Current
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(res['t'], res['I_Na'], 'r-', linewidth=2)
    ax2.axvline(stim_start, color='g', linestyle='--', alpha=0.5)
    ax2.axvline(stim_end, color='g', linestyle='--', alpha=0.5)
    ax2.set_title('Sodium Current ($I_{Na}$) - Depolarization Phase', fontweight='bold')
    ax2.set_ylabel('Current density ($\mu A/cm^2$)')

    # 3. Potassium Current
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(res['t'], res['I_K'], 'g-', linewidth=2)
    ax3.fill_between(res['t'], res['I_K'], 0, color='green', alpha=0.1)
    ax3.axvline(stim_start, color='r', linestyle='--', alpha=0.5)
    ax3.axvline(stim_end, color='r', linestyle='--', alpha=0.5)
    ax3.set_title('Potassium Current ($I_{K}$) - Repolarization Phase', fontweight='bold')
    ax3.set_ylabel('Current density ($\mu A/cm^2$)')

    # 4. Gating Variables
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(res['t'], res['m'], 'r-', label='m (Na activation)')
    ax4.plot(res['t'], res['h'], 'orange', label='h (Na inactivation)')
    ax4.plot(res['t'], res['n'], 'g-', label='n (K activation)')
    ax4.axvline(stim_start, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(stim_end, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('Gating Variables Dynamics', fontweight='bold')
    ax4.set_ylabel('Probability')
    ax4.legend()

    # 5. All Currents Combined
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(res['t'], res['I_ext'], 'k-', linewidth=2, label='External Input')
    ax5.plot(res['t'], res['I_Na'], 'r-', linewidth=1, label='$I_{Na}$')
    ax5.plot(res['t'], res['I_K'], 'g-', linewidth=1, label='$I_{K}$')
    ax5.plot(res['t'], res['I_L'], 'b-', linewidth=1, label='$I_{L}$')
    ax5.set_title('All Currents vs Time', fontweight='bold')
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Current density')
    ax5.legend(loc='upper right')

    # 6. Phase Plane (V vs INa) colored by Time
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(res['V'], res['I_Na'], c=res['t'], cmap='viridis', s=10, alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Time (ms)')
    ax6.set_title('Phase Plane: V vs $I_{Na}$ (colored by time)', fontweight='bold')
    ax6.set_xlabel('Membrane Potential (mV)')
    ax6.set_ylabel('Sodium Current')

    plt.tight_layout()
    save_and_show(fig, save_path)

def plot_lif_simulation(t, V, I, save_path=None):
    setup_style()
    fig = plt.figure(figsize=(12, 6))
    
    plt.plot(t, V, label='Membrane Potential', color='b')
    plt.plot(t, (I*5) - 70, 'r--', alpha=0.5, label='Scaled Input Current')
    plt.axhline(-50, color='r', linestyle=':', label='Threshold')
    
    plt.title("LIF Model Simulation", fontweight='bold')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid(True)
    
    save_and_show(fig, save_path)

def plot_reservoir_prediction(truth_near, pred_near, truth_far, pred_far, save_path=None):
    setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    
    # Near Future
    ax1.plot(truth_near, 'k', label='Ground Truth', alpha=0.6)
    ax1.plot(pred_near, 'b--', label='ESN Prediction')
    ax1.set_title('Reservoir: Near-Future Prediction (t+10)', fontweight='bold')
    ax1.legend()
    
    # Far Future
    ax2.plot(truth_far, 'k', label='Ground Truth', alpha=0.6)
    ax2.plot(pred_far, 'r--', label='ESN Prediction')
    ax2.set_title('Reservoir: Far-Future Prediction (t+100)', fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    save_and_show(fig, save_path)