import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

def setup_style():
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks", {'axes.grid': True, 'grid.linestyle': ':'})
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'figure.dpi': 120,
        'axes.spines.right': False,
        'axes.spines.top': False
    })

def save_and_show(fig, save_path):
    if save_path:
        print(f"Saving plot to {save_path}...")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# (Keep plot_hh_dashboard and plot_lif_simulation functions as they are, no changes needed for them)

def plot_hh_dashboard(res, stim_start, stim_end, save_path=None):
    setup_style()
    
    # CHANGE 1: Layout - 2 Rows, 3 Columns (Widescreen look)
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], wspace=0.25, hspace=0.3)

    # --- Plot 1: Membrane Potential (Top Left) ---
    ax1 = fig.add_subplot(gs[0, 0])
    # CHANGE 2: Shaded Region for Stimulus instead of lines
    ax1.axvspan(stim_start, stim_end, color='#e0e0e0', alpha=0.5, label='Stimulus Active')
    ax1.plot(res['t'], res['V'], color='#333333', linewidth=1.5, label='V_m')
    ax1.set_title('Membrane Potential', fontweight='bold', color='#444444')
    ax1.set_ylabel('Voltage (mV)')
    ax1.legend(loc='upper right', frameon=True, fontsize=10)

    # --- Plot 2: Gating Variables (Top Middle) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axvspan(stim_start, stim_end, color='#e0e0e0', alpha=0.5)
    # CHANGE 3: Custom Color Palette (Teal, Purple, Gold)
    ax2.plot(res['t'], res['m'], color='#2ca02c', linestyle='-', linewidth=2, label='m (Na act)')
    ax2.plot(res['t'], res['h'], color='#bcbd22', linestyle='--', linewidth=2, label='h (Na inact)')
    ax2.plot(res['t'], res['n'], color='#9467bd', linestyle='-.', linewidth=2, label='n (K act)')
    ax2.set_title('Gating Dynamics', fontweight='bold', color='#444444')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='center right', fontsize=10)

    # --- Plot 3: Phase Plane (Top Right) ---
    ax3 = fig.add_subplot(gs[0, 2])
    # CHANGE 4: 'Inferno' colormap and smaller dots for a smoother look
    scatter = ax3.scatter(res['V'], res['I_Na'], c=res['t'], cmap='inferno', s=5, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Time (ms)')
    ax3.set_title('Phase Space ($V$ vs $I_{Na}$)', fontweight='bold', color='#444444')
    ax3.set_xlabel('Voltage (mV)')
    ax3.set_ylabel('Na Current')

    # --- Plot 4: Sodium Current (Bottom Left) ---
    ax4 = fig.add_subplot(gs[1, 0])
    # CHANGE 5: Fill Between for currents (visual weight)
    ax4.plot(res['t'], res['I_Na'], color='#1f77b4', linewidth=1)
    ax4.fill_between(res['t'], res['I_Na'], 0, color='#1f77b4', alpha=0.3)
    ax4.set_title('Sodium Current ($I_{Na}$)', fontweight='bold', color='#444444')
    ax4.set_ylabel('$\mu A/cm^2$')
    ax4.set_xlabel('Time (ms)')

    # --- Plot 5: Potassium Current (Bottom Middle) ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(res['t'], res['I_K'], color='#ff7f0e', linewidth=1)
    ax5.fill_between(res['t'], res['I_K'], 0, color='#ff7f0e', alpha=0.3)
    ax5.set_title('Potassium Current ($I_{K}$)', fontweight='bold', color='#444444')
    ax5.set_xlabel('Time (ms)')

    # --- Plot 6: Combined Currents (Bottom Right) ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(res['t'], res['I_Na'], color='#1f77b4', alpha=0.6, label='$I_{Na}$')
    ax6.plot(res['t'], res['I_K'], color='#ff7f0e', alpha=0.6, label='$I_{K}$')
    ax6.plot(res['t'], res['I_L'], color='gray', alpha=0.4, label='$I_{Leak}$')
    ax6.plot(res['t'], res['I_ext'], color='black', linestyle=':', label='$I_{Input}$')
    ax6.set_title('Total Current Profile', fontweight='bold', color='#444444')
    ax6.set_xlabel('Time (ms)')
    ax6.legend(loc='upper right', fontsize=9)

    plt.suptitle("Hodgkin-Huxley Dynamics Simulation", fontsize=16, y=0.95)
    save_and_show(fig, save_path)

def plot_lif_simulation(t, V, I, save_path=None):
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Use dual axis to show Current and Voltage cleanly without scaling hacks
    ax2 = ax.twinx()
    
    # Plot Input Current as a filled area on secondary axis
    ax2.fill_between(t, I, 0, color='gray', alpha=0.2, step='mid', label='Input Current')
    ax2.set_ylabel('Input Current (nA)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, max(I)*3) # Push it to the bottom
    
    # Plot Voltage on primary axis
    ax.plot(t, V, color='#d62728', linewidth=2, label='Membrane Potential')
    ax.axhline(-50, color='black', linestyle='--', linewidth=1, label='Threshold')
    
    ax.set_title("LIF Model Response", fontweight='bold')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    
    # Combine legends
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    save_and_show(fig, save_path)


def plot_reservoir_prediction(truth_near, pred_near, truth_far, pred_far, save_path=None):
    setup_style()
    # Key Change 1: sharey=True ensures both subplots have the same Y-axis range
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True) 
    
    # --- Near Future Plot ---
    # Key Change 2: Thicker, darker ground truth line
    axes[0].plot(truth_near, color='#444444', alpha=0.7, linewidth=3, label='Actual (Ground Truth)')
    # Key Change 3: Clear prediction line with smaller dots to emphasize individual points
    axes[0].plot(pred_near, color='#1f77b4', linewidth=1.5, label='Predicted (ESN)')
    axes[0].scatter(np.arange(len(pred_near)), pred_near, color='#1f77b4', s=10, alpha=0.5, zorder=5) # Scatter for prediction points
    
    axes[0].set_title('Near-Future Prediction (x(t+10))', fontweight='bold')
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel("Normalized Signal Amplitude")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6) # Lighter grid

    # --- Far Future Plot ---
    # Key Change 4: Thicker, darker ground truth line
    axes[1].plot(truth_far, color='#444444', alpha=0.7, linewidth=3, label='Actual (Ground Truth)')
    # Key Change 5: Clear prediction line with smaller dots
    axes[1].plot(pred_far, color='#d62728', linewidth=1.5, label='Predicted (ESN)')
    axes[1].scatter(np.arange(len(pred_far)), pred_far, color='#d62728', s=10, alpha=0.5, zorder=5) # Scatter for prediction points

    axes[1].set_title('Far-Future Prediction (x(t+100))', fontweight='bold')
    axes[1].set_xlabel("Time Steps")
    # Y-label is not needed here because sharey=True
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6) # Lighter grid
    
    plt.suptitle("Reservoir Computing: Mackey-Glass Time Series Prediction", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    save_and_show(fig, save_path)