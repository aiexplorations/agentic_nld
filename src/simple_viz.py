"""
Simple focused visualizations for key dynamical system metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def create_key_metrics_plot(results1: Dict[str, Any], results2: Dict[str, Any] = None, 
                           save_path: str = "key_metrics.png") -> None:
    """
    Create a focused 4-panel plot showing the most important metrics.
    
    Args:
        results1: Results from first conversation
        results2: Results from second conversation (for perturbation analysis)
        save_path: Path to save the figure
    """
    
    # Setup figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Two-Agent Dynamical System: Key Metrics", fontsize=16, fontweight='bold')
    
    # Extract trajectories
    traj_a1 = results1['agent_a_trajectory']
    traj_b1 = results1['agent_b_trajectory']
    
    # 1. State Evolution (Top Left)
    ax1 = axes[0, 0]
    plot_state_evolution(ax1, traj_a1, traj_b1)
    
    # 2. Phase Space (Top Right)
    ax2 = axes[0, 1]
    plot_phase_space(ax2, traj_a1, traj_b1)
    
    # 3. State Divergence (Bottom Left)
    ax3 = axes[1, 0]
    plot_state_divergence(ax3, traj_a1, traj_b1)
    
    # 4. Perturbation Analysis or Chaos Indicators (Bottom Right)
    ax4 = axes[1, 1]
    if results2:
        traj_a2 = results2['agent_a_trajectory']
        traj_b2 = results2['agent_b_trajectory']
        plot_perturbation_analysis(ax4, traj_a1, traj_b1, traj_a2, traj_b2)
    else:
        plot_chaos_indicators(ax4, traj_a1, traj_b1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Key metrics plot saved as {save_path}")
    plt.show()


def plot_state_evolution(ax, traj_a, traj_b):
    """Plot state vector magnitudes over time"""
    time_steps = np.arange(len(traj_a))
    
    # Calculate magnitudes
    mag_a = np.linalg.norm(traj_a, axis=1)
    mag_b = np.linalg.norm(traj_b, axis=1)
    
    ax.plot(time_steps, mag_a, 'o-', color='#1f77b4', label='Agent A', linewidth=2, markersize=6)
    ax.plot(time_steps, mag_b, 's-', color='#ff7f0e', label='Agent B', linewidth=2, markersize=6)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('State Vector Magnitude')
    ax.set_title('State Evolution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_phase_space(ax, traj_a, traj_b):
    """Plot 2D phase space projection with enhanced features"""
    # Use first two dimensions
    ax.plot(traj_a[:, 0], traj_a[:, 1], 'o-', color='#1f77b4', 
            label='Agent A', alpha=0.8, linewidth=2, markersize=5)
    ax.plot(traj_b[:, 0], traj_b[:, 1], 's-', color='#ff7f0e', 
            label='Agent B', alpha=0.8, linewidth=2, markersize=5)
    
    # Mark start and end points with different symbols
    ax.scatter(traj_a[0, 0], traj_a[0, 1], color='#1f77b4', s=150, 
              marker='o', edgecolor='black', linewidth=2, zorder=5, label='A Start')
    ax.scatter(traj_a[-1, 0], traj_a[-1, 1], color='#1f77b4', s=150, 
              marker='X', edgecolor='black', linewidth=2, zorder=5, label='A End')
    
    ax.scatter(traj_b[0, 0], traj_b[0, 1], color='#ff7f0e', s=150, 
              marker='s', edgecolor='black', linewidth=2, zorder=5, label='B Start')
    ax.scatter(traj_b[-1, 0], traj_b[-1, 1], color='#ff7f0e', s=150, 
              marker='X', edgecolor='black', linewidth=2, zorder=5, label='B End')
    
    # Add direction arrows
    if len(traj_a) > 1:
        for i in range(0, len(traj_a)-1, max(1, len(traj_a)//3)):
            dx_a = traj_a[i+1, 0] - traj_a[i, 0]
            dy_a = traj_a[i+1, 1] - traj_a[i, 1]
            ax.arrow(traj_a[i, 0], traj_a[i, 1], dx_a*0.3, dy_a*0.3, 
                    head_width=0.02, head_length=0.02, fc='#1f77b4', ec='#1f77b4', alpha=0.6)
            
            dx_b = traj_b[i+1, 0] - traj_b[i, 0]
            dy_b = traj_b[i+1, 1] - traj_b[i, 1]
            ax.arrow(traj_b[i, 0], traj_b[i, 1], dx_b*0.3, dy_b*0.3, 
                    head_width=0.02, head_length=0.02, fc='#ff7f0e', ec='#ff7f0e', alpha=0.6)
    
    ax.set_xlabel('State Dimension 1')
    ax.set_ylabel('State Dimension 2')
    ax.set_title('Phase Space Trajectory')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def plot_state_divergence(ax, traj_a, traj_b):
    """Plot divergence between agent states with trend analysis"""
    time_steps = np.arange(len(traj_a))
    divergence = np.linalg.norm(traj_a - traj_b, axis=1)
    
    ax.plot(time_steps, divergence, 'o-', color='#d62728', linewidth=2, markersize=6)
    ax.fill_between(time_steps, divergence, alpha=0.3, color='#d62728')
    
    # Add trend line
    if len(time_steps) > 1:
        z = np.polyfit(time_steps, divergence, 1)
        p = np.poly1d(z)
        ax.plot(time_steps, p(time_steps), "--", alpha=0.8, color='black', linewidth=2,
               label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        
        # Add interpretation
        if z[0] > 0.01:
            trend_text = "Diverging"
            trend_color = 'red'
        elif z[0] < -0.01:
            trend_text = "Converging"
            trend_color = 'green'
        else:
            trend_text = "Stable"
            trend_color = 'blue'
        
        ax.text(0.02, 0.98, f'System: {trend_text}', transform=ax.transAxes, 
               verticalalignment='top', fontweight='bold', color=trend_color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('State Divergence')
    ax.set_title('Agent State Divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_perturbation_analysis(ax, traj_a1, traj_b1, traj_a2, traj_b2):
    """Compare trajectories from perturbed initial conditions"""
    time_steps = np.arange(min(len(traj_a1), len(traj_a2)))
    
    # Calculate magnitudes
    mag_a1 = np.linalg.norm(traj_a1[:len(time_steps)], axis=1)
    mag_a2 = np.linalg.norm(traj_a2[:len(time_steps)], axis=1)
    mag_b1 = np.linalg.norm(traj_b1[:len(time_steps)], axis=1)
    mag_b2 = np.linalg.norm(traj_b2[:len(time_steps)], axis=1)
    
    # Plot trajectories
    ax.plot(time_steps, mag_a1, '-', color='#1f77b4', linewidth=2, label='A - Run 1')
    ax.plot(time_steps, mag_a2, '--', color='#1f77b4', linewidth=2, alpha=0.7, label='A - Run 2')
    ax.plot(time_steps, mag_b1, '-', color='#ff7f0e', linewidth=2, label='B - Run 1')
    ax.plot(time_steps, mag_b2, '--', color='#ff7f0e', linewidth=2, alpha=0.7, label='B - Run 2')
    
    # Calculate final divergence
    final_div_a = abs(mag_a1[-1] - mag_a2[-1]) if len(mag_a1) > 0 else 0
    final_div_b = abs(mag_b1[-1] - mag_b2[-1]) if len(mag_b1) > 0 else 0
    
    # Add sensitivity indicator
    sensitivity_score = (final_div_a + final_div_b) / 2
    if sensitivity_score > 0.5:
        sensitivity_text = "HIGH SENSITIVITY"
        sensitivity_color = 'red'
    elif sensitivity_score > 0.2:
        sensitivity_text = "MODERATE SENSITIVITY"
        sensitivity_color = 'orange'
    else:
        sensitivity_text = "LOW SENSITIVITY"
        sensitivity_color = 'green'
    
    ax.text(0.02, 0.98, sensitivity_text, transform=ax.transAxes, 
           verticalalignment='top', fontweight='bold', color=sensitivity_color,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(0.02, 0.88, f'Final Divergence: {sensitivity_score:.3f}', transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('State Magnitude')
    ax.set_title('Perturbation Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_chaos_indicators(ax, traj_a, traj_b):
    """Plot indicators of chaotic behavior"""
    # Calculate local divergence rates
    if len(traj_a) < 3:
        ax.text(0.5, 0.5, 'Insufficient data\nfor chaos analysis', 
               ha='center', va='center', fontsize=12)
        ax.set_title('Chaos Indicators')
        return
    
    # Simple chaos indicators
    window_size = min(3, len(traj_a) - 1)
    local_lyapunov = []
    time_points = []
    
    for i in range(len(traj_a) - window_size):
        # Local state differences for Agent A
        segment_a = traj_a[i:i + window_size]
        differences = np.linalg.norm(np.diff(segment_a, axis=0), axis=1)
        
        if np.all(differences > 1e-12):
            log_diff = np.log(differences + 1e-12)
            local_lyap = np.mean(np.diff(log_diff))
            local_lyapunov.append(local_lyap)
            time_points.append(i + window_size/2)
    
    if local_lyapunov:
        ax.plot(time_points, local_lyapunov, 'o-', color='#2ca02c', linewidth=2, markersize=5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Chaos Threshold')
        
        # Highlight chaotic regions
        chaotic_mask = np.array(local_lyapunov) > 0
        if np.any(chaotic_mask):
            chaotic_points = np.array(time_points)[chaotic_mask]
            chaotic_values = np.array(local_lyapunov)[chaotic_mask]
            ax.scatter(chaotic_points, chaotic_values, color='red', s=100, 
                      alpha=0.7, marker='*', label='Chaotic Behavior', zorder=5)
        
        # Add chaos assessment
        avg_lyapunov = np.mean(local_lyapunov)
        chaos_percentage = np.sum(chaotic_mask) / len(local_lyapunov) * 100
        
        if avg_lyapunov > 0:
            chaos_text = "CHAOTIC SYSTEM"
            chaos_color = 'red'
        elif chaos_percentage > 30:
            chaos_text = "INTERMITTENT CHAOS"
            chaos_color = 'orange'
        else:
            chaos_text = "STABLE SYSTEM"
            chaos_color = 'green'
        
        ax.text(0.02, 0.98, chaos_text, transform=ax.transAxes, 
               verticalalignment='top', fontweight='bold', color=chaos_color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.text(0.02, 0.88, f'Chaos: {chaos_percentage:.1f}% of time', transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Local Lyapunov Exponent')
    ax.set_title('Chaos Indicators')
    ax.legend()
    ax.grid(True, alpha=0.3)