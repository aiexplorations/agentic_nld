"""
Visualization tools for two-agent dynamical system analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Any, Tuple
from scipy.stats import entropy
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DynamicalSystemVisualizer:
    """
    Comprehensive visualization tools for two-agent dynamical system analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 12)):
        self.figsize = figsize
        self.colors = {
            'agent_a': '#1f77b4',  # Blue
            'agent_b': '#ff7f0e',  # Orange  
            'system': '#2ca02c',   # Green
            'divergence': '#d62728', # Red
            'noise': '#9467bd',    # Purple
            'signal': '#17becf'    # Cyan
        }
    
    def create_comprehensive_analysis(
        self, 
        results1: Dict[str, Any], 
        results2: Dict[str, Any] = None,
        save_path: str = "dynamical_system_analysis.png"
    ) -> None:
        """
        Create a comprehensive multi-panel analysis visualization.
        
        Args:
            results1: Results from first conversation
            results2: Results from second conversation (for perturbation analysis)
            save_path: Path to save the figure
        """
        
        # Setup figure with subplots
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Extract trajectories
        traj_a1 = results1['agent_a_trajectory']
        traj_b1 = results1['agent_b_trajectory']
        
        if results2:
            traj_a2 = results2['agent_a_trajectory']
            traj_b2 = results2['agent_b_trajectory']
        
        # 1. State evolution over time
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_state_evolution(ax1, traj_a1, traj_b1, "Conversation 1")
        
        # 2. Phase space diagram (2D projection)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_phase_space_2d(ax2, traj_a1, traj_b1)
        
        # 3. State divergence over time
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_state_divergence(ax3, traj_a1, traj_b1)
        
        # 4. State component variance
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_state_variance(ax4, traj_a1, traj_b1)
        
        # 5. 3D Phase space trajectory
        ax5 = fig.add_subplot(gs[1, 0], projection='3d')
        self._plot_phase_space_3d(ax5, traj_a1, traj_b1)
        
        # 6. Conversation dynamics
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_conversation_dynamics(ax6, results1['conversation_history'])
        
        # 7. Lyapunov estimation
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_lyapunov_estimation(ax7, traj_a1, traj_b1)
        
        # 8. State space embedding
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_state_embedding(ax8, traj_a1, traj_b1)
        
        # Bottom row: Perturbation analysis (if available)
        if results2:
            # 9. Perturbation comparison
            ax9 = fig.add_subplot(gs[2, :2])
            self._plot_perturbation_comparison(ax9, traj_a1, traj_b1, traj_a2, traj_b2)
            
            # 10. Content divergence analysis
            ax10 = fig.add_subplot(gs[2, 2:])
            self._plot_content_divergence(ax10, results1['conversation_history'], 
                                        results2['conversation_history'])
        else:
            # 9. Signal vs Noise decomposition
            ax9 = fig.add_subplot(gs[2, :2])
            self._plot_signal_noise_analysis(ax9, results1)
            
            # 10. Attractor reconstruction
            ax10 = fig.add_subplot(gs[2, 2:])
            self._plot_attractor_reconstruction(ax10, traj_a1, traj_b1)
        
        plt.suptitle("Two-Agent Dynamical System Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive analysis saved as {save_path}")
        plt.show()
    
    def _plot_state_evolution(self, ax, traj_a, traj_b, title):
        """Plot state vector magnitudes over time"""
        time_steps = np.arange(len(traj_a))
        
        # Calculate magnitudes
        mag_a = np.linalg.norm(traj_a, axis=1)
        mag_b = np.linalg.norm(traj_b, axis=1)
        
        ax.plot(time_steps, mag_a, 'o-', color=self.colors['agent_a'], 
                label='Agent A', linewidth=2, markersize=4)
        ax.plot(time_steps, mag_b, 's-', color=self.colors['agent_b'], 
                label='Agent B', linewidth=2, markersize=4)
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('State Vector Magnitude')
        ax.set_title(f'State Evolution - {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_space_2d(self, ax, traj_a, traj_b):
        """Plot 2D phase space projection"""
        # Use first two principal components or first two dimensions
        ax.plot(traj_a[:, 0], traj_a[:, 1], 'o-', color=self.colors['agent_a'], 
                label='Agent A', alpha=0.7, linewidth=2)
        ax.plot(traj_b[:, 0], traj_b[:, 1], 's-', color=self.colors['agent_b'], 
                label='Agent B', alpha=0.7, linewidth=2)
        
        # Mark start and end points
        ax.scatter(traj_a[0, 0], traj_a[0, 1], color=self.colors['agent_a'], 
                  s=100, marker='o', edgecolor='black', linewidth=2, label='A Start')
        ax.scatter(traj_a[-1, 0], traj_a[-1, 1], color=self.colors['agent_a'], 
                  s=100, marker='X', edgecolor='black', linewidth=2, label='A End')
        
        ax.scatter(traj_b[0, 0], traj_b[0, 1], color=self.colors['agent_b'], 
                  s=100, marker='s', edgecolor='black', linewidth=2, label='B Start')
        ax.scatter(traj_b[-1, 0], traj_b[-1, 1], color=self.colors['agent_b'], 
                  s=100, marker='X', edgecolor='black', linewidth=2, label='B End')
        
        ax.set_xlabel('State Dimension 1')
        ax.set_ylabel('State Dimension 2')
        ax.set_title('Phase Space (2D Projection)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_space_3d(self, ax, traj_a, traj_b):
        """Plot 3D phase space trajectory"""
        # Use first three dimensions
        ax.plot(traj_a[:, 0], traj_a[:, 1], traj_a[:, 2], 
                color=self.colors['agent_a'], label='Agent A', linewidth=2)
        ax.plot(traj_b[:, 0], traj_b[:, 1], traj_b[:, 2], 
                color=self.colors['agent_b'], label='Agent B', linewidth=2)
        
        # Mark start points
        ax.scatter(traj_a[0, 0], traj_a[0, 1], traj_a[0, 2], 
                  color=self.colors['agent_a'], s=50, marker='o')
        ax.scatter(traj_b[0, 0], traj_b[0, 1], traj_b[0, 2], 
                  color=self.colors['agent_b'], s=50, marker='s')
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('3D Phase Space')
        ax.legend()
    
    def _plot_state_divergence(self, ax, traj_a, traj_b):
        """Plot divergence between agent states over time"""
        time_steps = np.arange(len(traj_a))
        divergence = np.linalg.norm(traj_a - traj_b, axis=1)
        
        ax.plot(time_steps, divergence, 'o-', color=self.colors['divergence'], 
                linewidth=2, markersize=4)
        ax.fill_between(time_steps, divergence, alpha=0.3, color=self.colors['divergence'])
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('State Divergence')
        ax.set_title('Agent State Divergence')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(time_steps) > 1:
            z = np.polyfit(time_steps, divergence, 1)
            p = np.poly1d(z)
            ax.plot(time_steps, p(time_steps), "--", alpha=0.8, color='red',
                   label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            ax.legend()
    
    def _plot_state_variance(self, ax, traj_a, traj_b):
        """Plot variance across state dimensions"""
        var_a = np.var(traj_a, axis=0)
        var_b = np.var(traj_b, axis=0)
        
        # Show first 20 dimensions for clarity
        dims_to_show = min(20, len(var_a))
        dimensions = np.arange(dims_to_show)
        
        width = 0.35
        ax.bar(dimensions - width/2, var_a[:dims_to_show], width, 
               label='Agent A', alpha=0.7, color=self.colors['agent_a'])
        ax.bar(dimensions + width/2, var_b[:dims_to_show], width, 
               label='Agent B', alpha=0.7, color=self.colors['agent_b'])
        
        ax.set_xlabel('State Dimensions')
        ax.set_ylabel('Variance')
        ax.set_title('State Component Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_conversation_dynamics(self, ax, conversation_history):
        """Plot conversation-level dynamics"""
        # Extract agent messages
        agent_messages = [msg for msg in conversation_history 
                         if msg['speaker'].startswith('Agent')]
        
        if not agent_messages:
            ax.text(0.5, 0.5, 'No agent messages', ha='center', va='center')
            return
        
        # Calculate message lengths and response times
        message_lengths = [len(msg['content']) for msg in agent_messages]
        speakers = [msg['speaker'] for msg in agent_messages]
        
        # Color code by speaker
        colors = [self.colors['agent_a'] if speaker == 'Agent_A' else self.colors['agent_b'] 
                 for speaker in speakers]
        
        turns = np.arange(len(message_lengths))
        ax.bar(turns, message_lengths, color=colors, alpha=0.7)
        
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Message Length (characters)')
        ax.set_title('Conversation Dynamics')
        
        # Add legend
        ax.bar([], [], color=self.colors['agent_a'], label='Agent A', alpha=0.7)
        ax.bar([], [], color=self.colors['agent_b'], label='Agent B', alpha=0.7)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_lyapunov_estimation(self, ax, traj_a, traj_b):
        """Plot local Lyapunov exponent estimation"""
        # Calculate local divergence rates
        window_size = min(3, len(traj_a) - 1)
        if window_size < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            return
        
        local_lyapunov = []
        time_points = []
        
        for i in range(len(traj_a) - window_size):
            # Local state differences
            segment_a = traj_a[i:i + window_size]
            differences = np.linalg.norm(np.diff(segment_a, axis=0), axis=1)
            
            if np.all(differences > 1e-12):
                log_diff = np.log(differences + 1e-12)
                local_lyap = np.mean(np.diff(log_diff))
                local_lyapunov.append(local_lyap)
                time_points.append(i + window_size/2)
        
        if local_lyapunov:
            ax.plot(time_points, local_lyapunov, 'o-', color=self.colors['system'], 
                   linewidth=2, markersize=4)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Chaos Threshold')
            
            # Highlight chaotic regions
            chaotic_mask = np.array(local_lyapunov) > 0
            if np.any(chaotic_mask):
                chaotic_points = np.array(time_points)[chaotic_mask]
                chaotic_values = np.array(local_lyapunov)[chaotic_mask]
                ax.scatter(chaotic_points, chaotic_values, color='red', s=50, 
                          alpha=0.7, label='Chaotic Regions')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Local Lyapunov Exponent')
        ax.set_title('Chaos Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_state_embedding(self, ax, traj_a, traj_b):
        """Plot delay embedding reconstruction"""
        # Simple delay embedding for visualization
        if len(traj_a) < 4:
            ax.text(0.5, 0.5, 'Insufficient data for embedding', ha='center', va='center')
            return
        
        # Use magnitude of state vectors
        signal_a = np.linalg.norm(traj_a, axis=1)
        signal_b = np.linalg.norm(traj_b, axis=1)
        
        # Create delay embedding (simple version)
        delay = 1
        if len(signal_a) > delay + 1:
            embedded_a_x = signal_a[:-delay-1]
            embedded_a_y = signal_a[delay:-1]
            embedded_a_z = signal_a[delay+1:]
            
            ax.scatter(embedded_a_x, embedded_a_y, c=embedded_a_z, 
                      cmap='viridis', alpha=0.7, s=30, label='Agent A')
        
        if len(signal_b) > delay + 1:
            embedded_b_x = signal_b[:-delay-1]
            embedded_b_y = signal_b[delay:-1]
            embedded_b_z = signal_b[delay+1:]
            
            scatter = ax.scatter(embedded_b_x, embedded_b_y, c=embedded_b_z, 
                               cmap='plasma', alpha=0.7, s=30, marker='s', label='Agent B')
        
        ax.set_xlabel('x(t)')
        ax.set_ylabel('x(t+τ)')
        ax.set_title('Delay Embedding')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_perturbation_comparison(self, ax, traj_a1, traj_b1, traj_a2, traj_b2):
        """Compare trajectories from perturbed initial conditions"""
        time_steps = np.arange(min(len(traj_a1), len(traj_a2)))
        
        # Calculate magnitudes
        mag_a1 = np.linalg.norm(traj_a1[:len(time_steps)], axis=1)
        mag_a2 = np.linalg.norm(traj_a2[:len(time_steps)], axis=1)
        mag_b1 = np.linalg.norm(traj_b1[:len(time_steps)], axis=1)
        mag_b2 = np.linalg.norm(traj_b2[:len(time_steps)], axis=1)
        
        # Plot trajectories
        ax.plot(time_steps, mag_a1, '-', color=self.colors['agent_a'], 
               linewidth=2, label='Agent A - Run 1')
        ax.plot(time_steps, mag_a2, '--', color=self.colors['agent_a'], 
               linewidth=2, alpha=0.7, label='Agent A - Run 2')
        ax.plot(time_steps, mag_b1, '-', color=self.colors['agent_b'], 
               linewidth=2, label='Agent B - Run 1')
        ax.plot(time_steps, mag_b2, '--', color=self.colors['agent_b'], 
               linewidth=2, alpha=0.7, label='Agent B - Run 2')
        
        # Calculate and plot divergence
        divergence_a = np.abs(mag_a1 - mag_a2)
        divergence_b = np.abs(mag_b1 - mag_b2)
        
        ax2 = ax.twinx()
        ax2.fill_between(time_steps, divergence_a, alpha=0.3, 
                        color=self.colors['agent_a'], label='A Divergence')
        ax2.fill_between(time_steps, divergence_b, alpha=0.3, 
                        color=self.colors['agent_b'], label='B Divergence')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('State Magnitude')
        ax2.set_ylabel('Trajectory Divergence')
        ax.set_title('Perturbation Analysis: Sensitive Dependence')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_content_divergence(self, ax, conv1, conv2):
        """Analyze content divergence between conversations"""
        # Extract agent messages
        messages1 = [msg['content'] for msg in conv1 if msg['speaker'].startswith('Agent')]
        messages2 = [msg['content'] for msg in conv2 if msg['speaker'].startswith('Agent')]
        
        # Calculate similarity over time
        min_len = min(len(messages1), len(messages2))
        similarities = []
        
        for i in range(min_len):
            words1 = set(messages1[i].lower().split())
            words2 = set(messages2[i].lower().split())
            
            if words1 and words2:
                overlap = len(words1.intersection(words2))
                total = len(words1.union(words2))
                similarity = overlap / total if total > 0 else 0
            else:
                similarity = 0
            
            similarities.append(similarity)
        
        turns = np.arange(len(similarities))
        ax.plot(turns, similarities, 'o-', color=self.colors['divergence'], 
               linewidth=2, markersize=6)
        ax.fill_between(turns, similarities, alpha=0.3, color=self.colors['divergence'])
        
        # Add threshold line
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                  label='50% Similarity Threshold')
        
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Content Similarity')
        ax.set_title('Content Divergence Analysis')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        avg_similarity = np.mean(similarities) if similarities else 0
        ax.text(0.02, 0.98, f'Avg Similarity: {avg_similarity:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _plot_signal_noise_analysis(self, ax, results):
        """Visualize signal vs noise components"""
        # This would require signal/noise analysis results
        # For now, create a placeholder
        categories = ['Semantic\nCoherence', 'Syntactic\nPatterns', 'Lexical\nRandomness', 
                     'Processing\nErrors', 'Semantic\nDrift']
        values = [0.3, 0.7, 0.5, 0.1, 0.9]  # Placeholder values
        colors = [self.colors['signal'], self.colors['signal'], self.colors['noise'], 
                 self.colors['noise'], self.colors['noise']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Metric Value')
        ax.set_title('Signal vs Noise Analysis')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        signal_patch = patches.Patch(color=self.colors['signal'], alpha=0.7, label='Signal')
        noise_patch = patches.Patch(color=self.colors['noise'], alpha=0.7, label='Noise')
        ax.legend(handles=[signal_patch, noise_patch])
    
    def _plot_attractor_reconstruction(self, ax, traj_a, traj_b):
        """Plot attractor reconstruction using delay coordinates"""
        # Combine agent states for system attractor
        if len(traj_a) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            return
        
        # Create system state (concatenate agent states)
        system_state = np.concatenate([traj_a, traj_b], axis=1)
        system_magnitude = np.linalg.norm(system_state, axis=1)
        
        # Delay embedding
        if len(system_magnitude) >= 3:
            x = system_magnitude[:-2]
            y = system_magnitude[1:-1] 
            z = system_magnitude[2:]
            
            # Create color gradient for time evolution
            colors = np.arange(len(x))
            scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=50, alpha=0.7)
            
            # Connect points to show trajectory
            ax.plot(x, y, '-', alpha=0.3, color='gray', linewidth=1)
            
            # Mark start and end
            ax.scatter(x[0], y[0], color='green', s=100, marker='o', 
                      edgecolor='black', linewidth=2, label='Start')
            ax.scatter(x[-1], y[-1], color='red', s=100, marker='X', 
                      edgecolor='black', linewidth=2, label='End')
        
        ax.set_xlabel('System State(t)')
        ax.set_ylabel('System State(t+τ)')
        ax.set_title('System Attractor Reconstruction')
        ax.legend()
        ax.grid(True, alpha=0.3)