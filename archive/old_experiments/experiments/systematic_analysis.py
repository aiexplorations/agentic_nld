"""
Systematic experimental analysis for technical report.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agent_system import SimpleTwoAgentSystem
from src.chaos_analysis import ChaosAnalyzer, SignalNoiseAnalyzer
from src.simple_viz import create_key_metrics_plot

load_dotenv()

class SystematicExperiment:
    """
    Comprehensive experimental analysis of two-agent dynamical system.
    """
    
    def __init__(self):
        self.results = {}
        self.analyzer = ChaosAnalyzer()
        self.signal_analyzer = SignalNoiseAnalyzer()
        
        # Create results directory
        self.results_dir = Path("experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def experiment_1_conversation_length_analysis(self) -> Dict:
        """
        Experiment 1: How do dynamics change with conversation length?
        """
        print("🧪 EXPERIMENT 1: Conversation Length Analysis")
        print("=" * 50)
        
        lengths = [5, 10, 15, 20]
        results = {}
        
        for length in lengths:
            print(f"\n📏 Testing {length}-turn conversations...")
            
            system = SimpleTwoAgentSystem(
                agent_a_prompt="You are a systematic researcher. Ask analytical questions. Be concise.",
                agent_b_prompt="You are a creative philosopher. Make unexpected connections. Be concise.",
                max_turns=length
            )
            
            conversation_results = system.run_conversation("What is the nature of intelligence?")
            
            # Extract metrics
            traj_a = conversation_results['agent_a_trajectory']
            traj_b = conversation_results['agent_b_trajectory']
            
            metrics = self._calculate_comprehensive_metrics(traj_a, traj_b, conversation_results)
            results[length] = metrics
            
            print(f"   ✓ Lyapunov: {metrics['lyapunov_a']:.6f}")
            print(f"   ✓ Divergence: {metrics['final_divergence']:.6f}")
            print(f"   ✓ Trajectory variance: {metrics['trajectory_variance']:.6f}")
        
        self.results['experiment_1'] = results
        return results
    
    def experiment_2_sensitivity_analysis(self, n_trials: int = 5) -> Dict:
        """
        Experiment 2: Sensitivity to initial conditions.
        """
        print(f"\n🧪 EXPERIMENT 2: Sensitivity Analysis ({n_trials} trials)")
        print("=" * 50)
        
        base_prompt_a = "You are a logical researcher. Be analytical and precise."
        base_prompt_b = "You are a creative thinker. Make imaginative connections."
        
        # Small perturbations
        perturbations = [
            ("", ""),  # Baseline
            (" Be concise.", " Be concise."),
            (" Keep responses short.", " Keep responses short."),
            (" Focus on clarity.", " Focus on clarity."),
            (" Think deeply.", " Think deeply.")
        ]
        
        results = {}
        baseline_trajectory = None
        
        for i, (pert_a, pert_b) in enumerate(perturbations):
            print(f"\n🔬 Perturbation {i}: '{pert_a.strip()}' / '{pert_b.strip()}'")
            
            trial_results = []
            
            for trial in range(n_trials):
                system = SimpleTwoAgentSystem(
                    agent_a_prompt=base_prompt_a + pert_a,
                    agent_b_prompt=base_prompt_b + pert_b,
                    max_turns=10
                )
                
                conv_result = system.run_conversation("How does consciousness emerge?")
                traj_a = conv_result['agent_a_trajectory']
                traj_b = conv_result['agent_b_trajectory']
                
                metrics = self._calculate_comprehensive_metrics(traj_a, traj_b, conv_result)
                trial_results.append(metrics)
                
                if i == 0 and trial == 0:  # Store baseline for comparison
                    baseline_trajectory = (traj_a, traj_b)
            
            # Calculate statistics across trials
            avg_metrics = self._average_metrics(trial_results)
            results[f"perturbation_{i}"] = {
                'perturbation': (pert_a, pert_b),
                'metrics': avg_metrics,
                'std_dev': self._std_dev_metrics(trial_results)
            }
            
            print(f"   ✓ Avg Lyapunov: {avg_metrics['lyapunov_a']:.6f} ± {results[f'perturbation_{i}']['std_dev']['lyapunov_a']:.6f}")
        
        self.results['experiment_2'] = results
        return results
    
    def experiment_3_phase_space_analysis(self) -> Dict:
        """
        Experiment 3: Phase space reconstruction and attractor analysis.
        """
        print(f"\n🧪 EXPERIMENT 3: Phase Space Analysis")
        print("=" * 50)
        
        # Run longer conversation for better phase space reconstruction
        system = SimpleTwoAgentSystem(
            agent_a_prompt="You are Dr. Analytics, a systematic cognitive scientist.",
            agent_b_prompt="You are Prof. Creativity, an imaginative philosopher.",
            max_turns=25
        )
        
        conv_result = system.run_conversation("Let's explore the fundamental nature of consciousness and intelligence.")
        traj_a = conv_result['agent_a_trajectory']
        traj_b = conv_result['agent_b_trajectory']
        
        # Phase space analysis
        results = {}
        
        # 1. Attractor dimension estimation
        results['attractor_analysis'] = self._analyze_attractor(traj_a, traj_b)
        
        # 2. Recurrence analysis
        results['recurrence_analysis'] = self._recurrence_analysis(traj_a, traj_b)
        
        # 3. Embedding dimension analysis
        results['embedding_analysis'] = self._embedding_dimension_analysis(traj_a, traj_b)
        
        print(f"   ✓ Estimated attractor dimension: {results['attractor_analysis']['correlation_dimension']:.3f}")
        print(f"   ✓ Recurrence rate: {results['recurrence_analysis']['recurrence_rate']:.3f}")
        print(f"   ✓ Optimal embedding dimension: {results['embedding_analysis']['optimal_dimension']}")
        
        # Generate phase space visualization
        self._create_phase_space_visualization(traj_a, traj_b, "experiment_3_phase_space.png")
        
        self.results['experiment_3'] = results
        return results
    
    def experiment_4_signal_noise_decomposition(self) -> Dict:
        """
        Experiment 4: Signal vs noise analysis across different conditions.
        """
        print(f"\n🧪 EXPERIMENT 4: Signal vs Noise Analysis")
        print("=" * 50)
        
        conditions = [
            ("High structure", "You are highly systematic. Follow logical patterns. Be precise.", 
             "You are highly creative. Think outside the box. Be imaginative."),
            ("Medium structure", "You are moderately systematic. Balance logic and intuition.", 
             "You are moderately creative. Mix analysis with creativity."),
            ("Low structure", "You respond naturally. Follow your instincts.", 
             "You respond naturally. Let ideas flow freely.")
        ]
        
        results = {}
        
        for condition_name, prompt_a, prompt_b in conditions:
            print(f"\n📊 Analyzing {condition_name.lower()} condition...")
            
            system = SimpleTwoAgentSystem(
                agent_a_prompt=prompt_a,
                agent_b_prompt=prompt_b,
                max_turns=15
            )
            
            conv_result = system.run_conversation("What is the relationship between order and chaos in complex systems?")
            
            # Signal/noise analysis
            signal_noise = self.signal_analyzer.decompose_conversation(
                conv_result['conversation_history'],
                {"agent_a": conv_result['agent_a_trajectory'], 
                 "agent_b": conv_result['agent_b_trajectory']}
            )
            
            # Additional metrics
            traj_a = conv_result['agent_a_trajectory']
            traj_b = conv_result['agent_b_trajectory']
            
            metrics = {
                'signal_noise_ratio': self._calculate_snr(signal_noise),
                'information_content': self._calculate_information_content(conv_result['conversation_history']),
                'trajectory_complexity': self._calculate_trajectory_complexity(traj_a, traj_b),
                'signal_noise_detailed': signal_noise
            }
            
            results[condition_name.lower().replace(' ', '_')] = metrics
            
            print(f"   ✓ SNR: {metrics['signal_noise_ratio']:.3f}")
            print(f"   ✓ Information content: {metrics['information_content']:.3f}")
            print(f"   ✓ Trajectory complexity: {metrics['trajectory_complexity']:.3f}")
        
        self.results['experiment_4'] = results
        return results
    
    def _calculate_comprehensive_metrics(self, traj_a: np.ndarray, traj_b: np.ndarray, 
                                        conv_result: Dict) -> Dict:
        """Calculate comprehensive metrics for a conversation."""
        metrics = {}
        
        # Lyapunov exponents
        metrics['lyapunov_a'] = self.analyzer.estimate_lyapunov_from_single_trajectory(traj_a)
        metrics['lyapunov_b'] = self.analyzer.estimate_lyapunov_from_single_trajectory(traj_b)
        
        # State divergence
        divergence = np.linalg.norm(traj_a - traj_b, axis=1)
        metrics['initial_divergence'] = divergence[0]
        metrics['final_divergence'] = divergence[-1]
        metrics['max_divergence'] = np.max(divergence)
        metrics['divergence_trend'] = np.polyfit(np.arange(len(divergence)), divergence, 1)[0]
        
        # Trajectory properties
        metrics['trajectory_length'] = len(traj_a)
        metrics['trajectory_variance'] = np.mean([np.var(traj_a), np.var(traj_b)])
        metrics['final_state_magnitude_a'] = np.linalg.norm(traj_a[-1])
        metrics['final_state_magnitude_b'] = np.linalg.norm(traj_b[-1])
        
        # Conversation properties
        agent_messages = [msg for msg in conv_result['conversation_history'] 
                         if msg['speaker'].startswith('Agent')]
        metrics['total_messages'] = len(agent_messages)
        metrics['avg_message_length'] = np.mean([len(msg['content']) for msg in agent_messages])
        metrics['message_length_variance'] = np.var([len(msg['content']) for msg in agent_messages])
        
        return metrics
    
    def _average_metrics(self, trial_results: List[Dict]) -> Dict:
        """Average metrics across trials."""
        avg_metrics = {}
        for key in trial_results[0].keys():
            values = [result[key] for result in trial_results]
            avg_metrics[key] = np.mean(values)
        return avg_metrics
    
    def _std_dev_metrics(self, trial_results: List[Dict]) -> Dict:
        """Calculate standard deviation of metrics across trials."""
        std_metrics = {}
        for key in trial_results[0].keys():
            values = [result[key] for result in trial_results]
            std_metrics[key] = np.std(values)
        return std_metrics
    
    def _analyze_attractor(self, traj_a: np.ndarray, traj_b: np.ndarray) -> Dict:
        """Analyze attractor properties."""
        # Combine trajectories for system analysis
        system_traj = np.concatenate([traj_a, traj_b], axis=1)
        
        # Estimate correlation dimension
        correlation_dim = self.analyzer._estimate_correlation_dimension(system_traj[:10])
        
        # Calculate attractor size
        attractor_size = np.max(np.ptp(system_traj, axis=0))
        
        return {
            'correlation_dimension': correlation_dim,
            'attractor_size': attractor_size,
            'trajectory_spread': np.std(system_traj, axis=0).mean()
        }
    
    def _recurrence_analysis(self, traj_a: np.ndarray, traj_b: np.ndarray) -> Dict:
        """Analyze recurrence properties."""
        # Simple recurrence analysis
        threshold = 0.1
        n_points = len(traj_a)
        
        recurrences = 0
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist_a = np.linalg.norm(traj_a[i] - traj_a[j])
                dist_b = np.linalg.norm(traj_b[i] - traj_b[j])
                if dist_a < threshold and dist_b < threshold:
                    recurrences += 1
        
        total_pairs = n_points * (n_points - 1) // 2
        recurrence_rate = recurrences / total_pairs if total_pairs > 0 else 0
        
        return {
            'recurrence_rate': recurrence_rate,
            'threshold': threshold,
            'total_comparisons': total_pairs
        }
    
    def _embedding_dimension_analysis(self, traj_a: np.ndarray, traj_b: np.ndarray) -> Dict:
        """Analyze optimal embedding dimension."""
        # Use state magnitude as time series
        signal_a = np.linalg.norm(traj_a, axis=1)
        signal_b = np.linalg.norm(traj_b, axis=1)
        
        # Test different embedding dimensions
        dimensions = range(2, min(8, len(signal_a)//2))
        best_dim_a = 3  # Default
        best_dim_b = 3  # Default
        
        if len(signal_a) > 6:
            # Simple test: minimize prediction error
            errors_a = []
            errors_b = []
            
            for dim in dimensions:
                try:
                    embedded_a = self.analyzer.phase_space_reconstruction(signal_a, delay=1, embedding_dim=dim)
                    embedded_b = self.analyzer.phase_space_reconstruction(signal_b, delay=1, embedding_dim=dim)
                    
                    # Simple prediction error (variance of differences)
                    if len(embedded_a) > 1:
                        error_a = np.var(np.diff(embedded_a, axis=0))
                        error_b = np.var(np.diff(embedded_b, axis=0))
                        errors_a.append(error_a)
                        errors_b.append(error_b)
                    else:
                        errors_a.append(float('inf'))
                        errors_b.append(float('inf'))
                except:
                    errors_a.append(float('inf'))
                    errors_b.append(float('inf'))
            
            if errors_a and min(errors_a) != float('inf'):
                best_dim_a = list(dimensions)[np.argmin(errors_a)]
            if errors_b and min(errors_b) != float('inf'):
                best_dim_b = list(dimensions)[np.argmin(errors_b)]
        
        return {
            'optimal_dimension': max(best_dim_a, best_dim_b),
            'agent_a_optimal': best_dim_a,
            'agent_b_optimal': best_dim_b
        }
    
    def _calculate_snr(self, signal_noise: Dict) -> float:
        """Calculate signal-to-noise ratio."""
        signal_components = signal_noise['signal']
        noise_components = signal_noise['noise']
        
        signal_power = np.mean(list(signal_components.values()))
        noise_power = np.mean(list(noise_components.values()))
        
        return signal_power / (noise_power + 1e-6)  # Avoid division by zero
    
    def _calculate_information_content(self, conversation: List[Dict]) -> float:
        """Calculate information content of conversation."""
        # Extract all words
        all_words = []
        for msg in conversation:
            if msg['speaker'].startswith('Agent'):
                all_words.extend(msg['content'].lower().split())
        
        if not all_words:
            return 0.0
        
        # Calculate word frequencies
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate entropy (information content)
        total_words = len(all_words)
        entropy = 0.0
        for count in word_counts.values():
            prob = count / total_words
            entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _calculate_trajectory_complexity(self, traj_a: np.ndarray, traj_b: np.ndarray) -> float:
        """Calculate trajectory complexity measure."""
        # Combine trajectories
        combined = np.concatenate([traj_a.flatten(), traj_b.flatten()])
        
        # Calculate approximate entropy
        def approx_entropy(data, m=2, r=0.2):
            N = len(data)
            if N < m + 1:
                return 0.0
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template = patterns[i]
                    matches = sum([1 for pattern in patterns 
                                 if _maxdist(template, pattern, m) <= r])
                    C[i] = matches / float(N - m + 1)
                
                phi = sum([np.log(c) for c in C if c > 0]) / float(N - m + 1)
                return phi
            
            return _phi(m) - _phi(m + 1)
        
        return approx_entropy(combined)
    
    def _create_phase_space_visualization(self, traj_a: np.ndarray, traj_b: np.ndarray, 
                                        filename: str):
        """Create detailed phase space visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Phase Space Analysis", fontsize=16)
        
        # 2D Phase space
        ax1 = axes[0, 0]
        ax1.plot(traj_a[:, 0], traj_a[:, 1], 'b-', alpha=0.7, label='Agent A')
        ax1.plot(traj_b[:, 0], traj_b[:, 1], 'r-', alpha=0.7, label='Agent B')
        ax1.scatter(traj_a[0, 0], traj_a[0, 1], color='blue', s=100, marker='o', label='A Start')
        ax1.scatter(traj_b[0, 0], traj_b[0, 1], color='red', s=100, marker='s', label='B Start')
        ax1.set_xlabel('State Dim 1')
        ax1.set_ylabel('State Dim 2')
        ax1.set_title('2D Phase Space')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # State evolution
        ax2 = axes[0, 1]
        time = np.arange(len(traj_a))
        mag_a = np.linalg.norm(traj_a, axis=1)
        mag_b = np.linalg.norm(traj_b, axis=1)
        ax2.plot(time, mag_a, 'b-', label='Agent A')
        ax2.plot(time, mag_b, 'r-', label='Agent B')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('State Magnitude')
        ax2.set_title('State Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Divergence
        ax3 = axes[1, 0]
        divergence = np.linalg.norm(traj_a - traj_b, axis=1)
        ax3.plot(time, divergence, 'g-', linewidth=2)
        ax3.fill_between(time, divergence, alpha=0.3, color='green')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('State Divergence')
        ax3.set_title('Agent Divergence')
        ax3.grid(True, alpha=0.3)
        
        # Delay embedding
        ax4 = axes[1, 1]
        if len(mag_a) > 3:
            ax4.plot(mag_a[:-2], mag_a[1:-1], 'b-', alpha=0.7, label='Agent A')
            ax4.plot(mag_b[:-2], mag_b[1:-1], 'r-', alpha=0.7, label='Agent B')
        ax4.set_xlabel('x(t)')
        ax4.set_ylabel('x(t+τ)')
        ax4.set_title('Delay Embedding')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, filename: str = "systematic_analysis_results.json"):
        """Save all experimental results."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_results = convert_numpy(self.results)
        
        with open(self.results_dir / filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"\n💾 Results saved to {self.results_dir / filename}")
    
    def run_all_experiments(self):
        """Run all experiments systematically."""
        print("🚀 STARTING SYSTEMATIC EXPERIMENTAL ANALYSIS")
        print("=" * 60)
        
        try:
            # Experiment 1: Conversation length
            self.experiment_1_conversation_length_analysis()
            
            # Experiment 2: Sensitivity analysis  
            self.experiment_2_sensitivity_analysis(n_trials=3)  # Reduced for speed
            
            # Experiment 3: Phase space analysis
            self.experiment_3_phase_space_analysis()
            
            # Experiment 4: Signal/noise analysis
            self.experiment_4_signal_noise_decomposition()
            
            # Save all results
            self.save_results()
            
            print(f"\n✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
            print(f"📊 Results saved in {self.results_dir}")
            
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run systematic analysis."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in .env file")
        return
    
    experiment = SystematicExperiment()
    experiment.run_all_experiments()


if __name__ == "__main__":
    main()