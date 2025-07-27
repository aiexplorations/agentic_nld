"""
Quick analysis for technical report data generation.
"""

import os
import numpy as np
import json
from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agent_system import SimpleTwoAgentSystem
from src.chaos_analysis import ChaosAnalyzer

load_dotenv()

def quick_experiment():
    """Quick experiment to generate key data points."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY")
        return {}
    
    print("🚀 Quick Analysis for Technical Report")
    print("=" * 40)
    
    analyzer = ChaosAnalyzer()
    results = {}
    
    # Test 1: Different conversation lengths
    print("\n📏 Testing conversation lengths...")
    length_results = {}
    
    for length in [5, 10, 15]:
        print(f"  Testing {length} turns...")
        
        system = SimpleTwoAgentSystem(
            agent_a_prompt="You are a systematic researcher.",
            agent_b_prompt="You are a creative philosopher.", 
            max_turns=length
        )
        
        conv_result = system.run_conversation("What is intelligence?")
        traj_a = conv_result['agent_a_trajectory']
        traj_b = conv_result['agent_b_trajectory']
        
        # Calculate key metrics
        lyap_a = analyzer.estimate_lyapunov_from_single_trajectory(traj_a)
        lyap_b = analyzer.estimate_lyapunov_from_single_trajectory(traj_b)
        divergence = np.linalg.norm(traj_a - traj_b, axis=1)
        
        length_results[length] = {
            'lyapunov_a': float(lyap_a),
            'lyapunov_b': float(lyap_b),
            'initial_divergence': float(divergence[0]),
            'final_divergence': float(divergence[-1]),
            'max_divergence': float(np.max(divergence)),
            'trajectory_length': len(traj_a),
            'final_state_norm_a': float(np.linalg.norm(traj_a[-1])),
            'final_state_norm_b': float(np.linalg.norm(traj_b[-1]))
        }
        
        print(f"    ✓ Lyapunov A: {lyap_a:.6f}, B: {lyap_b:.6f}")
        print(f"    ✓ Divergence: {divergence[0]:.3f} → {divergence[-1]:.3f}")
    
    results['conversation_lengths'] = length_results
    
    # Test 2: Sensitivity analysis
    print("\n🔬 Testing sensitivity to initial conditions...")
    
    base_prompt_a = "You are a logical researcher."
    base_prompt_b = "You are a creative philosopher."
    
    perturbations = [
        ("", ""),  # Baseline
        (" Be concise.", " Be concise."),
        (" Think deeply.", " Think deeply.")
    ]
    
    sensitivity_results = {}
    baseline_traj = None
    
    for i, (pert_a, pert_b) in enumerate(perturbations):
        print(f"  Testing perturbation {i}...")
        
        system = SimpleTwoAgentSystem(
            agent_a_prompt=base_prompt_a + pert_a,
            agent_b_prompt=base_prompt_b + pert_b,
            max_turns=8
        )
        
        conv_result = system.run_conversation("How does consciousness emerge?")
        traj_a = conv_result['agent_a_trajectory']
        traj_b = conv_result['agent_b_trajectory']
        
        if i == 0:
            baseline_traj = (traj_a, traj_b)
        
        # Calculate metrics
        lyap_a = analyzer.estimate_lyapunov_from_single_trajectory(traj_a)
        
        sensitivity_results[f'perturbation_{i}'] = {
            'perturbation': (pert_a, pert_b),
            'lyapunov_a': float(lyap_a),
            'final_state_norm_a': float(np.linalg.norm(traj_a[-1])),
            'trajectory_length': len(traj_a)
        }
        
        # Calculate divergence from baseline
        if baseline_traj and i > 0:
            min_len = min(len(traj_a), len(baseline_traj[0]))
            state_diff = np.linalg.norm(traj_a[:min_len][-1] - baseline_traj[0][:min_len][-1])
            sensitivity_results[f'perturbation_{i}']['divergence_from_baseline'] = float(state_diff)
            print(f"    ✓ Divergence from baseline: {state_diff:.6f}")
    
    results['sensitivity_analysis'] = sensitivity_results
    
    # Test 3: Phase space properties
    print("\n🌌 Analyzing phase space...")
    
    system = SimpleTwoAgentSystem(
        agent_a_prompt="You are Dr. Sarah, a cognitive scientist.",
        agent_b_prompt="You are Prof. Marcus, a philosopher.",
        max_turns=12
    )
    
    conv_result = system.run_conversation("Let's explore consciousness and AI.")
    traj_a = conv_result['agent_a_trajectory']
    traj_b = conv_result['agent_b_trajectory']
    
    # Phase space metrics
    system_traj = np.concatenate([traj_a, traj_b], axis=1)
    correlation_dim = analyzer._estimate_correlation_dimension(system_traj[:8])
    
    phase_results = {
        'correlation_dimension': float(correlation_dim),
        'attractor_size': float(np.max(np.ptp(system_traj, axis=0))),
        'trajectory_spread': float(np.std(system_traj, axis=0).mean()),
        'system_dimensionality': system_traj.shape[1]
    }
    
    results['phase_space_analysis'] = phase_results
    
    print(f"    ✓ Correlation dimension: {correlation_dim:.3f}")
    print(f"    ✓ Attractor size: {phase_results['attractor_size']:.3f}")
    
    # Save results
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "quick_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Quick analysis complete!")
    print(f"💾 Results saved to experiments/results/quick_analysis_results.json")
    
    return results

if __name__ == "__main__":
    quick_experiment()