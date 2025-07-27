"""
Comprehensive comparison study of different encoding schemes.
"""

import os
import numpy as np
from dotenv import load_dotenv
from src.agent_system import SimpleTwoAgentSystem
from src.chaos_analysis import ChaosAnalyzer
import time
import matplotlib.pyplot as plt

load_dotenv()

def run_encoding_comparison():
    """Compare different encoding schemes across multiple conversations."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in .env file")
        return
    
    print("🔬 COMPREHENSIVE ENCODING COMPARISON STUDY")
    print("=" * 60)
    
    # Test configurations - focused for efficiency
    encodings = ["hash", "advanced"]  # Compare baseline vs best
    conversation_lengths = [25]  # Single manageable length
    topics = [
        "What is consciousness?"
    ]
    
    results = {}
    
    for encoding in encodings:
        print(f"\n🧮 Testing {encoding.upper()} encoding...")
        results[encoding] = {}
        
        for length in conversation_lengths:
            print(f"  📏 Length: {length} turns")
            results[encoding][length] = []
            
            for i, topic in enumerate(topics):
                print(f"    💭 Topic {i+1}: {topic}")
                
                # Create system
                system = SimpleTwoAgentSystem(
                    agent_a_prompt="You are a systematic researcher. Be concise but substantive.",
                    agent_b_prompt="You are a creative thinker. Be concise but substantive.",
                    max_turns=length,
                    encoding_type=encoding,
                    state_dimension=64,
                    noise_scale=0.005
                )
                
                # Run conversation
                start_time = time.time()
                result = system.run_conversation(topic, verbose=False)
                duration = time.time() - start_time
                
                # Analyze
                analyzer = ChaosAnalyzer()
                traj_a = result['agent_a_trajectory']
                traj_b = result['agent_b_trajectory']
                
                lyap_a = analyzer.estimate_lyapunov_from_single_trajectory(traj_a)
                lyap_b = analyzer.estimate_lyapunov_from_single_trajectory(traj_b)
                
                final_divergence = np.linalg.norm(traj_a[-1] - traj_b[-1])
                trajectory_variance = np.var(np.linalg.norm(traj_a - traj_a[0], axis=1))
                
                # Store results
                results[encoding][length].append({
                    "topic": topic,
                    "lyapunov_a": lyap_a,
                    "lyapunov_b": lyap_b,
                    "max_lyapunov": max(lyap_a, lyap_b),
                    "final_divergence": final_divergence,
                    "trajectory_variance": trajectory_variance,
                    "duration": duration,
                    "total_messages": result['total_turns']
                })
                
                print(f"      ✅ λ_max: {max(lyap_a, lyap_b):.6f}, Div: {final_divergence:.3f}, Time: {duration:.1f}s")
    
    return results

def analyze_results(results):
    """Analyze and visualize the comparison results."""
    
    print(f"\n📊 ENCODING PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Calculate aggregate statistics
    encoding_stats = {}
    
    for encoding in results.keys():
        all_metrics = []
        for length in results[encoding].keys():
            for data in results[encoding][length]:
                all_metrics.append(data)
        
        if all_metrics:
            encoding_stats[encoding] = {
                'avg_max_lyapunov': np.mean([m['max_lyapunov'] for m in all_metrics]),
                'std_max_lyapunov': np.std([m['max_lyapunov'] for m in all_metrics]),
                'avg_divergence': np.mean([m['final_divergence'] for m in all_metrics]),
                'avg_variance': np.mean([m['trajectory_variance'] for m in all_metrics]),
                'avg_duration': np.mean([m['duration'] for m in all_metrics]),
                'chaos_score': np.mean([1 if m['max_lyapunov'] > 0 else 0 for m in all_metrics])
            }
    
    # Print comparison table
    print(f"\n{'Encoding':<12} {'Avg λ_max':<12} {'Std λ_max':<12} {'Avg Div':<10} {'Chaos %':<8} {'Avg Time':<10}")
    print("-" * 70)
    
    for encoding, stats in encoding_stats.items():
        print(f"{encoding:<12} {stats['avg_max_lyapunov']:<12.6f} {stats['std_max_lyapunov']:<12.6f} "
              f"{stats['avg_divergence']:<10.3f} {stats['chaos_score']*100:<8.0f} {stats['avg_duration']:<10.1f}")
    
    # Determine best encoding
    best_encoding = max(encoding_stats.keys(), key=lambda x: encoding_stats[x]['avg_max_lyapunov'])
    
    print(f"\n🏆 WINNER: {best_encoding.upper()} encoding")
    print(f"   Best average Lyapunov exponent: {encoding_stats[best_encoding]['avg_max_lyapunov']:.6f}")
    print(f"   Chaos detection rate: {encoding_stats[best_encoding]['chaos_score']*100:.0f}%")
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    for encoding in results.keys():
        stats = encoding_stats[encoding]
        print(f"\n   {encoding.upper()} Encoding:")
        print(f"     • Average Lyapunov: {stats['avg_max_lyapunov']:.6f} ± {stats['std_max_lyapunov']:.6f}")
        print(f"     • Chaos detection: {stats['chaos_score']*100:.0f}% of conversations")
        print(f"     • Average divergence: {stats['avg_divergence']:.3f}")
        print(f"     • Computation time: {stats['avg_duration']:.1f}s per conversation")
    
    return encoding_stats, best_encoding

def create_visualization(results, encoding_stats):
    """Create visualization of encoding comparison."""
    
    print(f"\n📈 Creating comparison visualization...")
    
    # Prepare data for plotting
    encodings = list(encoding_stats.keys())
    avg_lyapunov = [encoding_stats[enc]['avg_max_lyapunov'] for enc in encodings]
    std_lyapunov = [encoding_stats[enc]['std_max_lyapunov'] for enc in encodings]
    chaos_rates = [encoding_stats[enc]['chaos_score'] * 100 for enc in encodings]
    avg_times = [encoding_stats[enc]['avg_duration'] for enc in encodings]
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Encoding Scheme Comparison Study', fontsize=16, fontweight='bold')
    
    # Plot 1: Average Lyapunov Exponents
    bars1 = ax1.bar(encodings, avg_lyapunov, yerr=std_lyapunov, capsize=5, 
                   color=['lightblue', 'lightgreen', 'lightcoral'])
    ax1.set_title('Average Maximum Lyapunov Exponent')
    ax1.set_ylabel('Lyapunov Exponent')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Chaos Threshold')
    ax1.legend()
    
    # Add value labels on bars
    for bar, val, std in zip(bars1, avg_lyapunov, std_lyapunov):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Chaos Detection Rate
    bars2 = ax2.bar(encodings, chaos_rates, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax2.set_title('Chaos Detection Rate')
    ax2.set_ylabel('Percentage of Conversations with λ > 0')
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for bar, val in zip(bars2, chaos_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Average Final Divergence
    avg_div = [encoding_stats[enc]['avg_divergence'] for enc in encodings]
    bars3 = ax3.bar(encodings, avg_div, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax3.set_title('Average Final State Divergence')
    ax3.set_ylabel('Final Divergence')
    
    for bar, val in zip(bars3, avg_div):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Computation Time
    bars4 = ax4.bar(encodings, avg_times, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax4.set_title('Average Computation Time')
    ax4.set_ylabel('Time (seconds)')
    
    for bar, val in zip(bars4, avg_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('encoding_comparison_study.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: encoding_comparison_study.png")
    
    return fig

def main():
    """Main execution function."""
    
    print("🧬 ADVANCED ENCODING COMPARISON STUDY")
    print("Testing hash-based vs semantic vs advanced multi-feature encodings")
    print("=" * 70)
    
    # Run comparison
    results = run_encoding_comparison()
    
    # Analyze results
    encoding_stats, best_encoding = analyze_results(results)
    
    # Create visualization
    fig = create_visualization(results, encoding_stats)
    
    # Summary
    print(f"\n✅ STUDY COMPLETE!")
    print(f"🏆 Best performing encoding: {best_encoding.upper()}")
    print(f"📊 Detailed comparison chart: encoding_comparison_study.png")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if best_encoding == "advanced":
        print(f"   • Use ADVANCED encoding for maximum chaos detection sensitivity")
        print(f"   • Expect longer computation times but richer state dynamics")
        print(f"   • Ideal for research requiring detailed semantic/syntactic analysis")
    elif best_encoding == "semantic":
        print(f"   • Use SEMANTIC encoding for meaning-focused analysis")
        print(f"   • Good balance of performance and interpretability")
        print(f"   • Suitable for content-driven chaos studies")
    else:
        print(f"   • HASH encoding provides baseline performance")
        print(f"   • Fastest computation, good for large-scale studies")
        print(f"   • Consider as control condition for comparisons")
    
    return results, encoding_stats, best_encoding

if __name__ == "__main__":
    main()