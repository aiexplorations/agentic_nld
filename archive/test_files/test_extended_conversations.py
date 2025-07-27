"""
Test extended conversations (40-50 turns) with different encoding schemes.
"""

import os
import numpy as np
from dotenv import load_dotenv
from src.agent_system import SimpleTwoAgentSystem
from src.chaos_analysis import ChaosAnalyzer
from src.simple_viz import create_key_metrics_plot
import matplotlib.pyplot as plt
import time

load_dotenv()

def compare_encoding_schemes():
    """Compare different encoding schemes on extended conversations."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in .env file")
        return
    
    print("🔬 Extended Conversation Analysis with Different Encodings")
    print("=" * 60)
    
    # Test configurations
    encodings = ["hash", "advanced"]  # Start with 2 encodings
    conversation_lengths = [40]  # Start with 40 turns
    topics = [
        "What is the nature of consciousness and how might AI systems develop it?",
        "How do complex systems emerge from simple rules, and what does this mean for understanding intelligence?"
    ]  # Start with 2 topics
    
    results = {}
    
    for encoding in encodings:
        print(f"\n🧮 Testing {encoding.upper()} encoding...")
        results[encoding] = {}
        
        for length in conversation_lengths:
            print(f"\n  📏 Conversation length: {length} turns")
            results[encoding][length] = {}
            
            for i, topic in enumerate(topics):
                print(f"\n    💭 Topic {i+1}: {topic[:50]}...")
                
                # Create system with current encoding
                system = SimpleTwoAgentSystem(
                    agent_a_prompt="""You are Dr. Sarah Chen, a cognitive scientist and AI researcher. 
                    You approach questions systematically, often drawing connections between neuroscience, 
                    psychology, and artificial intelligence. You ask probing questions and build on ideas methodically.""",
                    
                    agent_b_prompt="""You are Professor Marcus Rivera, a philosopher of mind and technology ethicist. 
                    You think deeply about the implications of AI and consciousness, often bringing up ethical 
                    considerations and philosophical paradoxes. You enjoy exploring thought experiments.""",
                    
                    max_turns=length,
                    encoding_type=encoding,
                    state_dimension=64,
                    noise_scale=0.005  # Reduced noise for longer conversations
                )
                
                # Run conversation
                start_time = time.time()
                result = system.run_conversation(topic, verbose=False)
                duration = time.time() - start_time
                
                # Analyze results
                analyzer = ChaosAnalyzer()
                
                traj_a = result['agent_a_trajectory']
                traj_b = result['agent_b_trajectory']
                
                # Calculate metrics
                lyap_a = analyzer.estimate_lyapunov_from_single_trajectory(traj_a)
                lyap_b = analyzer.estimate_lyapunov_from_single_trajectory(traj_b)
                
                final_divergence = np.linalg.norm(traj_a[-1] - traj_b[-1])
                trajectory_variance = np.var(np.linalg.norm(traj_a - traj_a[0], axis=1))
                
                # State complexity over time
                state_complexity = [np.std(state) for state in traj_a]
                avg_complexity = np.mean(state_complexity)
                
                results[encoding][length][f"topic_{i+1}"] = {
                    "lyapunov_a": lyap_a,
                    "lyapunov_b": lyap_b,
                    "final_divergence": final_divergence,
                    "trajectory_variance": trajectory_variance,
                    "average_complexity": avg_complexity,
                    "conversation_duration": duration,
                    "total_messages": len([m for m in result['conversation_history'] if m['speaker'].startswith('Agent')]),
                    "encoding_info": result['agent_a_encoding_info']
                }
                
                print(f"      ✅ λ_A: {lyap_a:.6f}, λ_B: {lyap_b:.6f}, Final Div: {final_divergence:.3f}")
    
    return results

def analyze_encoding_performance(results):
    """Analyze and compare encoding performance."""
    
    print("\n📊 ENCODING PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Aggregate results by encoding
    encoding_stats = {}
    
    for encoding in results.keys():
        metrics = []
        for length in results[encoding].keys():
            for topic in results[encoding][length].keys():
                data = results[encoding][length][topic]
                metrics.append({
                    'lyapunov_max': max(data['lyapunov_a'], data['lyapunov_b']),
                    'lyapunov_avg': (data['lyapunov_a'] + data['lyapunov_b']) / 2,
                    'final_divergence': data['final_divergence'],
                    'complexity': data['average_complexity'],
                    'duration': data['conversation_duration']
                })
        
        # Calculate aggregate statistics
        encoding_stats[encoding] = {
            'avg_lyapunov': np.mean([m['lyapunov_avg'] for m in metrics]),
            'std_lyapunov': np.std([m['lyapunov_avg'] for m in metrics]),
            'avg_divergence': np.mean([m['final_divergence'] for m in metrics]),
            'avg_complexity': np.mean([m['complexity'] for m in metrics]),
            'avg_duration': np.mean([m['duration'] for m in metrics]),
            'chaos_strength': np.mean([m['lyapunov_max'] for m in metrics])
        }
    
    # Print comparison
    print(f"{'Encoding':<12} {'Avg λ':<12} {'Std λ':<12} {'Avg Div':<12} {'Complexity':<12} {'Duration':<12}")
    print("-" * 72)
    
    for encoding, stats in encoding_stats.items():
        print(f"{encoding:<12} {stats['avg_lyapunov']:<12.6f} {stats['std_lyapunov']:<12.6f} "
              f"{stats['avg_divergence']:<12.3f} {stats['avg_complexity']:<12.3f} {stats['avg_duration']:<12.1f}")
    
    # Determine best encoding for chaos detection
    best_encoding = max(encoding_stats.keys(), key=lambda x: encoding_stats[x]['chaos_strength'])
    print(f"\n🏆 Best encoding for chaos detection: {best_encoding.upper()}")
    print(f"   Chaos strength (max λ): {encoding_stats[best_encoding]['chaos_strength']:.6f}")
    
    return encoding_stats

def run_extended_conversation_with_best_encoding(best_encoding: str):
    """Run a detailed 50-turn conversation with the best encoding."""
    
    print(f"\n🚀 Running detailed 50-turn conversation with {best_encoding.upper()} encoding")
    print("=" * 60)
    
    system = SimpleTwoAgentSystem(
        agent_a_prompt="""You are Dr. Elena Vasquez, a complexity scientist studying emergent systems. 
        You're fascinated by how simple rules create complex behaviors. You often use analogies from 
        physics, biology, and mathematics. You build detailed theoretical frameworks.""",
        
        agent_b_prompt="""You are Dr. James Wright, a computer scientist and AI theorist. 
        You focus on practical implementations and computational mechanisms. You often propose 
        specific algorithms or architectures and test theoretical ideas against implementation realities.""",
        
        max_turns=50,
        encoding_type=best_encoding,
        state_dimension=64,
        noise_scale=0.003
    )
    
    # Run extended conversation
    topic = "How do we bridge the gap between current AI systems and artificial general intelligence?"
    result = system.run_conversation(topic, verbose=True)
    
    # Detailed analysis
    analyzer = ChaosAnalyzer()
    
    traj_a = result['agent_a_trajectory']
    traj_b = result['agent_b_trajectory']
    
    # Calculate comprehensive metrics
    lyap_a = analyzer.estimate_lyapunov_from_single_trajectory(traj_a)
    lyap_b = analyzer.estimate_lyapunov_from_single_trajectory(traj_b)
    
    # Correlation dimension estimation
    try:
        correlation_dim = analyzer.estimate_correlation_dimension(traj_a)
    except:
        correlation_dim = "Unable to calculate"
    
    # State evolution metrics
    state_norms_a = [np.linalg.norm(state) for state in traj_a]
    state_norms_b = [np.linalg.norm(state) for state in traj_b]
    
    divergence_over_time = [np.linalg.norm(traj_a[i] - traj_b[i]) for i in range(len(traj_a))]
    
    print(f"\n📈 DETAILED ANALYSIS RESULTS:")
    print(f"   Conversation length: {result['total_turns']} messages")
    print(f"   Duration: {result['conversation_duration']:.1f} seconds")
    print(f"   Encoding: {best_encoding} ({result['encoding_type']})")
    print(f"   State dimension: {result['state_dimension']}")
    print(f"   ")
    print(f"   Lyapunov exponents:")
    print(f"     Agent A: {lyap_a:.6f}")
    print(f"     Agent B: {lyap_b:.6f}")
    print(f"     Maximum: {max(lyap_a, lyap_b):.6f}")
    print(f"   ")
    print(f"   State dynamics:")
    print(f"     Final divergence: {divergence_over_time[-1]:.3f}")
    print(f"     Average state norm A: {np.mean(state_norms_a):.3f}")
    print(f"     Average state norm B: {np.mean(state_norms_b):.3f}")
    print(f"     Correlation dimension: {correlation_dim}")
    
    # Create visualization
    create_key_metrics_plot(result, save_path=f"extended_conversation_{best_encoding}_50turns.png")
    print(f"\n📊 Visualization saved: extended_conversation_{best_encoding}_50turns.png")
    
    # Sample conversation
    print(f"\n💬 CONVERSATION SAMPLE (first and last 3 exchanges):")
    print("-" * 50)
    
    agent_messages = [msg for msg in result['conversation_history'] if msg['speaker'].startswith('Agent')]
    
    print("BEGINNING:")
    for i, msg in enumerate(agent_messages[:6]):
        print(f"{i+1:2d}. {msg['speaker']}: {msg['content']}")
    
    print("\n...")
    print(f"[{len(agent_messages)-6} middle messages]")
    print("...")
    
    print("\nEND:")
    for i, msg in enumerate(agent_messages[-6:], len(agent_messages)-5):
        print(f"{i:2d}. {msg['speaker']}: {msg['content']}")
    
    return result

def main():
    """Main execution function."""
    
    print("🧬 EXTENDED CONVERSATION CHAOS ANALYSIS")
    print("Testing 40-50 turn conversations with advanced encoding schemes")
    print("=" * 70)
    
    # Step 1: Compare encoding schemes
    results = compare_encoding_schemes()
    
    # Step 2: Analyze performance
    encoding_stats = analyze_encoding_performance(results)
    
    # Step 3: Run detailed analysis with best encoding
    best_encoding = max(encoding_stats.keys(), key=lambda x: encoding_stats[x]['chaos_strength'])
    detailed_result = run_extended_conversation_with_best_encoding(best_encoding)
    
    print(f"\n✅ EXTENDED CONVERSATION ANALYSIS COMPLETE!")
    print(f"🏆 Best encoding: {best_encoding.upper()}")
    print(f"📊 Detailed analysis completed for 50-turn conversation")
    print(f"📈 Check visualization: extended_conversation_{best_encoding}_50turns.png")
    
    return results, encoding_stats, detailed_result

if __name__ == "__main__":
    main()