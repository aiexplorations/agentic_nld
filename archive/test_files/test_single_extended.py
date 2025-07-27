"""
Test a single extended conversation to analyze timing and performance.
"""

import os
import numpy as np
from dotenv import load_dotenv
from src.agent_system import SimpleTwoAgentSystem
from src.chaos_analysis import ChaosAnalyzer
from src.simple_viz import create_key_metrics_plot
import time

load_dotenv()

def run_single_extended_conversation():
    """Run a single 40-turn conversation with advanced encoding."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in .env file")
        return
    
    print("🚀 Running Single 40-Turn Extended Conversation")
    print("=" * 50)
    
    # Create system with advanced encoding
    system = SimpleTwoAgentSystem(
        agent_a_prompt="""You are Dr. Sarah Chen, a cognitive scientist and AI researcher. 
        You approach questions systematically, often drawing connections between neuroscience, 
        psychology, and artificial intelligence. You ask probing questions and build on ideas methodically.
        Keep responses concise but substantive.""",
        
        agent_b_prompt="""You are Professor Marcus Rivera, a philosopher of mind and technology ethicist. 
        You think deeply about the implications of AI and consciousness, often bringing up ethical 
        considerations and philosophical paradoxes. You enjoy exploring thought experiments.
        Keep responses concise but substantive.""",
        
        max_turns=40,
        encoding_type="advanced",
        state_dimension=64,
        noise_scale=0.005
    )
    
    topic = "What is the nature of consciousness and how might AI systems develop it?"
    
    print(f"📝 Topic: {topic}")
    print(f"🧮 Encoding: Advanced multi-feature")
    print(f"📏 Target length: 40 turns per agent (80 total messages)")
    print()
    
    # Run conversation with timing
    start_time = time.time()
    result = system.run_conversation(topic, verbose=True)
    total_duration = time.time() - start_time
    
    print(f"\n⏱️  TIMING ANALYSIS:")
    print(f"   Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"   Messages generated: {result['total_turns']}")
    print(f"   Average time per message: {total_duration/result['total_turns']:.1f}s")
    print(f"   Conversation duration: {result['conversation_duration']:.1f}s")
    
    # Analyze chaos dynamics
    print(f"\n🔬 CHAOS ANALYSIS:")
    analyzer = ChaosAnalyzer()
    
    traj_a = result['agent_a_trajectory']
    traj_b = result['agent_b_trajectory']
    
    lyap_a = analyzer.estimate_lyapunov_from_single_trajectory(traj_a)
    lyap_b = analyzer.estimate_lyapunov_from_single_trajectory(traj_b)
    
    final_divergence = np.linalg.norm(traj_a[-1] - traj_b[-1])
    trajectory_variance = np.var(np.linalg.norm(traj_a - traj_a[0], axis=1))
    
    print(f"   Lyapunov exponent A: {lyap_a:.6f}")
    print(f"   Lyapunov exponent B: {lyap_b:.6f}")
    print(f"   Maximum Lyapunov: {max(lyap_a, lyap_b):.6f}")
    print(f"   Final divergence: {final_divergence:.3f}")
    print(f"   Trajectory variance: {trajectory_variance:.3f}")
    
    # Check for chaos indicators
    if max(lyap_a, lyap_b) > 0:
        print(f"   🎯 CHAOS DETECTED: Positive Lyapunov exponent!")
    else:
        print(f"   📊 No chaos detected in this conversation")
    
    # State evolution analysis
    state_norms_a = [np.linalg.norm(state) for state in traj_a]
    state_norms_b = [np.linalg.norm(state) for state in traj_b]
    
    print(f"\n📈 STATE EVOLUTION:")
    print(f"   Initial state norm A: {state_norms_a[0]:.3f}")
    print(f"   Final state norm A: {state_norms_a[-1]:.3f}")
    print(f"   Initial state norm B: {state_norms_b[0]:.3f}")
    print(f"   Final state norm B: {state_norms_b[-1]:.3f}")
    print(f"   Average state norm A: {np.mean(state_norms_a):.3f}")
    print(f"   Average state norm B: {np.mean(state_norms_b):.3f}")
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    create_key_metrics_plot(result, save_path="single_extended_40turns.png")
    print(f"   Saved: single_extended_40turns.png")
    
    # Sample conversation
    print(f"\n💬 CONVERSATION SAMPLE:")
    print("=" * 40)
    
    agent_messages = [msg for msg in result['conversation_history'] if msg['speaker'].startswith('Agent')]
    
    print("FIRST 4 EXCHANGES:")
    for i, msg in enumerate(agent_messages[:8]):
        turn_num = (i // 2) + 1
        agent_letter = "A" if msg['speaker'] == 'Agent_A' else "B"
        print(f"Turn {turn_num}{agent_letter}: {msg['content']}")
        print()
    
    print("...")
    print(f"[{len(agent_messages)-8} more messages]")
    print("...")
    
    print("LAST 2 EXCHANGES:")
    for i, msg in enumerate(agent_messages[-4:]):
        turn_num = ((len(agent_messages) - 4 + i) // 2) + 1
        agent_letter = "A" if msg['speaker'] == 'Agent_A' else "B"
        print(f"Turn {turn_num}{agent_letter}: {msg['content']}")
        print()
    
    return result

if __name__ == "__main__":
    result = run_single_extended_conversation()
    print(f"\n✅ Extended conversation analysis complete!")