"""
Long conversation demo to explore chaos dynamics over extended interactions.
Simulates podcast-length conversations.
"""

import os
import numpy as np
from dotenv import load_dotenv
from src.agent_system import SimpleTwoAgentSystem
from src.chaos_analysis import ChaosAnalyzer, SignalNoiseAnalyzer
from src.visualizations import DynamicalSystemVisualizer
from src.simple_viz import create_key_metrics_plot

load_dotenv()

def run_podcast_length_conversation(turns: int = 25):
    """
    Run a podcast-length conversation between agents.
    
    Args:
        turns: Number of turns for each agent (total messages = turns * 2)
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in .env file")
        return

    print(f"🎙️ Podcast-Length Conversation Demo ({turns} turns each)")
    print("=" * 60)

    # Create system with podcast-style prompts
    system = SimpleTwoAgentSystem(
        agent_a_prompt="""You are Dr. Sarah Chen, a cognitive scientist and AI researcher. 
        You bring scientific rigor to discussions about consciousness and AI.
        You ask probing questions and build on ideas systematically.
        Keep responses natural and conversational, 2-3 sentences.""",
        
        agent_b_prompt="""You are Marcus Rodriguez, a philosopher and technology ethicist.
        You explore the deeper implications and unexpected connections in AI/consciousness discussions.
        You often provide thought-provoking analogies and challenge assumptions.
        Keep responses natural and conversational, 2-3 sentences.""",
        
        max_turns=turns
    )

    # Run main conversation
    print(f"\n🚀 Starting {turns*2}-message conversation...")
    results = system.run_conversation("We're here to explore one of the most fascinating questions of our time: What is the relationship between consciousness and artificial intelligence? Let's dive deep.")

    print(f"\n📊 Conversation completed with {results['total_turns']} total messages")
    print(f"🧠 Final Agent A state norm: {np.linalg.norm(results['agent_a_trajectory'][-1]):.3f}")
    print(f"🧠 Final Agent B state norm: {np.linalg.norm(results['agent_b_trajectory'][-1]):.3f}")

    return results

def run_perturbation_study(turns: int = 25):
    """
    Run perturbation analysis with longer conversations.
    """
    print(f"\n🔬 PERTURBATION STUDY ({turns} turns)")
    print("-" * 40)
    
    # Create two systems with slightly different prompts
    system1 = SimpleTwoAgentSystem(
        agent_a_prompt="""You are Dr. Sarah Chen, a cognitive scientist and AI researcher. 
        You bring scientific rigor to discussions about consciousness and AI.
        You ask probing questions and build on ideas systematically.
        Keep responses natural and conversational, 2-3 sentences.""",
        
        agent_b_prompt="""You are Marcus Rodriguez, a philosopher and technology ethicist.
        You explore the deeper implications and unexpected connections in AI/consciousness discussions.
        You often provide thought-provoking analogies and challenge assumptions.
        Keep responses natural and conversational, 2-3 sentences.""",
        
        max_turns=turns
    )
    
    system2 = SimpleTwoAgentSystem(
        agent_a_prompt="""You are Dr. Sarah Chen, a cognitive scientist and AI researcher. 
        You bring scientific rigor to discussions about consciousness and AI.
        You ask probing questions and build on ideas systematically.
        Keep responses natural and conversational, 2-3 sentences max.""",  # Added "max"
        
        agent_b_prompt="""You are Marcus Rodriguez, a philosopher and technology ethicist.
        You explore the deeper implications and unexpected connections in AI/consciousness discussions.
        You often provide thought-provoking analogies and challenge assumptions.
        Keep responses natural and conversational, 2-3 sentences max.""",  # Added "max"
        
        max_turns=turns
    )
    
    # Run both conversations
    prompt = "We're here to explore one of the most fascinating questions of our time: What is the relationship between consciousness and artificial intelligence? Let's dive deep."
    
    print("🎙️ Running conversation 1...")
    results1 = system1.run_conversation(prompt)
    
    print("🎙️ Running conversation 2 (with perturbation)...")
    results2 = system2.run_conversation(prompt)
    
    return results1, results2

def analyze_long_conversation_dynamics(results1, results2=None):
    """
    Analyze the dynamics of longer conversations.
    """
    print(f"\n📈 LONG CONVERSATION ANALYSIS:")
    print("-" * 40)
    
    traj_a1 = results1['agent_a_trajectory']
    traj_b1 = results1['agent_b_trajectory']
    
    # Basic trajectory analysis
    print(f"📊 Trajectory lengths: A={len(traj_a1)}, B={len(traj_b1)}")
    
    # State evolution analysis
    magnitude_a = np.linalg.norm(traj_a1, axis=1)
    magnitude_b = np.linalg.norm(traj_b1, axis=1)
    
    print(f"🔄 State magnitude evolution:")
    print(f"   Agent A: {magnitude_a[0]:.3f} → {magnitude_a[-1]:.3f} (Δ={magnitude_a[-1]-magnitude_a[0]:.3f})")
    print(f"   Agent B: {magnitude_b[0]:.3f} → {magnitude_b[-1]:.3f} (Δ={magnitude_b[-1]-magnitude_b[0]:.3f})")
    
    # Divergence analysis
    divergence = np.linalg.norm(traj_a1 - traj_b1, axis=1)
    initial_divergence = divergence[0]
    final_divergence = divergence[-1]
    max_divergence = np.max(divergence)
    
    print(f"↔️  State divergence:")
    print(f"   Initial: {initial_divergence:.3f}")
    print(f"   Final: {final_divergence:.3f}")
    print(f"   Maximum: {max_divergence:.3f}")
    print(f"   Trend: {'Diverging' if final_divergence > initial_divergence else 'Converging'}")
    
    # Chaos analysis
    analyzer = ChaosAnalyzer()
    
    if len(traj_a1) > 5:
        lyap_a = analyzer.estimate_lyapunov_from_single_trajectory(traj_a1)
        lyap_b = analyzer.estimate_lyapunov_from_single_trajectory(traj_b1)
        
        print(f"🌪️  Lyapunov exponents:")
        print(f"   Agent A: {lyap_a:.6f}")
        print(f"   Agent B: {lyap_b:.6f}")
        
        if lyap_a > 0 or lyap_b > 0:
            print("   🎯 CHAOTIC BEHAVIOR DETECTED!")
        else:
            print("   📊 System appears stable/non-chaotic")
    
    # Perturbation analysis
    if results2:
        traj_a2 = results2['agent_a_trajectory']
        traj_b2 = results2['agent_b_trajectory']
        
        # Compare final states
        min_len = min(len(traj_a1), len(traj_a2))
        final_diff_a = np.linalg.norm(traj_a1[:min_len][-1] - traj_a2[:min_len][-1])
        final_diff_b = np.linalg.norm(traj_b1[:min_len][-1] - traj_b2[:min_len][-1])
        
        print(f"\n🔬 PERTURBATION ANALYSIS:")
        print(f"   Final state differences:")
        print(f"   Agent A: {final_diff_a:.6f}")
        print(f"   Agent B: {final_diff_b:.6f}")
        
        # Content analysis
        messages1 = [msg['content'] for msg in results1['conversation_history'] if msg['speaker'].startswith('Agent')]
        messages2 = [msg['content'] for msg in results2['conversation_history'] if msg['speaker'].startswith('Agent')]
        
        # Calculate average content similarity
        similarities = []
        min_msgs = min(len(messages1), len(messages2))
        
        for i in range(min_msgs):
            words1 = set(messages1[i].lower().split())
            words2 = set(messages2[i].lower().split())
            if words1 and words2:
                overlap = len(words1.intersection(words2))
                total = len(words1.union(words2))
                similarity = overlap / total if total > 0 else 0
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        print(f"   Average content similarity: {avg_similarity:.4f}")
        
        if avg_similarity < 0.3:
            print("   🎯 STRONG SENSITIVE DEPENDENCE!")
        elif avg_similarity < 0.5:
            print("   ⚠️  Moderate sensitive dependence")
        else:
            print("   📊 Conversations remained similar")

def create_conversation_summary(results, title="Conversation"):
    """
    Create a summary of the conversation showing key moments.
    """
    print(f"\n📝 {title.upper()} SUMMARY:")
    print("-" * (len(title) + 10))
    
    conversation = results['conversation_history']
    agent_messages = [msg for msg in conversation if msg['speaker'].startswith('Agent')]
    
    # Show first few, middle few, and last few messages
    n = len(agent_messages)
    
    if n <= 10:
        # Show all messages
        for i, msg in enumerate(agent_messages):
            print(f"{i+1:2d}. {msg['speaker']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    else:
        # Show first 3, middle 2, last 3
        print("🟢 Opening:")
        for i in range(3):
            msg = agent_messages[i]
            print(f"{i+1:2d}. {msg['speaker']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
        
        print("\n🟡 Middle:")
        mid_start = n//2 - 1
        for i in range(mid_start, mid_start + 2):
            msg = agent_messages[i]
            print(f"{i+1:2d}. {msg['speaker']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
        
        print("\n🔴 Ending:")
        for i in range(n-3, n):
            msg = agent_messages[i]
            print(f"{i+1:2d}. {msg['speaker']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")

def main():
    """
    Main execution for long conversation analysis.
    """
    print("🎙️ Long Conversation Dynamics Analysis")
    print("=" * 50)
    
    # Run different conversation lengths
    conversation_lengths = [10, 20, 30]  # Podcast-style lengths
    
    for turns in conversation_lengths:
        print(f"\n" + "="*60)
        print(f"🎙️ ANALYZING {turns}-TURN CONVERSATIONS")
        print("="*60)
        
        # Run perturbation study
        results1, results2 = run_perturbation_study(turns)
        
        # Analyze dynamics
        analyze_long_conversation_dynamics(results1, results2)
        
        # Create visualizations
        print(f"\n📊 Creating visualizations for {turns}-turn conversation...")
        
        # Simple focused plot
        create_key_metrics_plot(results1, results2, f"conversation_{turns}turns_metrics.png")
        
        # Show conversation summary
        create_conversation_summary(results1, f"{turns}-Turn Conversation")
        
        print(f"\n✅ {turns}-turn analysis complete!")
    
    print(f"\n🎉 All conversation length analyses completed!")
    print(f"📈 Check the generated PNG files for detailed visualizations")

if __name__ == "__main__":
    main()