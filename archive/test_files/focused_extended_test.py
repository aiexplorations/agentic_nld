"""
Focused test of extended conversation with advanced encoding.
"""

import os
import numpy as np
from dotenv import load_dotenv
from src.agent_system import SimpleTwoAgentSystem
from src.chaos_analysis import ChaosAnalyzer
import time

load_dotenv()

def run_focused_extended_test():
    """Run a focused 30-turn conversation with advanced encoding."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in .env file")
        return
    
    print("🎯 FOCUSED EXTENDED CONVERSATION TEST")
    print("=" * 50)
    
    # Create system with advanced encoding
    system = SimpleTwoAgentSystem(
        agent_a_prompt="""You are Dr. Sarah Chen, a cognitive scientist. 
        You ask probing questions and explore consciousness systematically. 
        Be concise but build on previous ideas.""",
        
        agent_b_prompt="""You are Professor Rivera, a philosopher of mind. 
        You explore ethical implications and thought experiments about consciousness. 
        Be concise but engage deeply with the ideas.""",
        
        max_turns=15,  # 30 total messages
        encoding_type="advanced",
        state_dimension=64,
        noise_scale=0.01
    )
    
    topic = "What is consciousness and can machines achieve it?"
    
    print(f"📝 Topic: {topic}")
    print(f"🧮 Encoding: Advanced multi-feature")
    print(f"📏 Length: 15 turns per agent (30 total messages)")
    print()
    
    # Run conversation
    start_time = time.time()
    result = system.run_conversation(topic, verbose=True)
    duration = time.time() - start_time
    
    print(f"\n⏱️  TIMING: {duration:.1f}s ({result['total_turns']} messages)")
    
    # Analyze chaos
    analyzer = ChaosAnalyzer()
    traj_a = result['agent_a_trajectory']
    traj_b = result['agent_b_trajectory']
    
    lyap_a = analyzer.estimate_lyapunov_from_single_trajectory(traj_a)
    lyap_b = analyzer.estimate_lyapunov_from_single_trajectory(traj_b)
    
    final_divergence = np.linalg.norm(traj_a[-1] - traj_b[-1])
    
    print(f"\n🔬 CHAOS ANALYSIS:")
    print(f"   Lyapunov A: {lyap_a:.6f}")
    print(f"   Lyapunov B: {lyap_b:.6f}")
    print(f"   Max Lyapunov: {max(lyap_a, lyap_b):.6f}")
    print(f"   Final divergence: {final_divergence:.3f}")
    
    if max(lyap_a, lyap_b) > 0:
        print(f"   🎯 CHAOS DETECTED!")
    else:
        print(f"   📊 No chaos in this conversation")
    
    # Show conversation sample
    print(f"\n💬 CONVERSATION SAMPLE:")
    print("-" * 40)
    agent_messages = [msg for msg in result['conversation_history'] if msg['speaker'].startswith('Agent')]
    
    for i, msg in enumerate(agent_messages[:6]):
        speaker = "A" if msg['speaker'] == 'Agent_A' else "B"
        print(f"{i+1}. Agent {speaker}: {msg['content']}")
    
    if len(agent_messages) > 6:
        print("...")
        print(f"[{len(agent_messages)-6} more messages]")
    
    return result

if __name__ == "__main__":
    result = run_focused_extended_test()
    print(f"\n✅ Focused extended test complete!")