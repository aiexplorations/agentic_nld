"""
Test a single long conversation to see chaos dynamics.
"""

import os
import numpy as np
from dotenv import load_dotenv
from src.agent_system import SimpleTwoAgentSystem
from src.simple_viz import create_key_metrics_plot

load_dotenv()

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in .env file")
        return

    print("🎙️ Long Conversation Test (20 turns)")
    print("=" * 40)

    # Create system
    system = SimpleTwoAgentSystem(
        agent_a_prompt="You are a logical researcher. Ask probing questions. Be concise (1-2 sentences).",
        agent_b_prompt="You are a creative philosopher. Make unexpected connections. Be concise (1-2 sentences).",
        max_turns=20
    )

    # Run conversation
    print("\n🚀 Starting conversation...")
    results = system.run_conversation("What is consciousness?")

    # Basic analysis
    traj_a = results['agent_a_trajectory']
    traj_b = results['agent_b_trajectory']
    
    print(f"\n📊 Results:")
    print(f"   Total messages: {results['total_turns']}")
    print(f"   Trajectory length: {len(traj_a)}")
    print(f"   Final state norms: A={np.linalg.norm(traj_a[-1]):.3f}, B={np.linalg.norm(traj_b[-1]):.3f}")
    
    # State divergence
    divergence = np.linalg.norm(traj_a - traj_b, axis=1)
    print(f"   State divergence: {divergence[0]:.3f} → {divergence[-1]:.3f}")
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    create_key_metrics_plot(results, save_path="long_conversation_20turns.png")
    
    # Show conversation sample
    print(f"\n📝 Conversation sample:")
    messages = [msg for msg in results['conversation_history'] if msg['speaker'].startswith('Agent')]
    for i, msg in enumerate(messages[:6]):  # First 6 messages
        print(f"{i+1}. {msg['speaker']}: {msg['content'][:80]}...")
    
    if len(messages) > 6:
        print(f"... [{len(messages)-6} more messages] ...")
        for i, msg in enumerate(messages[-2:], len(messages)-1):  # Last 2 messages
            print(f"{i}. {msg['speaker']}: {msg['content'][:80]}...")

    print(f"\n✅ Analysis complete! Check 'long_conversation_20turns.png'")

if __name__ == "__main__":
    main()