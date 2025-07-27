"""
Complete working demo with chaos analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from src.agent_system import SimpleTwoAgentSystem
from src.chaos_analysis import ChaosAnalyzer, SignalNoiseAnalyzer
from src.visualizations import DynamicalSystemVisualizer

load_dotenv()

def run_full_demo():
    """Run complete demo with analysis"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in .env file")
        return

    print("🤖 Two-Agent Dynamical System - Complete Demo")
    print("=" * 50)

    # Create system
    system = SimpleTwoAgentSystem(
        agent_a_prompt="""You are Agent A, a logical and systematic thinker. 
        You prefer structured approaches and ask clarifying questions. 
        Keep responses to 1-2 sentences.""",
        
        agent_b_prompt="""You are Agent B, an intuitive and creative thinker.
        You make unexpected connections and provide imaginative insights.
        Keep responses to 1-2 sentences.""",
        
        max_turns=5
    )

    # Run conversation
    print("\n🚀 Running two-agent conversation...")
    results = system.run_conversation("How do consciousness and artificial intelligence relate?")

    # Create multiple runs for chaos analysis
    print("\n🔬 Running perturbation experiment...")
    
    # Slightly different prompt
    system2 = SimpleTwoAgentSystem(
        agent_a_prompt="""You are Agent A, a logical and systematic thinker. 
        You prefer structured approaches and ask clarifying questions. 
        Keep responses to 1-2 sentences max.""",  # Slight change
        
        agent_b_prompt="""You are Agent B, an intuitive and creative thinker.
        You make unexpected connections and provide imaginative insights.
        Keep responses to 1-2 sentences max.""",  # Slight change
        
        max_turns=5
    )
    
    results2 = system2.run_conversation("How do consciousness and artificial intelligence relate?")

    # Analyze results
    print("\n📊 ANALYSIS RESULTS:")
    print("-" * 30)
    
    # Basic metrics
    print(f"Conversation 1: {len(results['conversation_history'])} messages")
    print(f"Conversation 2: {len(results2['conversation_history'])} messages") 
    
    # Show state evolution
    agent_a_final_1 = results['agent_a_trajectory'][0]
    agent_b_final_1 = results['agent_b_trajectory'][0]
    agent_a_final_2 = results2['agent_a_trajectory'][0]
    agent_b_final_2 = results2['agent_b_trajectory'][0]
    
    print(f"\nAgent A final state difference: {np.linalg.norm(agent_a_final_1 - agent_a_final_2):.4f}")
    print(f"Agent B final state difference: {np.linalg.norm(agent_b_final_1 - agent_b_final_2):.4f}")
    
    # Content analysis
    messages_1 = [msg['content'] for msg in results['conversation_history'] if msg['speaker'].startswith('Agent')]
    messages_2 = [msg['content'] for msg in results2['conversation_history'] if msg['speaker'].startswith('Agent')]
    
    # Simple content divergence
    content_similarity = 0
    min_len = min(len(messages_1), len(messages_2))
    for i in range(min_len):
        words_1 = set(messages_1[i].lower().split())
        words_2 = set(messages_2[i].lower().split())
        if words_1 and words_2:
            overlap = len(words_1.intersection(words_2))
            total = len(words_1.union(words_2))
            content_similarity += overlap / total if total > 0 else 0
    
    avg_content_similarity = content_similarity / min_len if min_len > 0 else 0
    print(f"Average content similarity: {avg_content_similarity:.4f}")
    
    if avg_content_similarity < 0.5:
        print("🎯 SENSITIVE DEPENDENCE DETECTED: Small prompt changes led to different conversations!")
    else:
        print("📊 Conversations remained similar despite prompt changes")

    # Signal vs Noise Analysis
    print(f"\n🎛️ SIGNAL VS NOISE ANALYSIS:")
    analyzer = SignalNoiseAnalyzer()
    
    signal_noise = analyzer.decompose_conversation(
        results['conversation_history'],
        {"agent_a": results['agent_a_trajectory'], "agent_b": results['agent_b_trajectory']}
    )
    
    print("Signal Components:")
    for metric, value in signal_noise["signal"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("Noise Components:")
    for metric, value in signal_noise["noise"].items():
        print(f"  {metric}: {value:.4f}")

    # Display conversations
    print(f"\n📝 CONVERSATION 1:")
    print("-" * 20)
    for i, msg in enumerate(results['conversation_history']):
        speaker = msg['speaker']
        content = msg['content']
        if speaker == "System":
            print(f"\n🎯 {speaker}: {content}")
        else:
            print(f"\n{i}. {speaker}: {content}")

    print(f"\n📝 CONVERSATION 2 (with perturbation):")
    print("-" * 20)
    for i, msg in enumerate(results2['conversation_history']):
        speaker = msg['speaker']
        content = msg['content']
        if speaker == "System":
            print(f"\n🎯 {speaker}: {content}")
        else:
            print(f"\n{i}. {speaker}: {content}")

    # Create comprehensive visualizations
    print(f"\n📊 Creating comprehensive visualizations...")
    visualizer = DynamicalSystemVisualizer()
    
    # Generate comprehensive analysis plot
    visualizer.create_comprehensive_analysis(
        results, 
        results2, 
        save_path="comprehensive_analysis.png"
    )
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📈 Check 'comprehensive_analysis.png' for detailed visualizations")
    
    return results, results2

if __name__ == "__main__":
    run_full_demo()