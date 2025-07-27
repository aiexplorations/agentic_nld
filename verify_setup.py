"""
Verify that the complete chaos analysis system works correctly.
"""

import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Test imports
try:
    from src.agent_system import SimpleTwoAgentSystem
    from src.chaos_analysis import ChaosAnalyzer, SignalNoiseAnalyzer
    from src.simple_viz import create_key_metrics_plot
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

load_dotenv()

def verify_system():
    """Verify the complete system works."""
    
    print("🔍 SYSTEM VERIFICATION")
    print("=" * 30)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set (demos will not work)")
    else:
        print("✅ OpenAI API key configured")
    
    # Test system components
    print("\n📦 Testing core components...")
    
    # Test agent system
    try:
        system = SimpleTwoAgentSystem(
            agent_a_prompt="Test agent A",
            agent_b_prompt="Test agent B", 
            max_turns=2
        )
        print("✅ Agent system initialization works")
    except Exception as e:
        print(f"❌ Agent system error: {e}")
        return False
    
    # Test chaos analyzer
    try:
        analyzer = ChaosAnalyzer()
        # Test with dummy data
        dummy_traj = np.random.randn(10, 5)
        lyap = analyzer.estimate_lyapunov_from_single_trajectory(dummy_traj)
        print(f"✅ Chaos analyzer works (test Lyapunov: {lyap:.6f})")
    except Exception as e:
        print(f"❌ Chaos analyzer error: {e}")
        return False
    
    # Test signal/noise analyzer
    try:
        signal_analyzer = SignalNoiseAnalyzer()
        dummy_conv = [{"speaker": "Agent_A", "content": "Hello world test message"}]
        dummy_trajs = {"agent_a": np.random.randn(5, 3), "agent_b": np.random.randn(5, 3)}
        result = signal_analyzer.decompose_conversation(dummy_conv, dummy_trajs)
        print("✅ Signal/noise analyzer works")
    except Exception as e:
        print(f"❌ Signal/noise analyzer error: {e}")
        return False
    
    # Test visualization
    try:
        # Create dummy results for visualization test
        dummy_results = {
            'agent_a_trajectory': np.random.randn(8, 5),
            'agent_b_trajectory': np.random.randn(8, 5),
            'conversation_history': [
                {"speaker": "Agent_A", "content": "Test message 1"},
                {"speaker": "Agent_B", "content": "Test message 2"}
            ]
        }
        
        create_key_metrics_plot(dummy_results, save_path="test_viz.png")
        print("✅ Visualization system works")
        
        # Clean up test file
        if Path("test_viz.png").exists():
            Path("test_viz.png").unlink()
            
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False
    
    # Check file structure
    print("\n📁 Checking file structure...")
    
    required_files = [
        "src/agent_system.py",
        "src/chaos_analysis.py", 
        "src/visualizations.py",
        "src/simple_viz.py",
        "demo.py",
        "test_long_conversation.py",
        "TECHNICAL_REPORT.md",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    
    print("\n🎉 SYSTEM VERIFICATION COMPLETE!")
    print("✅ All components working correctly")
    print("📖 Ready to run experiments and generate technical reports")
    
    return True

if __name__ == "__main__":
    success = verify_system()
    if success:
        print("\n🚀 To get started:")
        print("   1. Set OPENAI_API_KEY in .env file") 
        print("   2. Run: python demo.py")
        print("   3. Read: TECHNICAL_REPORT.md")
    else:
        print("\n❌ System verification failed - please check errors above")