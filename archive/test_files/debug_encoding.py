"""
Debug the encoding systems with a simple test.
"""

import os
from dotenv import load_dotenv
from src.agent_system import SimpleTwoAgentSystem
import time

load_dotenv()

def test_basic_encoding():
    """Test basic encoding functionality."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in .env file")
        return
    
    print("🔧 Testing basic encoding functionality...")
    
    # Test hash encoding (5 turns)
    print("\n1. Testing HASH encoding (5 turns)...")
    system_hash = SimpleTwoAgentSystem(
        agent_a_prompt="You are a logical thinker. Be concise.",
        agent_b_prompt="You are a creative thinker. Be concise.",
        max_turns=5,
        encoding_type="hash"
    )
    
    start_time = time.time()
    result_hash = system_hash.run_conversation("What is intelligence?", verbose=False)
    hash_duration = time.time() - start_time
    
    print(f"   ✅ Hash encoding completed in {hash_duration:.1f}s")
    print(f"   Messages: {result_hash['total_turns']}")
    
    # Test advanced encoding (5 turns)
    print("\n2. Testing ADVANCED encoding (5 turns)...")
    system_advanced = SimpleTwoAgentSystem(
        agent_a_prompt="You are a logical thinker. Be concise.",
        agent_b_prompt="You are a creative thinker. Be concise.",
        max_turns=5,
        encoding_type="advanced"
    )
    
    start_time = time.time()
    result_advanced = system_advanced.run_conversation("What is intelligence?", verbose=False)
    advanced_duration = time.time() - start_time
    
    print(f"   ✅ Advanced encoding completed in {advanced_duration:.1f}s")
    print(f"   Messages: {result_advanced['total_turns']}")
    
    # Test longer conversation (10 turns)
    print("\n3. Testing longer conversation (10 turns with advanced)...")
    system_long = SimpleTwoAgentSystem(
        agent_a_prompt="You are a logical thinker. Be concise.",
        agent_b_prompt="You are a creative thinker. Be concise.",
        max_turns=10,
        encoding_type="advanced"
    )
    
    start_time = time.time()
    result_long = system_long.run_conversation("What is consciousness?", verbose=False)
    long_duration = time.time() - start_time
    
    print(f"   ✅ Long conversation completed in {long_duration:.1f}s")
    print(f"   Messages: {result_long['total_turns']}")
    
    # Compare encoding info
    print(f"\n📊 ENCODING COMPARISON:")
    print(f"Hash encoding info: {result_hash['agent_a_encoding_info']}")
    print(f"Advanced encoding info: {result_advanced['agent_a_encoding_info']['encoding_type']}")
    
    return result_hash, result_advanced, result_long

if __name__ == "__main__":
    test_basic_encoding()