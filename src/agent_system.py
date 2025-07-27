"""
Simplified two-agent system without LangGraph for initial testing
Enhanced with advanced encoding schemes and longer conversation support
"""

import numpy as np
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os
import time
from dotenv import load_dotenv
from .advanced_encoding import AdvancedTextEncoder, SemanticEncoder

load_dotenv()


class SimpleAgent:
    """Simplified agent without LangGraph"""
    
    def __init__(self, agent_id: str, system_prompt: str, state_dimension: int = 64, 
                 encoding_type: str = "hash", noise_scale: float = 0.01):
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.state_dimension = state_dimension
        self.encoding_type = encoding_type
        self.noise_scale = noise_scale
        
        # Initialize appropriate encoder
        if encoding_type == "advanced":
            self.encoder = AdvancedTextEncoder(state_dimension)
        elif encoding_type == "semantic":
            self.encoder = SemanticEncoder(state_dimension)
        else:
            self.encoder = None  # Use original hash-based encoding
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize state
        self.state_vector = np.random.randn(state_dimension) * 0.1
        self.message_history = []
        self.context_memory = []
        self.trajectory = [self.state_vector.copy()]  # Track full trajectory
        
        # Extended conversation support
        self.turn_count = 0
        self.response_times = []
    
    def encode_text(self, text: str) -> np.ndarray:
        """Text encoding using selected encoder"""
        if self.encoder is not None:
            return self.encoder.encode_text(text)
        else:
            # Original hash-based encoding
            encoding = np.zeros(self.state_dimension)
            words = text.lower().split()
            for i, word in enumerate(words[:self.state_dimension]):
                encoding[i % self.state_dimension] += hash(word) % 100 / 100.0
            return encoding
    
    def update_state(self, incoming_message: str = "") -> None:
        """Update agent state based on incoming message"""
        if incoming_message:
            encoded_input = self.encode_text(incoming_message)
            self.context_memory.append(incoming_message)
            if len(self.context_memory) > 5:
                self.context_memory.pop(0)
        else:
            encoded_input = np.zeros(self.state_dimension)
        
        # Nonlinear state evolution
        interaction = np.tanh(self.state_vector * 0.5 + encoded_input * 0.3)
        
        # Memory influence
        memory_influence = np.zeros(self.state_dimension)
        if self.context_memory:
            for i, msg in enumerate(self.context_memory[-3:]):
                weight = 0.5 ** (len(self.context_memory) - i)
                memory_influence += weight * self.encode_text(msg)[:self.state_dimension]
        
        # Update state with configurable noise
        new_state = (
            0.6 * self.state_vector +
            0.3 * interaction +
            0.1 * memory_influence +
            self.noise_scale * np.random.randn(self.state_dimension)  # configurable noise
        )
        
        self.state_vector = np.tanh(new_state)
        self.trajectory.append(self.state_vector.copy())  # Store in trajectory
        self.turn_count += 1
    
    def generate_response(self, incoming_message: str, conversation_history: List[Dict] = None) -> str:
        """Generate response using LLM"""
        
        start_time = time.time()
        
        # Update internal state first
        self.update_state(incoming_message)
        
        # Create context - use more history for longer conversations
        context = ""
        if conversation_history:
            # Adaptive context window based on conversation length
            context_window = min(5, max(3, len(conversation_history) // 10))
            recent_history = conversation_history[-context_window:]
            for msg in recent_history:
                context += f"{msg['speaker']}: {msg['content']}\n"
        
        # Enhanced state summary for longer conversations
        state_sum = np.sum(self.state_vector)
        state_complexity = np.std(self.state_vector)
        
        if state_complexity > 0.5:
            if state_sum > 0.2:
                mood = "highly engaged and enthusiastic"
            elif state_sum < -0.2:
                mood = "deeply analytical and contemplative"
            else:
                mood = "dynamically balanced"
        else:
            if state_sum > 0.2:
                mood = "engaged and positive"
            elif state_sum < -0.2:
                mood = "thoughtful and analytical"
            else:
                mood = "balanced"
        
        # Enhanced prompt for longer conversations
        prompt = f"""
{self.system_prompt}

Current state: {mood} (Turn {self.turn_count})
Encoding type: {self.encoding_type}

Recent conversation:
{context}

Latest message: {incoming_message}

Respond as {self.agent_id}. Keep it concise but substantive (1-2 sentences).
"""
        
        # Generate response
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Track response time
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        return response.content
    
    def get_encoding_info(self) -> Dict[str, Any]:
        """Get information about the encoding scheme used."""
        if self.encoder and hasattr(self.encoder, 'get_encoding_info'):
            return self.encoder.get_encoding_info()
        else:
            return {
                "encoding_type": "hash_based",
                "state_dimension": self.state_dimension,
                "description": "Simple hash-based word encoding"
            }


class SimpleTwoAgentSystem:
    """Simplified two-agent system"""
    
    def __init__(self, agent_a_prompt: str, agent_b_prompt: str, max_turns: int = 5,
                 encoding_type: str = "hash", state_dimension: int = 64, 
                 noise_scale: float = 0.01):
        self.agent_a = SimpleAgent("Agent_A", agent_a_prompt, state_dimension, encoding_type, noise_scale)
        self.agent_b = SimpleAgent("Agent_B", agent_b_prompt, state_dimension, encoding_type, noise_scale)
        self.max_turns = max_turns
        self.encoding_type = encoding_type
        self.state_dimension = state_dimension
        self.conversation_start_time = None
    
    def run_conversation(self, initial_prompt: str = "", verbose: bool = True) -> Dict[str, Any]:
        """Run conversation between agents"""
        
        self.conversation_start_time = time.time()
        conversation_history = []
        
        if initial_prompt:
            conversation_history.append({
                "speaker": "System",
                "content": initial_prompt,
                "timestamp": time.time()
            })
        
        # Start with Agent A
        current_message = initial_prompt
        
        for turn in range(self.max_turns * 2):  # Each agent gets max_turns
            
            if turn % 2 == 0:  # Agent A's turn
                if verbose:
                    print(f"Turn {turn//2 + 1} - Agent A thinking...")
                response = self.agent_a.generate_response(current_message, conversation_history)
                
                conversation_history.append({
                    "speaker": "Agent_A",
                    "content": response,
                    "timestamp": time.time(),
                    "turn": turn,
                    "agent_state_norm": np.linalg.norm(self.agent_a.state_vector),
                    "agent_turn_count": self.agent_a.turn_count
                })
                
                if verbose:
                    print(f"Agent A: {response}")
                current_message = response
                
            else:  # Agent B's turn
                if verbose:
                    print(f"Turn {turn//2 + 1} - Agent B thinking...")
                response = self.agent_b.generate_response(current_message, conversation_history)
                
                conversation_history.append({
                    "speaker": "Agent_B", 
                    "content": response,
                    "timestamp": time.time(),
                    "turn": turn,
                    "agent_state_norm": np.linalg.norm(self.agent_b.state_vector),
                    "agent_turn_count": self.agent_b.turn_count
                })
                
                if verbose:
                    print(f"Agent B: {response}")
                current_message = response
        
        # Collect trajectories
        agent_a_trajectory = np.array(self.agent_a.trajectory)
        agent_b_trajectory = np.array(self.agent_b.trajectory)
        
        # Calculate additional metrics
        conversation_duration = time.time() - self.conversation_start_time
        
        return {
            "conversation_history": conversation_history,
            "agent_a_trajectory": agent_a_trajectory,
            "agent_b_trajectory": agent_b_trajectory,
            "total_turns": len([msg for msg in conversation_history if msg["speaker"] in ["Agent_A", "Agent_B"]]),
            "encoding_type": self.encoding_type,
            "state_dimension": self.state_dimension,
            "conversation_duration": conversation_duration,
            "agent_a_encoding_info": self.agent_a.get_encoding_info(),
            "agent_b_encoding_info": self.agent_b.get_encoding_info(),
            "agent_a_response_times": self.agent_a.response_times,
            "agent_b_response_times": self.agent_b.response_times
        }