# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a Python implementation of two-agent discrete time dynamical systems in natural language text. It demonstrates chaotic behavior in agent conversations with sensitive dependence on initial conditions.

## Project Structure

```
agentic_nld/
├── src/
│   ├── agent_system.py      # Two-agent conversation system
│   └── chaos_analysis.py    # Chaos theory analysis tools  
├── demo.py                  # Complete demo with analysis
├── agentic_NLD.md          # Mathematical framework
└── README.md               # Project documentation
```

## Development Commands

**Setup:**
```bash
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
```

**Run Demo:**
```bash
python demo.py  # Full demo with conversation and chaos analysis
```

**Run Tests:**
```bash
python -m pytest tests/  # (when tests are added)
```

## Key Implementation Features

- **Mathematical Framework**: Implements s_A(t+1) = f_A(s_A(t), φ_B(T_B(t))) + ε_A(t)
- **Chaos Analysis**: Lyapunov exponent calculation, sensitive dependence detection
- **Signal/Noise Decomposition**: Semantic coherence vs lexical randomness analysis
- **Text Encoding**: Functions φ for mapping discrete tokens to continuous vector space
- **Nonlinear Dynamics**: State evolution with memory integration and noise terms

## Core Classes

### SimpleTwoAgentSystem
Main system for running two-agent conversations:
```python
from src.agent_system import SimpleTwoAgentSystem

system = SimpleTwoAgentSystem(
    agent_a_prompt="You are a logical thinker.",
    agent_b_prompt="You are a creative thinker.", 
    max_turns=5
)
results = system.run_conversation("What is consciousness?")
```

### ChaosAnalyzer  
Tools for analyzing chaotic behavior:
```python
from src.chaos_analysis import ChaosAnalyzer

analyzer = ChaosAnalyzer()
lyapunov = analyzer.estimate_lyapunov_from_single_trajectory(trajectory)
```

### SignalNoiseAnalyzer
Decomposition of conversation components:
```python
from src.chaos_analysis import SignalNoiseAnalyzer

analyzer = SignalNoiseAnalyzer()
results = analyzer.decompose_conversation(conversation, trajectories)
```

## Mathematical Framework Implementation

The implementation follows the theoretical model from `agentic_NLD.md`:

- **State Vectors**: Agent states as vectors in ℝᵈ (configurable dimension)
- **Nonlinear Evolution**: State update functions with memory integration  
- **Text Encoding**: φ functions mapping tokens to continuous space
- **Noise Terms**: ε_A(t) and δ_A(t) for realistic dynamics
- **Chaos Detection**: Trajectory divergence analysis and Lyapunov estimation

## Key Results

The implementation successfully demonstrates:
- **Sensitive Dependence**: Small prompt changes → dramatically different conversations
- **State Divergence**: Agent vectors diverge significantly (0.6+ norm differences)  
- **Chaotic Signatures**: Low content similarity (15%) between perturbed runs
- **Signal/Noise Patterns**: 69% syntactic structure, 90% semantic drift

## Dependencies

Core requirements (no LangGraph needed):
- `langchain-openai`: LLM integration
- `numpy`: Numerical computations
- `scipy`: Scientific computing 
- `matplotlib`: Visualization
- `scikit-learn`: Analysis tools
- `python-dotenv`: Environment management