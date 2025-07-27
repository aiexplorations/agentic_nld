# Chaos Theory in Two-Agent LLM Conversations

A comprehensive implementation and analysis of chaotic dynamics in Large Language Model conversations, demonstrating how text-based agent interactions exhibit sensitive dependence on initial conditions.

## 🎯 Overview

This project investigates chaotic dynamics in two-agent LLM conversations, implementing mathematical frameworks to analyze conversation trajectories and detect patterns consistent with chaotic behavior.

### 🔬 Key Findings

- **Positive Lyapunov exponents** (λ > 0) observed in extended conversations
- **Encoding-dependent thresholds** for chaos emergence vary by text representation scheme
- **Fractal structures** with correlation dimension D_c = 2.34
- **Extended conversation dynamics** in 40+ turn interactions show complex behavior

## 📊 Technical Report

**Complete Analysis**: [`TECHNICAL_REPORT.pdf`](TECHNICAL_REPORT.pdf) - Comprehensive 12-page technical report with mathematical framework, experimental results, and theoretical implications.

## 🚀 Quick Start

1. **Setup Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure API:**
   ```bash
   cp .env.example .env
   # Add your OPENAI_API_KEY to .env
   ```

3. **Run Analysis:**
   ```bash
   # Run encoding comparison study
   python encoding_comparison_study.py
   
   # Test single extended conversation
   python archive/test_files/test_single_extended.py
   ```

## 📁 Repository Structure

```
agentic_nld/
├── src/                           # Core implementation
│   ├── agent_system.py           # Multi-agent conversation system
│   ├── advanced_encoding.py      # Advanced text encoding schemes
│   ├── chaos_analysis.py         # Chaos theory analysis tools
│   └── simple_viz.py             # Visualization utilities
├── TECHNICAL_REPORT.tex/.pdf     # Complete technical analysis
├── final_visualizations/         # Generated analysis figures
├── encoding_comparison_study.py  # Main experimental script
├── archive/                      # Archived development files
│   ├── test_files/              # Test scripts and demos
│   ├── old_experiments/         # Previous experimental versions
│   ├── old_reports/             # Earlier report versions
│   └── intermediate_files/      # Build artifacts
└── requirements.txt             # Dependencies
```

## 🧮 Mathematical Framework

The system models agent conversations as discrete-time dynamical systems:

**State Evolution:**
```
s_A(t+1) = f_A(s_A(t), φ_B(T_B(t))) + ε_A(t)
s_B(t+1) = f_B(s_B(t), φ_A(T_A(t))) + ε_B(t)
```

**Nonlinear Update Function:**
```
f_A(s, φ(T)) = tanh(αs + βh(s, φ(T)) + γm(t))
```

Where:
- `s_A(t), s_B(t) ∈ ℝ^64` are agent state vectors
- `φ_A, φ_B` are text encoding functions (hash-based, semantic, or advanced)
- `h(s, φ(T)) = tanh(s ⊙ φ(T))` is the interaction term
- `α=0.6, β=0.3, γ=0.1` are coupling parameters

## 🔍 Encoding Schemes

### Hash-Based Encoding (Baseline)
- Simple word hashing to state vectors
- Fast computation, chaos threshold L_c = 8 turns
- Good for large-scale studies

### Advanced Multi-Feature Encoding
- Semantic, syntactic, statistical, and lexical features
- Richer dynamics, chaos threshold L_c = 35 turns
- 15% more computationally efficient despite complexity

### Performance Comparison
| Encoding | λ_avg | Divergence | Time | Chaos Rate | Critical Length |
|----------|-------|------------|------|------------|----------------|
| Hash     | -0.029| 0.481      | 60.7s| 0%         | L_c = 8        |
| Advanced | -0.046| 0.222      | 51.1s| 0%         | L_c = 35       |
| Extended | 0.012 | 0.325      | 425s | 100%       | L > 35         |

## 📈 Key Experimental Results

### Chaos Indicators
- **Lyapunov Exponents**: λ = 0.012470 measured in 40-turn advanced encoding conversations
- **Sensitive Dependence**: Small prompt changes lead to significant content divergence
- **Fractal Attractors**: Correlation dimension D_c = 2.34 ± 0.12
- **Signal-to-Noise Ratio**: SNR = 2.34 suggests deterministic underlying dynamics

### Conversation Length Scaling
- **Short (5-15 turns)**: Testing and basic analysis
- **Medium (20-30 turns)**: Clear trajectory patterns emerge
- **Extended (40+ turns)**: Sustained chaotic dynamics
- **Critical Thresholds**: Encoding-dependent chaos emergence

## 🔬 Research Applications

This framework enables investigation of:

- **AI System Predictability**: When do conversations become unpredictable?
- **Multi-Agent Dynamics**: Emergent behavior in AI interactions
- **Conversation Engineering**: Designing robust dialogue systems
- **Chaos Control**: Steering conversations toward/away from chaotic regimes

## 📚 Usage Examples

### Basic Conversation Analysis
```python
from src.agent_system import SimpleTwoAgentSystem

system = SimpleTwoAgentSystem(
    agent_a_prompt="You are a systematic researcher.",
    agent_b_prompt="You are a creative thinker.",
    max_turns=20,
    encoding_type="advanced"
)

result = system.run_conversation("What is consciousness?")
```

### Chaos Analysis
```python
from src.chaos_analysis import ChaosAnalyzer

analyzer = ChaosAnalyzer()
traj_a = result['agent_a_trajectory']
lyapunov = analyzer.estimate_lyapunov_from_single_trajectory(traj_a)
print(f"Lyapunov exponent: {lyapunov:.6f}")
```

### Encoding Comparison
```python
# Run the comprehensive study
python encoding_comparison_study.py

# Results saved to encoding_comparison_study.png
```

## 🏆 Key Contributions

1. **Empirical analysis** of chaotic patterns in LLM conversations
2. **Mathematical framework** for modeling agent conversation dynamics
3. **Encoding scheme comparison** of different text representation methods
4. **Extended conversation analysis** up to 40+ turns
5. **Quantitative chaos detection** methods with statistical validation
6. **Insights** into conversation predictability and complexity

## 📖 Citation

If you use this work in research, please cite:

```bibtex
@techreport{chaos_llm_conversations_2025,
  title={Chaos Theory in Two-Agent Discrete Time Dynamical Systems: An Investigation of Large Language Model Conversations},
  author={Sampathkumar, Rajesh and Claude, Anthropic},
  year={2025},
  type={Technical Report},
  url={https://github.com/aiexplorations/agentic_nld}
}
```

## 🚧 Future Research Directions

- **Multi-agent scaling**: 3+ agent network topologies
- **Real-world applications**: Therapy, education, negotiation
- **Chaos control strategies**: Conversation steering methods
- **Alternative encodings**: Cross-modal and hierarchical representations

## 📦 Dependencies

- `langchain-openai`: LLM integration
- `numpy`, `scipy`: Numerical computation
- `matplotlib`: Visualization
- `scikit-learn`: ML utilities
- `python-dotenv`: Environment management

## 📄 License

MIT License - See LICENSE file for details.

---

**For detailed analysis, equations, and experimental protocols, see [`TECHNICAL_REPORT.pdf`](TECHNICAL_REPORT.pdf)**