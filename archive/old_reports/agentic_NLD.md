# Two-Agent Discrete Time Dynamical System in Natural Language Text


A 2-agent text communication system can be modeled as a discrete-time dynamical system with the following mathematical structure:

## State Space Representation

**Agent States:**
- $\mathbf{s}_A(t) \in \mathbb{R}^d$ : Agent A's internal state (context embeddings, memory, goals)
- $\mathbf{s}_B(t) \in \mathbb{R}^d$ : Agent B's internal state

**Communication:**
- $\mathbf{T}_A(t) = [t_1, t_2, ..., t_{n_A}] \in V^{n_A}$ : Token sequence from Agent A
- $\mathbf{T}_B(t) = [t_1, t_2, ..., t_{n_B}] \in V^{n_B}$ : Token sequence from Agent B
- $V$ : vocabulary set

## Evolution Dynamics

**State Updates:**
$$\mathbf{s}_A(t+1) = f_A(\mathbf{s}_A(t), \phi_B(\mathbf{T}_B(t))) + \boldsymbol{\epsilon}_A(t)$$
$$\mathbf{s}_B(t+1) = f_B(\mathbf{s}_B(t), \phi_A(\mathbf{T}_A(t))) + \boldsymbol{\epsilon}_B(t)$$

Where:
- $\phi_A, \phi_B: V^* \rightarrow \mathbb{R}^d$ are text encoding functions
- $\boldsymbol{\epsilon}_A(t), \boldsymbol{\epsilon}_B(t)$ are noise terms

**Text Generation:**
$$\mathbf{T}_A(t+1) = g_A(\mathbf{s}_A(t+1)) + \boldsymbol{\delta}_A(t)$$
$$\mathbf{T}_B(t+1) = g_B(\mathbf{s}_B(t+1)) + \boldsymbol{\delta}_B(t)$$

## Chaos Conditions

**For sensitive dependence on initial conditions:**

1. **Nonlinearity**: $f_A, f_B, g_A, g_B$ must be nonlinear
2. **Deterministic dominance**: $||\boldsymbol{\epsilon}|| << ||f(\mathbf{s})||$
3. **Lyapunov criterion**: $\lambda_1 > 0$ where:

$$\lambda_1 = \lim_{T \rightarrow \infty} \frac{1}{T} \sum_{t=0}^{T-1} \ln \left|\frac{\partial F}{\partial \mathbf{s}}\right|_{\mathbf{s}(t)}$$

Where $F$ is the composed system evolution.

## Signal vs. Noise Decomposition

**Signal Components:**
- **Semantic coherence**: $S_{sem}(t) = \text{cosine\_sim}(\phi(\mathbf{T}(t)), \phi(\mathbf{T}(t-1)))$
- **Syntactic patterns**: $S_{syn}(t) = \text{entropy}(\text{POS\_sequence}(\mathbf{T}(t)))$
- **Deterministic trajectory**: $S_{det}(t) = ||\mathbf{s}(t) - \hat{\mathbf{s}}(t)||$ where $\hat{\mathbf{s}}$ is predicted state

**Noise Components:**
- **Lexical randomness**: $N_{lex}(t) = \text{perplexity}(\mathbf{T}(t))$
- **Processing errors**: $N_{proc}(t) = ||\boldsymbol{\epsilon}(t)||$
- **Semantic drift**: $N_{drift}(t) = \text{random\_component}(\phi(\mathbf{T}(t)))$

## Practical Measurement Strategy

**For chaos detection:**
1. **Embedding space**: Map token sequences to continuous vectors using $\phi$
2. **Phase space reconstruction**: Use delay embedding of conversation trajectory
3. **Divergence measurement**: 
   $$D(t) = ||\mathbf{s}_A^{(1)}(t) - \mathbf{s}_A^{(2)}(t)|| + ||\mathbf{s}_B^{(1)}(t) - \mathbf{s}_B^{(2)}(t)||$$
   
   Where superscripts (1), (2) denote slightly different initial conditions.

4. **Lyapunov exponent estimation**:
   $$\lambda = \lim_{t \rightarrow \infty} \frac{1}{t} \ln\left(\frac{D(t)}{D(0)}\right)$$

**Key Question**: What encoding function $\phi$ preserves the dynamical structure while mapping discrete tokens to continuous space?

Would you like to dive into specific choices for $\phi$ or discuss experimental design for measuring these quantities?