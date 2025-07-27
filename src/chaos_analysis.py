"""
Chaos analysis tools for the two-agent dynamical system.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


class ChaosAnalyzer:
    """
    Tools for analyzing chaotic behavior in the two-agent system.
    
    Implements:
    - Lyapunov exponent estimation
    - Phase space reconstruction
    - Divergence measurement
    - Signal vs noise decomposition
    """
    
    def __init__(self, state_dimension: int = 384):
        self.state_dimension = state_dimension
    
    def calculate_lyapunov_exponent(
        self, 
        trajectory1: np.ndarray, 
        trajectory2: np.ndarray,
        perturbation_size: float = 1e-6
    ) -> float:
        """
        Calculate the largest Lyapunov exponent using two trajectories with slightly
        different initial conditions.
        
        λ = lim(t→∞) (1/t) * ln(D(t)/D(0))
        
        Args:
            trajectory1: First trajectory (T, state_dim)
            trajectory2: Second trajectory with perturbed initial conditions
            perturbation_size: Size of initial perturbation
            
        Returns:
            Estimated Lyapunov exponent
        """
        if len(trajectory1) != len(trajectory2):
            raise ValueError("Trajectories must have same length")
        
        if len(trajectory1) < 10:
            return 0.0  # Not enough data
        
        # Calculate distances at each time step
        distances = np.linalg.norm(trajectory1 - trajectory2, axis=1)
        
        # Avoid log(0) by setting minimum distance
        distances = np.maximum(distances, 1e-12)
        
        # Initial distance (should be approximately perturbation_size)
        d0 = distances[0] if distances[0] > 0 else perturbation_size
        
        # Calculate log divergence
        log_divergence = np.log(distances / d0)
        
        # Estimate Lyapunov exponent as slope of log divergence
        time_steps = np.arange(len(log_divergence))
        
        # Use linear regression to find slope
        if len(time_steps) > 1:
            lyapunov = np.polyfit(time_steps, log_divergence, 1)[0]
        else:
            lyapunov = 0.0
        
        return lyapunov
    
    def phase_space_reconstruction(
        self, 
        time_series: np.ndarray, 
        delay: int = 1, 
        embedding_dim: int = 3
    ) -> np.ndarray:
        """
        Reconstruct phase space using delay embedding.
        
        Args:
            time_series: 1D time series data
            delay: Time delay for embedding
            embedding_dim: Embedding dimension
            
        Returns:
            Reconstructed phase space (N-delay*(embedding_dim-1), embedding_dim)
        """
        N = len(time_series)
        M = N - delay * (embedding_dim - 1)
        
        if M <= 0:
            raise ValueError("Time series too short for embedding parameters")
        
        embedded = np.zeros((M, embedding_dim))
        for i in range(embedding_dim):
            embedded[:, i] = time_series[i * delay:i * delay + M]
        
        return embedded
    
    def calculate_divergence_trajectory(
        self, 
        trajectory1: np.ndarray, 
        trajectory2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate divergence D(t) = ||s_A^(1)(t) - s_A^(2)(t)|| for each time step.
        
        Args:
            trajectory1: First trajectory
            trajectory2: Second trajectory
            
        Returns:
            Divergence at each time step
        """
        return np.linalg.norm(trajectory1 - trajectory2, axis=1)
    
    def analyze_conversation_chaos(
        self, 
        agent_a_trajectory: np.ndarray,
        agent_b_trajectory: np.ndarray,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Comprehensive chaos analysis of a conversation.
        
        Args:
            agent_a_trajectory: Agent A's state trajectory
            agent_b_trajectory: Agent B's state trajectory
            conversation_history: List of conversation messages
            
        Returns:
            Dictionary of chaos metrics
        """
        metrics = {}
        
        # Basic trajectory statistics
        if len(agent_a_trajectory) > 1:
            metrics["agent_a_state_variance"] = np.var(agent_a_trajectory, axis=0).mean()
            metrics["agent_b_state_variance"] = np.var(agent_b_trajectory, axis=0).mean()
            
            # State divergence over time
            state_distances = self.calculate_divergence_trajectory(
                agent_a_trajectory, agent_b_trajectory
            )
            metrics["mean_state_distance"] = np.mean(state_distances)
            metrics["max_state_distance"] = np.max(state_distances)
            metrics["state_distance_trend"] = self._calculate_trend(state_distances)
        
        # Conversation-level metrics
        if conversation_history:
            metrics.update(self._analyze_conversation_patterns(conversation_history))
        
        # Phase space analysis (use state magnitude as 1D signal)
        if len(agent_a_trajectory) > 10:
            agent_a_magnitude = np.linalg.norm(agent_a_trajectory, axis=1)
            agent_b_magnitude = np.linalg.norm(agent_b_trajectory, axis=1)
            
            try:
                # Attempt phase space reconstruction
                embedded_a = self.phase_space_reconstruction(agent_a_magnitude, delay=2, embedding_dim=3)
                embedded_b = self.phase_space_reconstruction(agent_b_magnitude, delay=2, embedding_dim=3)
                
                # Calculate correlation dimension (simplified)
                metrics["phase_space_complexity_a"] = self._estimate_correlation_dimension(embedded_a)
                metrics["phase_space_complexity_b"] = self._estimate_correlation_dimension(embedded_b)
                
            except ValueError:
                # Not enough data for phase space reconstruction
                metrics["phase_space_complexity_a"] = 0.0
                metrics["phase_space_complexity_b"] = 0.0
        
        return metrics
    
    def estimate_lyapunov_from_single_trajectory(
        self, 
        trajectory: np.ndarray, 
        window_size: int = 10
    ) -> float:
        """
        Estimate Lyapunov exponent from a single trajectory using local divergence.
        
        Args:
            trajectory: State trajectory (T, state_dim)
            window_size: Window size for local analysis
            
        Returns:
            Estimated Lyapunov exponent
        """
        if len(trajectory) < window_size * 2:
            return 0.0
        
        lyapunov_estimates = []
        
        for i in range(len(trajectory) - window_size):
            # Extract local segment
            segment = trajectory[i:i + window_size]
            
            # Calculate local divergence
            distances = np.linalg.norm(np.diff(segment, axis=0), axis=1)
            
            if len(distances) > 1 and np.all(distances > 0):
                # Log growth rate
                log_distances = np.log(distances + 1e-12)
                local_lyapunov = np.mean(np.diff(log_distances))
                lyapunov_estimates.append(local_lyapunov)
        
        return np.mean(lyapunov_estimates) if lyapunov_estimates else 0.0
    
    def _calculate_trend(self, series: np.ndarray) -> float:
        """Calculate linear trend in a time series"""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        return slope
    
    def _analyze_conversation_patterns(self, conversation_history: List[Dict[str, str]]) -> Dict[str, float]:
        """Analyze patterns in conversation for chaos indicators"""
        metrics = {}
        
        # Message length variability
        message_lengths = [len(msg["content"]) for msg in conversation_history]
        metrics["message_length_variance"] = np.var(message_lengths)
        
        # Response time patterns (if timestamps available)
        if all("timestamp" in msg for msg in conversation_history):
            response_times = []
            for i in range(1, len(conversation_history)):
                dt = conversation_history[i]["timestamp"] - conversation_history[i-1]["timestamp"]
                response_times.append(dt)
            
            if response_times:
                metrics["response_time_variance"] = np.var(response_times)
                metrics["response_time_trend"] = self._calculate_trend(np.array(response_times))
        
        # Semantic entropy (simplified)
        all_words = []
        for msg in conversation_history:
            all_words.extend(msg["content"].lower().split())
        
        if all_words:
            word_counts = {}
            for word in all_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Calculate entropy
            total_words = len(all_words)
            probabilities = [count / total_words for count in word_counts.values()]
            metrics["semantic_entropy"] = entropy(probabilities)
        
        return metrics
    
    def _estimate_correlation_dimension(self, embedded_data: np.ndarray, max_r: float = 1.0) -> float:
        """
        Estimate correlation dimension using the Grassberger-Procaccia algorithm.
        
        Args:
            embedded_data: Phase space embedded data
            max_r: Maximum radius for correlation integral
            
        Returns:
            Estimated correlation dimension
        """
        if len(embedded_data) < 10:
            return 0.0
        
        # Calculate pairwise distances
        distances = pdist(embedded_data)
        
        # Range of radii
        r_values = np.logspace(-3, np.log10(max_r), 20)
        correlations = []
        
        for r in r_values:
            # Count pairs within radius r
            count = np.sum(distances < r)
            total_pairs = len(distances)
            correlation = count / total_pairs if total_pairs > 0 else 0
            correlations.append(correlation + 1e-12)  # Avoid log(0)
        
        # Estimate dimension as slope of log(C(r)) vs log(r)
        log_r = np.log(r_values)
        log_c = np.log(correlations)
        
        # Find linear region (middle portion)
        start_idx = len(log_r) // 4
        end_idx = 3 * len(log_r) // 4
        
        if end_idx > start_idx:
            slope = np.polyfit(log_r[start_idx:end_idx], log_c[start_idx:end_idx], 1)[0]
            return max(0, slope)  # Dimension should be non-negative
        
        return 0.0


class SignalNoiseAnalyzer:
    """
    Implements signal vs noise decomposition for conversation analysis.
    """
    
    def __init__(self):
        pass
    
    def decompose_conversation(
        self, 
        conversation_history: List[Dict[str, str]],
        agent_trajectories: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Decompose conversation into signal and noise components.
        
        Returns:
            Dictionary with signal and noise metrics
        """
        results = {"signal": {}, "noise": {}}
        
        # Signal components
        results["signal"]["semantic_coherence"] = self._calculate_semantic_coherence(conversation_history)
        results["signal"]["syntactic_patterns"] = self._calculate_syntactic_entropy(conversation_history)
        results["signal"]["deterministic_trajectory"] = self._calculate_trajectory_predictability(agent_trajectories)
        
        # Noise components
        results["noise"]["lexical_randomness"] = self._calculate_lexical_randomness(conversation_history)
        results["noise"]["processing_errors"] = self._detect_processing_errors(conversation_history)
        results["noise"]["semantic_drift"] = self._calculate_semantic_drift(conversation_history)
        
        return results
    
    def _calculate_semantic_coherence(self, conversation_history: List[Dict[str, str]]) -> float:
        """Calculate semantic coherence using cosine similarity"""
        if len(conversation_history) < 2:
            return 1.0
        
        # Simple word vector approach
        coherence_scores = []
        for i in range(1, len(conversation_history)):
            msg1 = conversation_history[i-1]["content"].lower().split()
            msg2 = conversation_history[i]["content"].lower().split()
            
            # Create simple word vectors
            all_words = list(set(msg1 + msg2))
            if not all_words:
                continue
            
            vec1 = np.array([msg1.count(word) for word in all_words])
            vec2 = np.array([msg2.count(word) for word in all_words])
            
            # Cosine similarity
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_syntactic_entropy(self, conversation_history: List[Dict[str, str]]) -> float:
        """Calculate syntactic pattern entropy (simplified)"""
        # Count sentence structures (very simplified)
        patterns = []
        for msg in conversation_history:
            content = msg["content"]
            # Simple pattern: count questions, statements, etc.
            if "?" in content:
                patterns.append("question")
            elif "!" in content:
                patterns.append("exclamation")
            else:
                patterns.append("statement")
        
        if not patterns:
            return 0.0
        
        # Calculate pattern entropy
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        total = len(patterns)
        probabilities = [count / total for count in pattern_counts.values()]
        
        return entropy(probabilities)
    
    def _calculate_trajectory_predictability(self, agent_trajectories: Dict[str, np.ndarray]) -> float:
        """Calculate how predictable the state trajectories are"""
        predictability_scores = []
        
        for agent_name, trajectory in agent_trajectories.items():
            if len(trajectory) < 3:
                continue
            
            # Simple linear prediction
            errors = []
            for i in range(2, len(trajectory)):
                # Predict next state as linear extrapolation
                predicted = 2 * trajectory[i-1] - trajectory[i-2]
                actual = trajectory[i]
                error = np.linalg.norm(predicted - actual)
                errors.append(error)
            
            if errors:
                # Inverse of mean error (higher = more predictable)
                predictability = 1.0 / (1.0 + np.mean(errors))
                predictability_scores.append(predictability)
        
        return np.mean(predictability_scores) if predictability_scores else 0.0
    
    def _calculate_lexical_randomness(self, conversation_history: List[Dict[str, str]]) -> float:
        """Calculate lexical randomness (simplified perplexity measure)"""
        all_words = []
        for msg in conversation_history:
            all_words.extend(msg["content"].lower().split())
        
        if not all_words:
            return 0.0
        
        # Simple unigram model
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        total_words = len(all_words)
        unique_words = len(word_counts)
        
        # Higher ratio of unique to total words = higher randomness
        return unique_words / total_words if total_words > 0 else 0.0
    
    def _detect_processing_errors(self, conversation_history: List[Dict[str, str]]) -> float:
        """Detect processing errors/artifacts in messages"""
        error_indicators = ["[thinking]", "[processing]", "...", "*pause*", "[error]"]
        error_count = 0
        total_messages = len(conversation_history)
        
        for msg in conversation_history:
            content = msg["content"].lower()
            for indicator in error_indicators:
                if indicator in content:
                    error_count += 1
                    break
        
        return error_count / total_messages if total_messages > 0 else 0.0
    
    def _calculate_semantic_drift(self, conversation_history: List[Dict[str, str]]) -> float:
        """Calculate semantic drift over the conversation"""
        if len(conversation_history) < 3:
            return 0.0
        
        # Compare first and last portions of conversation
        first_portion = conversation_history[:len(conversation_history)//3]
        last_portion = conversation_history[-len(conversation_history)//3:]
        
        first_words = set()
        last_words = set()
        
        for msg in first_portion:
            first_words.update(msg["content"].lower().split())
        
        for msg in last_portion:
            last_words.update(msg["content"].lower().split())
        
        if not first_words or not last_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(first_words.intersection(last_words))
        total_unique = len(first_words.union(last_words))
        
        # Drift = 1 - overlap ratio
        return 1.0 - (overlap / total_unique) if total_unique > 0 else 1.0