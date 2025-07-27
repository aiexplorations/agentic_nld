"""
Advanced text encoding schemes for more sophisticated agent state representation.
"""

import numpy as np
import re
from typing import List, Dict, Any
from collections import Counter
import hashlib

class AdvancedTextEncoder:
    """
    Advanced text encoding that captures semantic, syntactic, and statistical features.
    """
    
    def __init__(self, state_dimension: int = 64):
        self.state_dimension = state_dimension
        
        # Define feature dimensions
        self.semantic_dim = state_dimension // 4      # 16 dims for semantic features
        self.syntactic_dim = state_dimension // 4     # 16 dims for syntactic features  
        self.statistical_dim = state_dimension // 4   # 16 dims for statistical features
        self.lexical_dim = state_dimension - (self.semantic_dim + self.syntactic_dim + self.statistical_dim)  # remaining for lexical
        
        # Common words for TF-IDF-like weighting
        self.common_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        ])
        
        # Initialize encoding matrices for consistency
        np.random.seed(42)  # For reproducible embeddings
        self.word_embeddings = {}
        
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using multiple sophisticated features.
        
        Args:
            text: Input text string
            
        Returns:
            state_dimension-dimensional encoding vector
        """
        if not text or not text.strip():
            return np.zeros(self.state_dimension)
            
        # Clean and tokenize
        words = self._tokenize(text)
        
        # Extract different types of features
        semantic_features = self._extract_semantic_features(text, words)
        syntactic_features = self._extract_syntactic_features(text, words)
        statistical_features = self._extract_statistical_features(text, words)
        lexical_features = self._extract_lexical_features(words)
        
        # Combine all features
        encoding = np.concatenate([
            semantic_features,
            syntactic_features, 
            statistical_features,
            lexical_features
        ])
        
        # Normalize to prevent explosion
        encoding = np.tanh(encoding)
        
        return encoding
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return words
    
    def _extract_semantic_features(self, text: str, words: List[str]) -> np.ndarray:
        """
        Extract semantic features based on word co-occurrence and meaning.
        """
        features = np.zeros(self.semantic_dim)
        
        if not words:
            return features
            
        # Feature 1-4: Content word density
        content_words = [w for w in words if w not in self.common_words and len(w) > 3]
        features[0] = len(content_words) / max(len(words), 1)
        
        # Feature 5-8: Semantic categories (approximate using word patterns)
        abstract_indicators = ['think', 'believe', 'consider', 'understand', 'concept', 'idea', 'theory']
        concrete_indicators = ['see', 'touch', 'hear', 'physical', 'object', 'material', 'tangible']
        emotional_indicators = ['feel', 'emotion', 'happy', 'sad', 'angry', 'excited', 'calm', 'worried']
        action_indicators = ['do', 'make', 'create', 'build', 'move', 'action', 'perform', 'execute']
        
        features[1] = sum(1 for w in words if any(ind in w for ind in abstract_indicators)) / max(len(words), 1)
        features[2] = sum(1 for w in words if any(ind in w for ind in concrete_indicators)) / max(len(words), 1)
        features[3] = sum(1 for w in words if any(ind in w for ind in emotional_indicators)) / max(len(words), 1)
        features[4] = sum(1 for w in words if any(ind in w for ind in action_indicators)) / max(len(words), 1)
        
        # Feature 9-12: Topic coherence (word repetition and thematic consistency)
        word_counts = Counter(words)
        max_freq = max(word_counts.values()) if word_counts else 1
        features[5] = max_freq / max(len(words), 1)  # Repetition rate
        
        # Feature 13-16: Semantic complexity (unique word ratio, compound words)
        features[6] = len(set(words)) / max(len(words), 1)  # Lexical diversity
        features[7] = sum(1 for w in words if len(w) > 8) / max(len(words), 1)  # Complex words
        
        # Fill remaining semantic features with word embedding-like representations
        for i, word in enumerate(words[:8]):  # Use first 8 words
            if i < 8:
                word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
                features[8 + i] = (word_hash % 100) / 100.0
                
        return features
    
    def _extract_syntactic_features(self, text: str, words: List[str]) -> np.ndarray:
        """
        Extract syntactic features based on sentence structure.
        """
        features = np.zeros(self.syntactic_dim)
        
        if not text.strip():
            return features
            
        # Feature 1-4: Sentence structure
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        features[0] = len(sentences) / max(len(text.split()), 1)  # Sentence density
        if sentences:
            avg_words_per_sentence = sum(len(s.split()) for s in sentences) / len(sentences)
            features[1] = min(avg_words_per_sentence / 20.0, 1.0)  # Normalize to [0,1]
        
        # Feature 5-8: Punctuation patterns
        features[2] = text.count('?') / max(len(text), 1) * 100  # Question density
        features[3] = text.count('!') / max(len(text), 1) * 100  # Exclamation density
        features[4] = text.count(',') / max(len(text), 1) * 100  # Comma density
        features[5] = text.count(';') / max(len(text), 1) * 100  # Semicolon density
        
        # Feature 9-12: Word patterns
        if words:
            # Capitalization patterns
            cap_words = sum(1 for w in words if w != w.lower())
            features[6] = cap_words / len(words)
            
            # Average word length
            avg_word_len = sum(len(w) for w in words) / len(words)
            features[7] = min(avg_word_len / 10.0, 1.0)  # Normalize
        
        # Feature 13-16: Grammatical indicators (approximate)
        conjunctions = ['and', 'but', 'or', 'so', 'yet', 'for', 'nor', 'because', 'since', 'although']
        prepositions = ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about']
        
        features[8] = sum(1 for w in words if w in conjunctions) / max(len(words), 1)
        features[9] = sum(1 for w in words if w in prepositions) / max(len(words), 1)
        
        # Remaining features: positional word encoding
        for i, word in enumerate(words[:6]):
            if i < 6:
                pos_hash = int(hashlib.md5(f"{word}_{i}".encode()).hexdigest()[:8], 16)
                features[10 + i] = (pos_hash % 100) / 100.0
        
        return features
    
    def _extract_statistical_features(self, text: str, words: List[str]) -> np.ndarray:
        """
        Extract statistical features about text distribution.
        """
        features = np.zeros(self.statistical_dim)
        
        if not words:
            return features
            
        # Feature 1-4: Length statistics
        features[0] = len(text) / 1000.0  # Character count (normalized)
        features[1] = len(words) / 100.0  # Word count (normalized)
        
        if words:
            word_lengths = [len(w) for w in words]
            features[2] = np.mean(word_lengths) / 10.0  # Average word length
            features[3] = np.std(word_lengths) / 5.0   # Word length variance
        
        # Feature 5-8: Frequency distributions
        word_counts = Counter(words)
        if word_counts:
            max_freq = max(word_counts.values())
            min_freq = min(word_counts.values())
            features[4] = max_freq / len(words)  # Most frequent word ratio
            features[5] = len([w for w, c in word_counts.items() if c == 1]) / len(word_counts)  # Hapax ratio
        
        # Feature 9-12: Character-level statistics
        char_counts = Counter(text.lower())
        vowels = sum(char_counts.get(v, 0) for v in 'aeiou')
        consonants = sum(char_counts.get(c, 0) for c in 'bcdfghjklmnpqrstvwxyz')
        
        if vowels + consonants > 0:
            features[6] = vowels / (vowels + consonants)  # Vowel ratio
            features[7] = consonants / (vowels + consonants)  # Consonant ratio
        
        # Feature 13-16: Entropy-based measures
        if word_counts:
            total_words = sum(word_counts.values())
            word_probs = [count / total_words for count in word_counts.values()]
            entropy = -sum(p * np.log(p + 1e-10) for p in word_probs)
            features[8] = entropy / 10.0  # Normalized entropy
        
        # Remaining features: n-gram statistics
        if len(words) > 1:
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            features[9] = len(bigram_counts) / max(len(bigrams), 1)  # Bigram diversity
        
        # Fill remaining with text hash features
        text_hash = hashlib.md5(text.encode()).hexdigest()
        for i in range(6):
            if 10 + i < self.statistical_dim:
                hex_chunk = text_hash[i*2:(i+1)*2]
                features[10 + i] = int(hex_chunk, 16) / 255.0
                
        return features
    
    def _extract_lexical_features(self, words: List[str]) -> np.ndarray:
        """
        Extract lexical features using improved word embeddings.
        """
        features = np.zeros(self.lexical_dim)
        
        if not words:
            return features
            
        # Create word embeddings with more sophistication
        word_vectors = []
        for word in words[:self.lexical_dim]:  # Use up to lexical_dim words
            if word not in self.word_embeddings:
                # Create a more sophisticated embedding
                word_hash = hashlib.md5(word.encode()).hexdigest()
                
                # Multiple hash features for richer representation
                hash_features = []
                for i in range(0, min(len(word_hash), 32), 2):
                    hex_val = int(word_hash[i:i+2], 16)
                    hash_features.append(hex_val / 255.0)
                
                # Add length and character features
                len_feature = min(len(word) / 15.0, 1.0)
                first_char_feature = ord(word[0]) / 127.0 if word else 0
                last_char_feature = ord(word[-1]) / 127.0 if word else 0
                
                # Combine features
                embedding = hash_features[:13] + [len_feature, first_char_feature, last_char_feature]
                embedding = embedding[:16]  # Ensure fixed size
                
                while len(embedding) < 16:
                    embedding.append(0.0)
                    
                self.word_embeddings[word] = np.array(embedding)
            
            word_vectors.append(self.word_embeddings[word])
        
        if word_vectors:
            # Aggregate word vectors using multiple strategies
            word_matrix = np.array(word_vectors)
            
            # Mean pooling
            mean_vector = np.mean(word_matrix, axis=0)
            
            # Max pooling  
            max_vector = np.max(word_matrix, axis=0)
            
            # Combine mean and max
            combined = np.concatenate([mean_vector[:8], max_vector[:8]])
            
            # Ensure correct dimension
            if len(combined) > self.lexical_dim:
                features = combined[:self.lexical_dim]
            else:
                features[:len(combined)] = combined
        
        return features
    
    def get_encoding_info(self) -> Dict[str, Any]:
        """Return information about the encoding scheme."""
        return {
            "total_dimension": self.state_dimension,
            "semantic_dimension": self.semantic_dim,
            "syntactic_dimension": self.syntactic_dim,
            "statistical_dimension": self.statistical_dim,
            "lexical_dimension": self.lexical_dim,
            "encoding_type": "advanced_multi_feature",
            "features": {
                "semantic": [
                    "content_word_density", "abstract_indicators", "concrete_indicators",
                    "emotional_indicators", "action_indicators", "repetition_rate",
                    "lexical_diversity", "complex_words", "word_embeddings"
                ],
                "syntactic": [
                    "sentence_density", "avg_sentence_length", "question_density",
                    "exclamation_density", "comma_density", "semicolon_density",
                    "capitalization_ratio", "avg_word_length", "conjunctions",
                    "prepositions", "positional_encoding"
                ],
                "statistical": [
                    "character_count", "word_count", "avg_word_length", "word_length_std",
                    "max_frequency_ratio", "hapax_ratio", "vowel_ratio", "consonant_ratio",
                    "word_entropy", "bigram_diversity", "text_hash_features"
                ],
                "lexical": [
                    "sophisticated_word_embeddings", "mean_pooling", "max_pooling",
                    "length_features", "character_features"
                ]
            }
        }


class SemanticEncoder:
    """
    Simplified semantic encoder focusing on meaning preservation.
    """
    
    def __init__(self, state_dimension: int = 64):
        self.state_dimension = state_dimension
        
        # Semantic categories with associated words
        self.semantic_categories = {
            'cognitive': ['think', 'understand', 'know', 'learn', 'remember', 'forget', 'realize', 'comprehend'],
            'emotional': ['feel', 'emotion', 'happy', 'sad', 'angry', 'excited', 'calm', 'worried', 'love', 'hate'],
            'physical': ['see', 'touch', 'hear', 'taste', 'smell', 'move', 'walk', 'run', 'sit', 'stand'],
            'abstract': ['concept', 'idea', 'theory', 'principle', 'philosophy', 'meaning', 'purpose', 'essence'],
            'social': ['people', 'society', 'community', 'relationship', 'friend', 'family', 'team', 'group'],
            'temporal': ['time', 'past', 'present', 'future', 'now', 'then', 'before', 'after', 'when', 'while'],
            'causal': ['because', 'cause', 'effect', 'result', 'consequence', 'reason', 'due', 'since'],
            'modal': ['possible', 'impossible', 'necessary', 'optional', 'must', 'should', 'could', 'might']
        }
        
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text focusing on semantic meaning."""
        if not text or not text.strip():
            return np.zeros(self.state_dimension)
            
        words = text.lower().split()
        encoding = np.zeros(self.state_dimension)
        
        # Allocate dimensions to different semantic aspects
        cat_dim = self.state_dimension // len(self.semantic_categories)
        
        for i, (category, category_words) in enumerate(self.semantic_categories.items()):
            start_idx = i * cat_dim
            end_idx = min(start_idx + cat_dim, self.state_dimension)
            
            # Count matches in this category
            matches = sum(1 for word in words if any(cw in word for cw in category_words))
            activation = matches / max(len(words), 1)
            
            # Distribute activation across allocated dimensions
            for j in range(start_idx, end_idx):
                if j < self.state_dimension:
                    word_idx = (j - start_idx) % max(len(words), 1)
                    word_hash = hash(words[word_idx] if words else "empty") % 1000
                    encoding[j] = activation * (word_hash / 1000.0)
        
        return np.tanh(encoding)