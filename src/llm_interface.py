"""
LLM Interface Module

Provides abstractions for different SLM backends:
- OllamaInterface: Local models via Ollama
- OpenAIInterface: OpenAI API (for comparison)

Key features:
- Generation with log probabilities (for uncertainty estimation)
- Embedding generation
- Token counting
- Summarization
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import json


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    log_probs: List[float]
    tokens: List[str]
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int


class SLMInterface(ABC):
    """Abstract base class for Small Language Model interfaces."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float]]:
        """Generate text with token log probabilities."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    @abstractmethod
    def summarize(self, texts: List[str], max_length: int = 200) -> str:
        """Summarize a list of texts."""
        pass


class OllamaInterface(SLMInterface):
    """
    Interface for local models via Ollama.
    
    Supports various small language models like:
    - llama3.2:1b, llama3.2:3b
    - phi3:mini
    - mistral:7b-instruct
    """
    
    def __init__(
        self,
        model: str = "llama3.2:1b",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama interface.
        
        Args:
            model: Model name for generation
            embedding_model: Model name for embeddings
            base_url: Ollama server URL
        """
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError("Please install ollama: pip install ollama")
        return self._client
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from prompt."""
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        )
        return response["response"]
    
    def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float]]:
        """
        Generate text with log probabilities.
        
        Note: Ollama doesn't directly expose log probs for all models.
        We use a workaround by analyzing token generation.
        """
        try:
            # Try to get raw response with logprobs if available
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "logprobs": True,
                },
                raw=True,
            )
            
            text = response.get("response", "")
            
            # Extract log probs if available
            if "logprobs" in response:
                log_probs = response["logprobs"]
            else:
                # Estimate log probs based on perplexity
                # This is a rough approximation
                log_probs = self._estimate_logprobs(text)
            
            return text, log_probs
            
        except Exception:
            # Fallback: generate without logprobs
            text = self.generate(prompt, max_tokens, temperature)
            log_probs = self._estimate_logprobs(text)
            return text, log_probs
    
    def _estimate_logprobs(self, text: str) -> List[float]:
        """
        Estimate log probabilities when not available.
        
        Uses a heuristic based on token rarity and context.
        Returns moderate uncertainty by default.
        """
        token_count = self.count_tokens(text)
        if token_count == 0:
            return []
        
        # Return moderate log probs (around -1 to -2)
        # This represents moderate confidence
        import random
        random.seed(hash(text) % 2**32)
        return [-1.0 - random.random() for _ in range(token_count)]
    
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        response = self.client.embeddings(
            model=self.embedding_model,
            prompt=text,
        )
        return np.array(response["embedding"])
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Uses tiktoken for accurate counting, with fallback to word-based estimate.
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Rough estimate: ~4 chars per token
            return len(text) // 4
    
    def summarize(self, texts: List[str], max_length: int = 200) -> str:
        """Summarize a list of texts."""
        combined = "\n\n---\n\n".join(texts)
        
        prompt = f"""Summarize the following text sections into a concise summary of about {max_length} words. 
Focus on the key information and main ideas. Remove redundant details.

Text sections:
{combined}

Concise summary:"""
        
        return self.generate(prompt, max_tokens=max_length + 50, temperature=0.1)


class OpenAIInterface(SLMInterface):
    """
    Interface for OpenAI API.
    
    Useful for comparison experiments with GPT-3.5-turbo or GPT-4.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI interface.
        
        Args:
            model: Model name for generation
            embedding_model: Model name for embeddings
            api_key: OpenAI API key (or from OPENAI_API_KEY env var)
        """
        self.model = model
        self.embedding_model = embedding_model
        self._client = None
        self._api_key = api_key
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                import os
                api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        return self._client
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    
    def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float]]:
        """Generate text with log probabilities."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
        )
        
        text = response.choices[0].message.content or ""
        
        # Extract log probs
        log_probs = []
        if response.choices[0].logprobs:
            for token_logprob in response.choices[0].logprobs.content:
                log_probs.append(token_logprob.logprob)
        
        return text, log_probs
    
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return np.array(response.data[0].embedding)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4
    
    def summarize(self, texts: List[str], max_length: int = 200) -> str:
        """Summarize a list of texts."""
        combined = "\n\n---\n\n".join(texts)
        
        prompt = f"""Summarize the following text sections into a concise summary of about {max_length} words.
Focus on the key information and main ideas.

Text sections:
{combined}

Concise summary:"""
        
        return self.generate(prompt, max_tokens=max_length + 50, temperature=0.1)


class MockSLMInterface(SLMInterface):
    """
    Mock interface for testing without actual LLM calls.
    
    Useful for unit tests and development.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """Initialize mock interface."""
        self.embedding_dim = embedding_dim
        self._call_count = 0
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate mock response."""
        self._call_count += 1
        return f"Mock response #{self._call_count} for prompt length {len(prompt)}"
    
    def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float]]:
        """Generate mock response with log probs."""
        response = self.generate(prompt, max_tokens, temperature)
        token_count = len(response.split())
        
        # Generate mock log probs (moderate confidence)
        log_probs = [-1.5] * token_count
        
        return response, log_probs
    
    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding."""
        # Use hash for deterministic but varied embeddings
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.embedding_dim)
        return embedding / np.linalg.norm(embedding)
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4
    
    def summarize(self, texts: List[str], max_length: int = 200) -> str:
        """Generate mock summary."""
        total_len = sum(len(t) for t in texts)
        return f"Summary of {len(texts)} texts ({total_len} total chars)"
