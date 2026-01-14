"""
LlamaHook: Signal Extraction for Neural Pulse

Hooks into Llama-3-8B generation to extract temporal signals:
1. Entropy H(t): Shannon entropy of softmax distribution at each token
2. Attention Dispersion A(t) v2: Max normalized attention to context (sink-removed)

Key improvements in Attention v2:
- Removes attention sink (first token) to avoid bias
- Tracks MAX attention to ANY context token (not average)
- Normalizes by total attention budget
- Result: Robust metric in [0, 1] independent of generation length

Usage:
    hook = LlamaSignalHook(model_name="meta-llama/Llama-3.1-8B-Instruct")
    trace = hook.generate_with_signals(prompt="What is 2+2?", max_new_tokens=100)
    # trace contains: generated_text, entropy_trace, attention_trace
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationTrace:
    """Complete generation trace with signals"""
    prompt: str
    generated_text: str
    generated_tokens: List[str]
    entropy_trace: List[float]  # H(t) - Token probability entropy
    attention_trace: List[float]  # A(t) - Attention to context mass (DEPRECATED - noisy)
    perplexity_trace: List[float]  # P(t) - Perplexity (exp(entropy)) (DEPRECATED - wrong sign)
    attention_entropy_trace: List[float]  # H_attn(t) - Attention distribution entropy (DEPRECATED - weak)
    semantic_drift_trace: List[float]  # D(t) - Cosine similarity to prompt embedding (PHASE 2a - PRIMARY)
    logits_trace: Optional[List[torch.Tensor]] = None  # Raw logits (optional)
    attention_weights: Optional[List[torch.Tensor]] = None  # Raw attention (optional)


class LlamaSignalHook:
    """
    Signal extraction hook for Llama-3-8B.

    Intercepts generation process to compute temporal signals:
    - Entropy H(t): Measures uncertainty in token prediction
    - Attention Dispersion A(t) v2: Measures context engagement

    Hypothesis:
    - SECA attacks cause high entropy (waffling) + low attention to context (detachment)
    - Clean prompts cause low entropy (confidence) + high attention to context (grounding)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = None
    ):
        """
        Initialize LlamaHook.

        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None (auto-detect)
            dtype: torch.float16 (CUDA) or torch.float32 (CPU)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype or (torch.float16 if self.device == 'cuda' else torch.float32)

        logger.info(f"Loading model: {model_name} on {self.device} ({self.dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map='auto' if self.device == 'cuda' else None,
            output_attentions=True  # CRITICAL: Enable attention output
        )

        if self.device == 'cpu':
            self.model.to(self.device)

        self.model.eval()

        logger.info("LlamaHook initialized successfully")

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute Shannon entropy of softmax distribution.

        H(t) = -Σ p_i * log(p_i) where p_i = softmax(logits)

        High entropy → High uncertainty (waffling)
        Low entropy → Low uncertainty (confident)

        Args:
            logits: Logits for next token prediction [vocab_size]

        Returns:
            Entropy value [0, log(vocab_size)]
        """
        # Convert to float32 to avoid precision issues with float16
        logits_f32 = logits.float()

        # Compute probabilities
        probs = torch.softmax(logits_f32, dim=-1)

        # Compute entropy using stable formula
        # Use epsilon appropriate for float32
        epsilon = 1e-9

        # Clamp probabilities to avoid log(0)
        probs_clamped = torch.clamp(probs, min=epsilon)

        # Compute log probabilities
        log_probs = torch.log(probs_clamped)

        # Compute entropy: -sum(p * log(p))
        # Zero out contributions where prob is effectively zero to avoid 0*-inf = nan
        entropy_terms = torch.where(
            probs > epsilon,
            probs * log_probs,
            torch.zeros_like(probs)
        )

        entropy = -torch.sum(entropy_terms).item()

        return entropy

    def compute_attention_dispersion_v2(
        self,
        attention_weights: torch.Tensor,
        context_length: int,
        generated_length: int
    ) -> float:
        """
        Compute Attention to Context (v3 - FIXED).

        This measures how much the model attends to the original context (prompt)
        when generating new tokens. Attention weights are already normalized
        (sum to 1.0 via softmax), so we just need to sum attention mass over context.

        Formula:
            A(t) = sum(attention[last_token, context_tokens])

        Where context_tokens excludes the attention sink (first token).

        Args:
            attention_weights: Attention tensor tuple (one per layer)
                              Each: [batch, n_heads, seq_len, seq_len]
            context_length: Number of context (prompt) tokens
            generated_length: Number of generated tokens so far

        Returns:
            Attention to context value [0, 1]
            - 1.0 = All attention on context (high engagement)
            - 0.0 = No attention on context (detachment)
        """
        if attention_weights is None or len(attention_weights) == 0:
            logger.warning("No attention weights available")
            return 0.0

        # Get last layer attention (most relevant for final predictions)
        # Shape: [batch, n_heads, seq_len, seq_len]
        last_layer_attn = attention_weights[-1]

        # Average across heads: [batch, seq_len, seq_len]
        # Shape: [seq_len, seq_len] after squeezing batch dim
        attn_avg_heads = last_layer_attn.mean(dim=1).squeeze(0)

        # Extract attention from last generated token
        # Shape: [seq_len]
        last_token_attn = attn_avg_heads[-1, :]

        # Attention weights should sum to 1.0 (verify)
        # total_mass = last_token_attn.sum().item()
        # assert 0.99 <= total_mass <= 1.01, f"Attention doesn't sum to 1: {total_mass}"

        # Remove attention sink (first token at index 0)
        # Context tokens: [1, context_length)
        if context_length <= 1:
            return 0.0  # No context besides sink

        # Sum attention mass over context tokens (excluding sink)
        context_attn_mass = last_token_attn[1:context_length].sum().item()

        # This is already in [0, 1] since attention sums to 1
        # High value = model attends to context
        # Low value = model ignores context (attends to generated tokens)

        return context_attn_mass

    def compute_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        context_length: int
    ) -> float:
        """
        Compute entropy of attention distribution (v3 - NEW).

        Measures how "scattered" or "focused" the attention is.
        - High entropy = attention spread across many tokens (confused/searching)
        - Low entropy = attention focused on few tokens (confident)

        Formula:
            H_attn(t) = -sum(p_i * log(p_i))

        where p_i is attention weight to token i.

        Args:
            attention_weights: Attention tensor tuple
            context_length: Number of context tokens

        Returns:
            Attention entropy value [0, log(seq_len)]
        """
        if attention_weights is None or len(attention_weights) == 0:
            return 0.0

        # Get last layer, last token attention
        last_layer_attn = attention_weights[-1]
        attn_avg_heads = last_layer_attn.mean(dim=1).squeeze(0)
        last_token_attn = attn_avg_heads[-1, :]

        # Convert to float32 for numerical stability (same as entropy fix)
        probs = last_token_attn.float()

        # Compute entropy
        epsilon = 1e-9
        probs_clamped = torch.clamp(probs, min=epsilon)
        log_probs = torch.log(probs_clamped)

        entropy_terms = torch.where(
            probs > epsilon,
            probs * log_probs,
            torch.zeros_like(probs)
        )

        attention_entropy = -torch.sum(entropy_terms).item()

        return attention_entropy

    def compute_perplexity(self, entropy: float) -> float:
        """
        Compute perplexity from entropy.

        Perplexity = 2^entropy (or e^entropy depending on log base)

        Since we use natural log, perplexity = e^entropy.
        This amplifies differences - useful for visualization.

        Args:
            entropy: Entropy value

        Returns:
            Perplexity value
        """
        import math
        return math.exp(entropy)

    def compute_semantic_drift(
        self,
        prompt_embedding: torch.Tensor,
        current_hidden_state: torch.Tensor
    ) -> float:
        """
        Compute semantic drift as cosine similarity to prompt embedding.

        PHASE 2a PRIMARY SIGNAL - Semantic Drift Trajectory

        Theory: Hallucinations progressively drift away from the semantic
        anchor of the prompt. Normal generations stay grounded.

        Args:
            prompt_embedding: Average-pooled hidden state of prompt tokens [hidden_dim]
            current_hidden_state: Hidden state of current generation step [hidden_dim]

        Returns:
            Cosine similarity in [0, 1] (1 = perfectly aligned, 0 = orthogonal)
        """
        # Ensure both are 1D vectors
        if prompt_embedding.dim() > 1:
            prompt_embedding = prompt_embedding.squeeze()
        if current_hidden_state.dim() > 1:
            current_hidden_state = current_hidden_state.squeeze()

        # Compute cosine similarity
        # cos(θ) = (A · B) / (||A|| ||B||)
        dot_product = torch.dot(prompt_embedding, current_hidden_state)
        norm_prompt = torch.norm(prompt_embedding)
        norm_current = torch.norm(current_hidden_state)

        # Avoid division by zero
        if norm_prompt < 1e-8 or norm_current < 1e-8:
            return 0.0

        cosine_sim = dot_product / (norm_prompt * norm_current)

        # Clamp to [0, 1] range (cosine can be [-1, 1], but we expect positive)
        # If negative similarity, it means complete semantic reversal (very bad!)
        cosine_sim = torch.clamp(cosine_sim, min=0.0, max=1.0)

        return cosine_sim.item()

    def generate_with_signals(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        return_raw_data: bool = False
    ) -> GenerationTrace:
        """
        Generate text and extract signals at each step.

        This is the main interface for extracting temporal signals during generation.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = default)
            do_sample: Use sampling (True) or greedy (False)
            return_raw_data: Include raw logits/attention in trace

        Returns:
            GenerationTrace with signals and generated text
        """
        logger.info(f"Generating with signals (max_tokens={max_new_tokens})")

        # Format prompt with chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        context_length = inputs['input_ids'].size(1)

        # Storage for signals
        entropy_trace = []
        attention_trace = []
        perplexity_trace = []
        attention_entropy_trace = []
        semantic_drift_trace = []
        logits_trace = [] if return_raw_data else None
        attention_weights_trace = [] if return_raw_data else None
        generated_tokens = []

        # Generation loop
        with torch.no_grad():
            current_input_ids = inputs['input_ids']

            # PHASE 2a: Extract prompt embedding (once, before generation loop)
            # Get hidden states for prompt tokens
            prompt_outputs = self.model(
                input_ids=inputs['input_ids'],
                output_hidden_states=True,
                output_attentions=False  # Don't need attention for prompt
            )
            # Use last layer hidden states, average-pool over prompt tokens
            prompt_hidden_states = prompt_outputs.hidden_states[-1]  # [batch, seq, hidden]
            prompt_embedding = prompt_hidden_states[0, :, :].mean(dim=0)  # [hidden_dim]

            for step in range(max_new_tokens):
                # Forward pass
                outputs = self.model(
                    input_ids=current_input_ids,
                    output_attentions=True,
                    output_hidden_states=True  # PHASE 2a: Need hidden states for drift
                )

                logits = outputs.logits[0, -1, :]  # Last token logits
                attention_weights = outputs.attentions  # Tuple of attention tensors
                hidden_states = outputs.hidden_states  # Tuple of hidden state tensors

                # Compute all signals
                H_t = self.compute_entropy(logits)
                A_t = self.compute_attention_dispersion_v2(
                    attention_weights,
                    context_length=context_length,
                    generated_length=step + 1
                )
                P_t = self.compute_perplexity(H_t)
                H_attn_t = self.compute_attention_entropy(
                    attention_weights,
                    context_length=context_length
                )

                # PHASE 2a: Compute semantic drift
                current_hidden_state = hidden_states[-1][0, -1, :]  # Last layer, last token
                D_t = self.compute_semantic_drift(prompt_embedding, current_hidden_state)

                entropy_trace.append(H_t)
                attention_trace.append(A_t)
                perplexity_trace.append(P_t)
                attention_entropy_trace.append(H_attn_t)
                semantic_drift_trace.append(D_t)

                if return_raw_data:
                    logits_trace.append(logits.cpu())
                    attention_weights_trace.append([a.cpu() for a in attention_weights])

                # Sample next token
                if do_sample:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated_tokens.append(self.tokenizer.decode(next_token.item()))

                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Append to input for next step
                current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0)], dim=1)

        # Decode generated text
        generated_text = self.tokenizer.decode(
            current_input_ids[0, context_length:],
            skip_special_tokens=True
        )

        return GenerationTrace(
            prompt=prompt,
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            entropy_trace=entropy_trace,
            attention_trace=attention_trace,
            perplexity_trace=perplexity_trace,
            attention_entropy_trace=attention_entropy_trace,
            semantic_drift_trace=semantic_drift_trace,
            logits_trace=logits_trace,
            attention_weights=attention_weights_trace
        )

    def extract_traces_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> List[GenerationTrace]:
        """
        Extract signals for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Max tokens per generation
            temperature: Sampling temperature

        Returns:
            List of GenerationTrace objects
        """
        traces = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            trace = self.generate_with_signals(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            traces.append(trace)
        return traces

    def save_traces(self, traces: List[GenerationTrace], output_path: str):
        """
        Save traces to JSON file.

        Args:
            traces: List of GenerationTrace objects
            output_path: Path to save JSON
        """
        import json

        data = {
            'traces': [
                {
                    'prompt': t.prompt,
                    'generated_text': t.generated_text,
                    'generated_tokens': t.generated_tokens,
                    'entropy_trace': t.entropy_trace,
                    'attention_trace': t.attention_trace,
                    'perplexity_trace': t.perplexity_trace,
                    'attention_entropy_trace': t.attention_entropy_trace,
                    'signals': {
                        'avg_entropy': np.mean(t.entropy_trace) if t.entropy_trace else 0.0,
                        'max_entropy': np.max(t.entropy_trace) if t.entropy_trace else 0.0,
                        'avg_attention': np.mean(t.attention_trace) if t.attention_trace else 0.0,
                        'min_attention': np.min(t.attention_trace) if t.attention_trace else 0.0,
                        'avg_perplexity': np.mean(t.perplexity_trace) if t.perplexity_trace else 0.0,
                        'avg_attention_entropy': np.mean(t.attention_entropy_trace) if t.attention_entropy_trace else 0.0
                    }
                }
                for t in traces
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(traces)} traces to {output_path}")


def main():
    """CLI interface for signal extraction"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='LlamaHook Signal Extraction')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Model name')
    parser.add_argument('--prompts', type=str, required=True,
                       help='Path to prompts JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output traces JSON')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Max new tokens')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Load prompts
    with open(args.prompts, 'r') as f:
        prompt_data = json.load(f)

    if isinstance(prompt_data, list):
        if isinstance(prompt_data[0], dict):
            prompts = [item.get('prompt', item.get('adversarial_prompt', '')) for item in prompt_data]
        else:
            prompts = prompt_data
    else:
        prompts = [prompt_data]

    # Initialize hook
    hook = LlamaSignalHook(model_name=args.model, device=args.device)

    # Extract traces
    traces = hook.extract_traces_batch(
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )

    # Save
    hook.save_traces(traces, args.output)

    logger.info("Signal extraction complete!")


if __name__ == '__main__':
    main()
