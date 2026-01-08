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
    entropy_trace: List[float]  # H(t) for each token t
    attention_trace: List[float]  # A(t) for each token t
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
        probs = torch.softmax(logits, dim=-1)
        # Add epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs).item()
        return entropy

    def compute_attention_dispersion_v2(
        self,
        attention_weights: torch.Tensor,
        context_length: int,
        generated_length: int
    ) -> float:
        """
        Compute Attention Dispersion v2 (sink-removed, normalized).

        Improvements over v1:
        1. Removes first token (attention sink) to avoid bias
        2. Tracks MAX attention to ANY context token (not average)
        3. Normalizes by total attention budget
        4. Result in [0, 1] independent of sequence length

        Formula:
            A(t) = max_i(attention_to_context_token_i) / mean(attention_to_all_tokens)

        Where:
        - i ranges over context tokens (excluding sink)
        - Denominator normalizes by total attention budget

        Args:
            attention_weights: Attention tensor [n_layers, n_heads, seq_len, seq_len]
            context_length: Number of context (prompt) tokens
            generated_length: Number of generated tokens so far

        Returns:
            Attention dispersion value [0, 1]
        """
        # Extract attention from last token (currently being generated)
        # Shape: [n_layers, n_heads, seq_len, seq_len]
        # We want: last_token_attention = attention[:, :, -1, :]

        if attention_weights is None or len(attention_weights) == 0:
            logger.warning("No attention weights available")
            return 0.0

        # Stack all layers
        # attention_weights is tuple of tensors, one per layer
        stacked_attention = torch.stack(attention_weights, dim=0)  # [n_layers, batch, n_heads, seq_len, seq_len]

        # Get last token's attention (what the new token attends to)
        # Shape: [n_layers, batch, n_heads, seq_len]
        last_token_attn = stacked_attention[:, 0, :, -1, :]  # Assume batch_size=1

        # Average across layers and heads
        # Shape: [seq_len]
        avg_attn = last_token_attn.mean(dim=0).mean(dim=0)

        # Remove attention sink (first token)
        # Shape: [seq_len - 1]
        if avg_attn.size(0) > 1:
            avg_attn_no_sink = avg_attn[1:]  # Remove index 0
        else:
            return 0.0  # Edge case: only one token

        # Split into context and generated regions
        # Context: tokens [1, context_length) (excluding sink at 0)
        # Generated: tokens [context_length, seq_len)

        if context_length <= 1:
            return 0.0  # No context to attend to

        context_attn = avg_attn_no_sink[:context_length-1]  # Exclude sink
        total_attn = avg_attn_no_sink.mean().item()

        if total_attn == 0:
            return 0.0

        # Compute MAX attention to ANY context token
        max_context_attn = context_attn.max().item() if context_attn.numel() > 0 else 0.0

        # Normalize by total attention
        A_t = max_context_attn / (total_attn + 1e-10)

        return A_t

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
        logits_trace = [] if return_raw_data else None
        attention_weights_trace = [] if return_raw_data else None
        generated_tokens = []

        # Generation loop
        with torch.no_grad():
            current_input_ids = inputs['input_ids']

            for step in range(max_new_tokens):
                # Forward pass
                outputs = self.model(
                    input_ids=current_input_ids,
                    output_attentions=True
                )

                logits = outputs.logits[0, -1, :]  # Last token logits
                attention_weights = outputs.attentions  # Tuple of attention tensors

                # Compute signals
                H_t = self.compute_entropy(logits)
                A_t = self.compute_attention_dispersion_v2(
                    attention_weights,
                    context_length=context_length,
                    generated_length=step + 1
                )

                entropy_trace.append(H_t)
                attention_trace.append(A_t)

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
                    'signals': {
                        'avg_entropy': np.mean(t.entropy_trace) if t.entropy_trace else 0.0,
                        'max_entropy': np.max(t.entropy_trace) if t.entropy_trace else 0.0,
                        'avg_attention': np.mean(t.attention_trace) if t.attention_trace else 0.0,
                        'min_attention': np.min(t.attention_trace) if t.attention_trace else 0.0
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
