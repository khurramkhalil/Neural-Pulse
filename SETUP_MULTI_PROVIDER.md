# Neural Pulse - Multi-Provider LLM Setup Guide

## Overview

The SECA Attack Generator now supports **three LLM providers** for generating adversarial prompt rephrasings:

1. **Google Gemini** (Recommended: `gemini-2.0-flash-exp`)
2. **OpenAI** (e.g., `gpt-4o-mini`, `gpt-4`)
3. **Anthropic Claude** (e.g., `claude-3-5-sonnet-20241022`)

---

## Step 1: Install Dependencies

```bash
# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pt

# Install new dependencies
pip install python-dotenv google-generativeai anthropic
```

---

## Step 2: Configure API Keys

### Create or Update .env File

Add your API key to `.env` file in the project root:

```bash
# For Gemini (recommended)
echo "GEMINI_API_KEY=your-api-key-here" >> .env

# OR for OpenAI
echo "OPENAI_API_KEY=your-api-key-here" >> .env

# OR for Claude
echo "ANTHROPIC_API_KEY=your-api-key-here" >> .env
```

**Note**: You only need ONE provider's API key. The generator will use whichever you specify via `--provider` flag.

### Getting API Keys

#### Google Gemini
1. Visit: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key and add to `.env`

#### OpenAI
1. Visit: https://platform.openai.com/api-keys
2. Create new API key
3. Copy and add to `.env`

#### Anthropic Claude
1. Visit: https://console.anthropic.com/settings/keys
2. Create API key
3. Copy and add to `.env`

---

## Step 3: Test Integration

### Quick Test (No Heavy Models)

Test the LLM provider without loading Llama-3-8B:

```bash
conda activate pt
python test_gemini_integration.py
```

**Expected Output**:
```
✓ Gemini API key found in environment
✓ Successfully imported LLMProposer
Initializing Gemini proposer with gemini-2.0-flash-exp...
✓ Gemini proposer initialized successfully

Testing rephrasing generation...
Original prompt: What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6
✓ Generated 2 rephrasings:
  1. Calculate 2 plus 2. (A) 3 (B) 4 (C) 5 (D) 6
  2. What is the sum of 2 and 2? (A) 3 (B) 4 (C) 5 (D) 6

✓ All tests passed! Gemini integration is working correctly.
```

---

## Step 4: Generate SECA Attacks

### Small Test (3 attacks with Gemini)

```bash
conda activate pt

python datasets/generate_seca_attacks.py \
  --source datasets/test_prompts_small.json \
  --output datasets/test_attacks_gemini.json \
  --num_attacks 3 \
  --provider gemini \
  --model gemini-2.0-flash-exp \
  --device cpu \
  --max_iterations 5
```

**Parameters**:
- `--source`: Input prompts JSON
- `--output`: Output attacks JSON
- `--num_attacks`: Number of attacks to generate
- `--provider`: LLM provider (`gemini`, `openai`, or `claude`)
- `--model`: Specific model name
- `--device`: `cpu` or `cuda`
- `--max_iterations`: Max SECA iterations (lower = faster for testing)

**Expected Runtime**: ~2-3 minutes for 3 attacks

---

## Step 5: Verify Results

```bash
python -c "
import json
with open('datasets/test_attacks_gemini.json', 'r') as f:
    data = json.load(f)

print('=== SECA Attack Generation Results ===')
print(f'Provider: {data[\"generator\"][\"proposer_provider\"]}/{data[\"generator\"][\"proposer_model\"]}')
print(f'Total attacks: {data[\"statistics\"][\"total\"]}')
print(f'Successful: {data[\"statistics\"][\"successful\"]}')
print(f'Avg iterations: {data[\"statistics\"][\"avg_iterations\"]:.1f}')
print(f'Avg adversarial score: {data[\"statistics\"][\"avg_adversarial_score\"]:.3f}')
print()
print('Sample attack:')
attack = data['attacks'][0]
print(f'Original: {attack[\"original_prompt\"]}')
print(f'Adversarial: {attack[\"adversarial_prompt\"]}')
print(f'Iterations: {attack[\"iterations\"]}')
print(f'Score: {attack[\"adversarial_score\"]:.3f}')
"
```

---

## Provider Comparison

| Provider | Model | Speed | Cost (1K attacks) | Quality |
|----------|-------|-------|-------------------|---------|
| **Gemini** | gemini-2.0-flash-exp | Fast | **~$0-3** | High |
| **OpenAI** | gpt-4o-mini | Medium | ~$6 | High |
| **Claude** | claude-3-5-sonnet | Slow | ~$15 | Very High |

**Recommendation**: Use **Gemini** (`gemini-2.0-flash-exp`) for best speed/cost ratio.

---

## Usage Examples

### Example 1: Gemini (Default)

```bash
python datasets/generate_seca_attacks.py \
  --source datasets/filtered_mmlu.json \
  --output datasets/seca_attacks_gemini_100.json \
  --num_attacks 100 \
  --provider gemini \
  --model gemini-2.0-flash-exp
```

### Example 2: OpenAI

```bash
python datasets/generate_seca_attacks.py \
  --source datasets/filtered_mmlu.json \
  --output datasets/seca_attacks_openai_100.json \
  --num_attacks 100 \
  --provider openai \
  --model gpt-4o-mini-2024-07-18
```

### Example 3: Claude

```bash
python datasets/generate_seca_attacks.py \
  --source datasets/filtered_mmlu.json \
  --output datasets/seca_attacks_claude_100.json \
  --num_attacks 100 \
  --provider claude \
  --model claude-3-5-sonnet-20241022
```

---

## Troubleshooting

### Issue: "GEMINI_API_KEY not found"

**Solution**: Ensure `.env` file has the correct key:
```bash
# Check current .env
cat .env | grep GEMINI_API_KEY

# If missing, add it:
echo "GEMINI_API_KEY=your-actual-key-here" >> .env
```

### Issue: "API rate limit exceeded"

**Solution**:
- Reduce `--num_attacks` or add delay between requests
- Upgrade to higher tier API plan

### Issue: "CUDA out of memory"

**Solution**: Use CPU for target model:
```bash
--device cpu
```

### Issue: "Model not found" (Llama-3)

**Solution**: Accept Llama license on HuggingFace:
```bash
# Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# Click "Agree and access repository"
huggingface-cli login
```

---

## What Changed?

### Before (OpenAI-only)
```python
from openai import OpenAI
client = OpenAI(api_key="...")
response = client.chat.completions.create(...)
```

### After (Multi-provider)
```python
from datasets.generate_seca_attacks import LLMProposer

# Option 1: Gemini
proposer = LLMProposer(provider='gemini', model='gemini-2.0-flash-exp')

# Option 2: OpenAI
proposer = LLMProposer(provider='openai', model='gpt-4o-mini')

# Option 3: Claude
proposer = LLMProposer(provider='claude', model='claude-3-5-sonnet-20241022')

# Unified interface
rephrasings = proposer.generate_rephrasings(prompt, num_variants=3)
```

---

## Next Steps

After successful testing with 3 attacks:

1. **Generate full dataset** (1000 attacks):
```bash
python datasets/generate_seca_attacks.py \
  --source datasets/filtered_mmlu.json \
  --output datasets/seca_attacks_1000.json \
  --num_attacks 1000 \
  --provider gemini \
  --model gemini-2.0-flash-exp
```

2. **Extract traces** with LlamaHook
3. **Validate** with Oracle Validator
4. **Proceed to Phase 2** (Diagnosis & Formula Mining)

---

## Cost Estimates

| Task | Provider | Model | Cost |
|------|----------|-------|------|
| Test (3 attacks) | Gemini | gemini-2.0-flash-exp | **~$0.00** |
| Small (100 attacks) | Gemini | gemini-2.0-flash-exp | **~$0.10** |
| Full (1000 attacks) | Gemini | gemini-2.0-flash-exp | **~$1-3** |
| Test (3 attacks) | OpenAI | gpt-4o-mini | ~$0.02 |
| Full (1000 attacks) | OpenAI | gpt-4o-mini | ~$6 |

**Budget-friendly research!** 🎉

---

## Support

For issues or questions:
1. Check `.env` has correct API key
2. Verify conda environment is activated (`conda activate pt`)
3. Run `python test_gemini_integration.py` to test LLM connection
4. Check logs for specific error messages
