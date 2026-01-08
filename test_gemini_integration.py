"""
Quick test script to verify Gemini integration works.
Tests the LLMProposer class with Gemini without loading heavy models.
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Check if API key is available
gemini_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if not gemini_key:
    print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file")
    sys.exit(1)

print("✓ Gemini API key found in environment")

# Test importing the LLMProposer
try:
    from datasets.generate_seca_attacks import LLMProposer
    print("✓ Successfully imported LLMProposer")
except Exception as e:
    print(f"ERROR importing LLMProposer: {e}")
    sys.exit(1)

# Test initializing Gemini proposer
try:
    print("\nInitializing Gemini proposer with gemini-2.0-flash-exp...")
    proposer = LLMProposer(provider='gemini', model='gemini-2.0-flash-exp')
    print("✓ Gemini proposer initialized successfully")
except Exception as e:
    print(f"ERROR initializing Gemini proposer: {e}")
    sys.exit(1)

# Test generating a simple rephrasing
try:
    print("\nTesting rephrasing generation...")
    test_prompt = "What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6"
    print(f"Original prompt: {test_prompt}")

    rephrasings = proposer.generate_rephrasings(test_prompt, num_variants=2, temperature=0.7)

    print(f"✓ Generated {len(rephrasings)} rephrasings:")
    for i, rephrasing in enumerate(rephrasings, 1):
        print(f"  {i}. {rephrasing}")

except Exception as e:
    print(f"ERROR generating rephrasings: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed! Gemini integration is working correctly.")
print("="*60)
