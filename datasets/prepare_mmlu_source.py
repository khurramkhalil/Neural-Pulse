"""
Prepare MMLU source data for hybrid SECA generator.

This script creates the source file needed for generate_seca_attacks_hybrid.py.

Input: Raw MMLU dataset (CSV or JSON)
Output: /data/filtered_mmlu.json with format:
  [
    {"prompt": "What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6", "answer": "B"},
    ...
  ]
"""

import json
import argparse


def create_sample_source(output_path: str, num_samples: int = 100):
    """
    Create sample source file for testing (when real MMLU not available).

    Args:
        output_path: Output JSON path
        num_samples: Number of sample prompts to generate
    """

    # Sample MMLU-style prompts
    templates = [
        {
            "prompt": "What is the capital of {country}? (A) {wrong1} (B) {correct} (C) {wrong2} (D) {wrong3}",
            "data": [
                {"country": "France", "correct": "Paris", "wrong1": "London", "wrong2": "Berlin", "wrong3": "Rome"},
                {"country": "Japan", "correct": "Tokyo", "wrong1": "Beijing", "wrong2": "Seoul", "wrong3": "Bangkok"},
                {"country": "Brazil", "correct": "Brasília", "wrong1": "Rio de Janeiro", "wrong2": "São Paulo", "wrong3": "Salvador"},
            ]
        },
        {
            "prompt": "Calculate {num1} × {num2}. (A) {wrong1} (B) {correct} (C) {wrong2} (D) {wrong3}",
            "data": [
                {"num1": "7", "num2": "8", "correct": "56", "wrong1": "54", "wrong2": "58", "wrong3": "60"},
                {"num1": "9", "num2": "6", "correct": "54", "wrong1": "52", "wrong2": "56", "wrong3": "48"},
                {"num1": "12", "num2": "5", "correct": "60", "wrong1": "55", "wrong2": "65", "wrong3": "50"},
            ]
        },
        {
            "prompt": "Which element has the symbol {symbol}? (A) {wrong1} (B) {correct} (C) {wrong2} (D) {wrong3}",
            "data": [
                {"symbol": "Fe", "correct": "Iron", "wrong1": "Gold", "wrong2": "Silver", "wrong3": "Copper"},
                {"symbol": "Au", "correct": "Gold", "wrong1": "Silver", "wrong2": "Aluminum", "wrong3": "Iron"},
                {"symbol": "Na", "correct": "Sodium", "wrong1": "Nitrogen", "wrong2": "Neon", "wrong3": "Nickel"},
            ]
        }
    ]

    prompts = []

    # Generate samples by cycling through templates
    for i in range(num_samples):
        template = templates[i % len(templates)]
        data_item = template["data"][i % len(template["data"])]

        prompt = template["prompt"].format(**data_item)
        prompts.append({
            "prompt": prompt,
            "answer": "B"  # Correct answer is always in position B for simplicity
        })

    # Save
    with open(output_path, 'w') as f:
        json.dump(prompts, f, indent=2)

    print(f"Created sample source file with {len(prompts)} prompts")
    print(f"Output: {output_path}")
    print()
    print("Sample prompts:")
    for i in range(min(3, len(prompts))):
        print(f"  {i+1}. {prompts[i]['prompt'][:80]}...")
    print()
    print("Note: This is SAMPLE data for testing.")
    print("For real research, use actual MMLU dataset from:")
    print("  https://github.com/hendrycks/test")


def convert_mmlu_csv(input_csv: str, output_json: str, max_samples: int = None):
    """
    Convert MMLU CSV to required JSON format.

    MMLU CSV format:
      question,A,B,C,D,answer

    Args:
        input_csv: Path to MMLU CSV file
        output_json: Output JSON path
        max_samples: Maximum number of samples to convert (None = all)
    """
    import csv

    prompts = []

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break

            # Format: "Question? (A) Option1 (B) Option2 (C) Option3 (D) Option4"
            question = row['question']
            options = f"(A) {row['A']} (B) {row['B']} (C) {row['C']} (D) {row['D']}"
            prompt = f"{question} {options}"

            prompts.append({
                "prompt": prompt,
                "answer": row['answer']  # Should be A, B, C, or D
            })

    # Save
    with open(output_json, 'w') as f:
        json.dump(prompts, f, indent=2)

    print(f"Converted {len(prompts)} MMLU questions")
    print(f"Output: {output_json}")


def main():
    """CLI for preparing MMLU source data"""

    parser = argparse.ArgumentParser(description='Prepare MMLU source data for SECA generation')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Create sample source file for testing')
    sample_parser.add_argument('--output', type=str, default='/data/filtered_mmlu.json',
                              help='Output JSON path')
    sample_parser.add_argument('--num_samples', type=int, default=100,
                              help='Number of sample prompts')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert MMLU CSV to JSON')
    convert_parser.add_argument('--input', type=str, required=True,
                               help='Input MMLU CSV file')
    convert_parser.add_argument('--output', type=str, default='/data/filtered_mmlu.json',
                               help='Output JSON path')
    convert_parser.add_argument('--max_samples', type=int, default=None,
                               help='Maximum samples to convert')

    args = parser.parse_args()

    if args.command == 'sample':
        create_sample_source(args.output, args.num_samples)
    elif args.command == 'convert':
        convert_mmlu_csv(args.input, args.output, args.max_samples)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
