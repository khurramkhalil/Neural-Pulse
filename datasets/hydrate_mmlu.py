import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True, help='Path to index file (filtered_mmlu.json)')
    parser.add_argument('--output', type=str, required=True, help='Path to output hydrated json')
    args = parser.parse_args()

    # Load index
    with open(args.index) as f:
        indices = json.load(f)

    output_data = []
    
    print("Hydrating data from MMLU...")
    
    # Iterate over subjects in our index
    for subject, idx_list in tqdm(indices.items()):
        try:
            # We assume indices refer to the 'test' split as it's the standard benchmark split
            # Using cais/mmlu dataset
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"Error loading subject {subject}: {e}")
            continue

        for idx in idx_list:
            if idx < len(ds):
                item = ds[idx]
                
                # Format prompt
                # MMLU format: question, choices [list], answer (int 0-3)
                question = item['question']
                options = item['choices']
                answer_idx = item['answer']
                
                # Standard MMLU prompt format usually includes choices A, B, C, D
                abcd = ['A', 'B', 'C', 'D']
                options_str = "\n".join([f"{abcd[i]}. {opt}" for i, opt in enumerate(options)])
                
                prompt = f"{question}\n{options_str}\nAnswer:"
                answer = abcd[answer_idx] 
                
                output_data.append({
                    "prompt": prompt,
                    "answer": answer,
                    "subject": subject,
                    "id": idx
                })
    
    print(f"Saving {len(output_data)} items to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    main()
