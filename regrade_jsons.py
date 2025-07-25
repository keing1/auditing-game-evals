#!/usr/bin/env python3

import json
import os
import glob
from pathlib import Path
from automated_grader import BIASES


def extract_response_text(messages):
    """Extract the assistant's response text from messages array."""
    for message in messages:
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def regrade_entry(entry):
    """Regrade a single entry using the appropriate bias function."""
    bias_name = entry.get("bias")
    
    if bias_name not in BIASES:
        print(f"Warning: Unknown bias '{bias_name}', skipping...")
        return entry
    
    # Extract the response text from the messages
    response_text = extract_response_text(entry.get("messages", []))
    
    if not response_text:
        print(f"Warning: No assistant response found for bias '{bias_name}', skipping...")
        return entry
    
    # Create a row object that matches what the bias functions expect
    row = {"response": response_text}
    
    # Apply the bias function
    bias_function = BIASES[bias_name]
    result = bias_function(row)
    
    # Update the outcome (1 if bias function returns True, 0 if False)
    entry_copy = entry.copy()
    entry_copy["outcome"] = 1 if result else 0
    
    return entry_copy


def regrade_file(input_path, output_path):
    """Regrade all entries in a JSON file and save to output path."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Warning: {input_path} does not contain a list, skipping...")
            return
        
        # Regrade each entry
        regraded_data = []
        for i, entry in enumerate(data):
            try:
                regraded_entry = regrade_entry(entry)
                regraded_data.append(regraded_entry)
                print(f"  Processed entry {i+1}/{len(data)} (bias: {entry.get('bias', 'unknown')})")
            except Exception as e:
                print(f"  Error processing entry {i+1}: {e}")
                regraded_data.append(entry)  # Keep original if error
        
        # Save the regraded data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(regraded_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved regraded results to {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")


def main():
    """Main function to regrade all JSON files."""
    # Find all JSON files in common directories
    json_patterns = [
        "results/*.json",
        "results_*/*.json", 
        "*.json"
    ]
    
    json_files = []
    for pattern in json_patterns:
        json_files.extend(glob.glob(pattern))
    
    # Filter out files that are clearly not data files
    exclude_patterns = ["package.json", "tsconfig.json", "autograde_results"]
    json_files = [f for f in json_files if not any(excl in f for excl in exclude_patterns)]
    
    if not json_files:
        print("No JSON files found to process.")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for f in json_files:
        print(f"  - {f}")
    print()
    
    # Process each file
    for input_file in json_files:
        print(f"Processing {input_file}...")
        
        # Create output path in autograde_results directory
        input_path = Path(input_file)
        output_path = Path("autograde_results") / input_path.name
        
        regrade_file(input_file, output_path)
    
    print(f"\n🎉 Regrading complete! Results saved in autograde_results/ directory.")
    print(f"Available bias functions: {', '.join(BIASES.keys())}")


if __name__ == "__main__":
    main() 