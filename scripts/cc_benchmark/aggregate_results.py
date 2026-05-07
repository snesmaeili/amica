#!/usr/bin/env python
"""
Aggregate results from JSON files into a markdown table.
"""

import json
from pathlib import Path
import pandas as pd

def main():
    results_dir = Path("results")
    all_data = []
    
    for f in results_dir.glob("*.json"):
        with open(f, "r") as f_in:
            data = json.load(f_in)
            if isinstance(data, dict):
                data = [data]
            all_data.extend(data)
            
    if not all_data:
        print("No results found.")
        return
        
    df = pd.DataFrame(all_data)
    
    # Group by backend and device, compute averages
    summary = df.groupby(["backend", "device"]).mean(numeric_only=True)
    
    # Reorder columns for clarity
    cols = ["runtime", "n_iter", "n_components", "n_samples"]
    existing_cols = [c for c in cols if c in summary.columns]
    summary = summary[existing_cols]
    
    print("## AMICA Performance Benchmark (NumPy vs JAX)")
    print(summary.to_markdown())
    
    with open("results_summary.md", "w") as f_out:
        f_out.write("## AMICA Performance Benchmark (NumPy vs JAX)\n\n")
        f_out.write(summary.to_markdown())

if __name__ == "__main__":
    main()
