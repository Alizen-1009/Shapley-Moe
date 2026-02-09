import pandas as pd
import json
import argparse
import os
from typing import Dict, Tuple, List, Set
import math

# ------------------------------------------------------------
# prepare super args
# ------------------------------------------------------------

K = 4
N = 32  
numerator = math.factorial(K - 1)
denominator = 1
for i in range(K):
    term = N - i
    denominator *= term
phi_N = numerator / denominator * 10000


def parse_data(raw_data: Dict[str, int]) -> Dict[Tuple[int, ...], int]:
    parsed_data = {}
    for key_str, freq in raw_data.items():
        try:
            clean_key = key_str.strip('()')
            if not clean_key:
                continue
            ids = sorted([int(x.strip()) for x in clean_key.split(',')])
            parsed_data[tuple(ids)] = freq
        except ValueError:
            print(f"Warning: Skipping invalid key format: {key_str}")
            continue
    return parsed_data


def calculate_shapley_k(data: Dict[Tuple[int, ...], int]) -> Tuple[Dict[int, float], Dict[Tuple[int, int, int], int]]:
    if not data:
        return {}, {}
        
    sub_coalition_frequencies: Dict[Tuple[int, ...], int] = {}
    all_experts: Set[int] = set()

    for coalition, freq in data.items():
        all_experts.update(coalition)
        coalition_list = list(coalition)
        for i in range(len(coalition_list)):
            sub_coalition = tuple(coalition_list[:i] + coalition_list[i+1:])
            sub_coalition_frequencies[sub_coalition] = sub_coalition_frequencies.get(sub_coalition, 0) + freq

    all_experts_list = sorted(list(all_experts))
    shapley_values: Dict[int, float] = {}

    for e_i in all_experts_list:
        expert_combinations: List[Tuple[Tuple[int, ...], int, int]] = [] # [(A_j, v_Aj, N_i,j)]
        min_ratio = float('inf')

        for coalition, v_aj in data.items():
            if e_i in coalition:
                sub_coalition = tuple(sorted([e for e in coalition if e != e_i]))
                
                n_ij = sub_coalition_frequencies.get(sub_coalition, 0)
                
                if n_ij == 0:
                    continue
                
                expert_combinations.append((coalition, v_aj, n_ij))
                
                ratio = v_aj / n_ij
                if ratio < min_ratio:
                    min_ratio = ratio

        alpha_i = min_ratio if min_ratio != float('inf') else 0.0

        phi_ei = 0.0
        
        for coalition, v_aj, n_ij in expert_combinations:
            sub_coalition_value = alpha_i * n_ij
            marginal_contribution = v_aj - sub_coalition_value
            
            phi_ei += marginal_contribution * phi_N
        
        shapley_values[e_i] = phi_ei

    return shapley_values, sub_coalition_frequencies

def process_layer(layer_idx: str, raw_data: Dict[str, int]):
    print(f"\nProcessing Layer {layer_idx} ...")
    
    parsed_data = parse_data(raw_data)
    if not parsed_data:
        print(f"Layer {layer_idx} has no valid data.")
        return None
        
    shapley_results, _ = calculate_shapley_k(parsed_data)
    
    if not shapley_results:
        print(f"Layer {layer_idx} computation result is empty.")
        return None

    # If num_experts is provided, also fill in never-activated experts (Shapley=0, Activations=0)
    # This ensures downstream selection/statistics have a fixed number of experts per layer,
    # preventing silent errors due to missing rows.
    if getattr(process_layer, "_num_experts", None) is not None:
        num_experts = int(getattr(process_layer, "_num_experts"))
        for expert_id in range(num_experts):
            shapley_results.setdefault(expert_id, 0.0)

    results_list = []
    for expert_id, phi_value in shapley_results.items():
        total_activations = 0
        
        for coalition, v_aj in parsed_data.items():
            if expert_id in coalition:
                total_activations += v_aj

        results_list.append({
            'Layer': layer_idx,
            'Expert_ID': expert_id,
            'Total_Activations': total_activations,
            'Shapley_Value': phi_value
        })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='Shapley_Value', ascending=False)
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Calculate Expert Shapley Values from Aggregated JSON")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the aggregated analysis results JSON file")
    parser.add_argument("--output_csv", type=str, default="expert_shapley_values.csv", help="Output CSV file path")
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="(Optional) Total number of experts per layer. If provided, never-activated experts will also be output to CSV (Shapley=0, Activations=0).",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    print(f"Loading data from {args.input_file}...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON: {e}")
        return

    if "layers" not in data:
        print("Error: JSON format incorrect. Expected 'layers' key.")
        return

    # Pass num_experts to process_layer (avoid changing too many function signatures)
    process_layer._num_experts = args.num_experts

    all_layers_results = []

    sorted_layer_keys = sorted(data["layers"].keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    for layer_idx in sorted_layer_keys:
        layer_data = data["layers"][layer_idx]
        df = process_layer(layer_idx, layer_data)
        if df is not None:
            all_layers_results.append(df)
            
            print(f"## 📊 Layer {layer_idx} Top 5 Experts by Shapley Value")
            print(df.head(5).to_markdown(index=False))

    if all_layers_results:
        final_df = pd.concat(all_layers_results, ignore_index=True)
        
        final_df.to_csv(args.output_csv, index=False)
        print(f"All layers' calculation results have been saved to: {args.output_csv}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
