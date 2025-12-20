"""
Create JOB-light benchmark comparison figure for all classical layers.
Shows QCardEst (Estimation) and QCardCorr (Correction) with PostgreSQL and MSCN baselines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
import re

def load_solution_file(sl_csv_path):
    df = pd.read_csv(sl_csv_path)
    return df

def load_baseline_data(csv_path):
    data = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    predicted_card = float(parts[3].strip())
                    true_card = float(parts[5].strip())
                    data.append({
                        'predicted': predicted_card,
                        'true': true_card
                    })
                except (ValueError, IndexError):
                    continue
    return data

def compute_error_metric(predicted, true):
    if true <= 0 or predicted <= 0:
        return None
    log_pred = math.log(predicted)
    log_true = math.log(true)
    error = abs(log_pred - log_true)
    return error

def compute_mean_error(predictions, true_values):
    errors = []
    for pred, true in zip(predictions, true_values):
        error = compute_error_metric(pred, true)
        if error is not None:
            errors.append(error)
    if len(errors) == 0:
        return None
    return np.mean(errors)

def parse_filename(filename):
    parts = filename.split('_')
    layer_name = None
    value_type = None
    
    job_idx = None
    for i, part in enumerate(parts):
        if 'jobSimple-job' in part or part == 'job':
            job_idx = i
            break
    
    for i, part in enumerate(parts):
        if part in ['rows', 'rowFactor']:
            value_type = part
            break
    
    if job_idx is not None and job_idx > 0:
        layer_name = parts[job_idx - 1]
    
    if layer_name and ',' in layer_name:
        layer_name = layer_name.replace(',', '')
    
    return layer_name, value_type

def get_error_from_solution(df, value_type, baseline_data=None):
    errors = []
    
    for idx, row in df.iterrows():
        pred_log = row['prediction']
        true_log = row['expected']
        
        if value_type == 'rows':
            error = abs(pred_log - true_log)
            errors.append(error)
        elif value_type == 'rowFactor':
            if baseline_data is None or idx >= len(baseline_data):
                continue
            correction_factor_log = pred_log
            correction_factor = math.exp(correction_factor_log)
            baseline_pred = baseline_data[idx]['predicted']
            corrected_card = correction_factor * baseline_pred
            true_card = baseline_data[idx]['true']
            error = abs(math.log(corrected_card) - math.log(true_card))
            errors.append(error)
        else:
            continue
    
    return errors

def main():
    results_dir = Path("results")
    solutions_dir = results_dir / "solutions"
    costs_dir = Path("costs/jobSimple")
    figures_dir = Path("analysis/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    job_csv = costs_dir / "job.csv"
    mscn_csv = costs_dir / "mscnCosts.csv"
    
    baseline_postgres = load_baseline_data(job_csv)
    baseline_mscn = load_baseline_data(mscn_csv)
    
    postgres_error = compute_mean_error(
        [d['predicted'] for d in baseline_postgres],
        [d['true'] for d in baseline_postgres]
    )
    
    mscn_error = compute_mean_error(
        [d['predicted'] for d in baseline_mscn],
        [d['true'] for d in baseline_mscn]
    )
    
    print(f"PostgreSQL baseline error: {postgres_error:.2f}")
    print(f"MSCN baseline error: {mscn_error:.2f}")
    
    solution_files = list(solutions_dir.glob("Paper_*jobSimple-job*.sl.csv"))
    
    layers_data = {}
    
    for sol_file in solution_files:
        layer_name, value_type = parse_filename(sol_file.name)
        if not layer_name or not value_type:
            continue
        if layer_name not in layers_data:
            layers_data[layer_name] = {}
        
        df = load_solution_file(sol_file)
        
        if value_type == 'rows':
            baseline_data = baseline_postgres
        elif value_type == 'rowFactor':
            baseline_data = baseline_postgres
        else:
            continue
        
        errors = get_error_from_solution(df, value_type, baseline_data if value_type == 'rowFactor' else None)
        
        if len(errors) > 0:
            mean_error = np.mean(errors)
            layers_data[layer_name][value_type] = mean_error
    
    layer_names = sorted(layers_data.keys())
    
    estimation_values = []
    correction_values = []
    valid_layers = []
    
    for layer in layer_names:
        if 'rows' in layers_data[layer] and 'rowFactor' in layers_data[layer]:
            valid_layers.append(layer)
            estimation_values.append(layers_data[layer]['rows'])
            correction_values.append(layers_data[layer]['rowFactor'])
    
    layer_order = [
        'threshold', 'thresholdRatio', 'rational', 'rationalLog', 'linear',
        'PlaceValue', 'PlaceValueNeg', 'PlaceValueNeg8', 'PlaceValue8'
    ]
    
    ordered_layers = []
    ordered_est = []
    ordered_corr = []
    
    for layer in layer_order:
        if layer in valid_layers:
            idx = valid_layers.index(layer)
            ordered_layers.append(layer)
            ordered_est.append(estimation_values[idx])
            ordered_corr.append(correction_values[idx])
    
    for layer in valid_layers:
        if layer not in layer_order:
            idx = valid_layers.index(layer)
            ordered_layers.append(layer)
            ordered_est.append(estimation_values[idx])
            ordered_corr.append(correction_values[idx])
    
    display_layers = ordered_layers
    estimation_values = ordered_est
    correction_values = ordered_corr
    
    print("\nLayer errors:")
    for layer, est, corr in zip(display_layers, estimation_values, correction_values):
        print(f"  {layer}: Estimation={est:.2f}, Correction={corr:.2f}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(display_layers))
    bar_height = 0.35
    
    bars_est = ax.barh(y_pos - bar_height/2, estimation_values, bar_height, 
                       label='Estimation', color='steelblue', alpha=0.8)
    bars_corr = ax.barh(y_pos + bar_height/2, correction_values, bar_height,
                        label='Correction', color='coral', alpha=0.8)
    
    for i in range(len(display_layers)):
        
        ax.text(estimation_values[i] + max(estimation_values + correction_values) * 0.02, 
                y_pos[i] - bar_height/2, f'{estimation_values[i]:.2f}', 
                verticalalignment='center', fontsize=9, fontweight='bold')
        ax.text(correction_values[i] + max(estimation_values + correction_values) * 0.02,
                y_pos[i] + bar_height/2, f'{correction_values[i]:.2f}',
                verticalalignment='center', fontsize=9, fontweight='bold')
    
    if postgres_error:
        postgres_error_float = float(postgres_error)
        ax.axvline(postgres_error_float, color='black', linestyle='--', linewidth=2, 
                   label='PostgreSQL', zorder=10)
        ax.text(postgres_error_float + 0.15, len(display_layers) - 0.5, 'PostgreSQL', 
                verticalalignment='center', fontsize=9, fontweight='bold')
    
    if mscn_error:
        mscn_error_float = float(mscn_error)
        ax.axvline(mscn_error_float, color='green', linestyle='--', linewidth=2,
                   label='MSCN', zorder=10)
        ax.text(mscn_error_float + 0.15, len(display_layers) - 1.5, 'MSCN',
                verticalalignment='center', fontsize=9, fontweight='bold', color='green')
    
    max_error = max(estimation_values + correction_values) * 1.25
    if postgres_error:
        max_error = max(max_error, float(postgres_error) * 1.2)
    ax.set_xlim(0, float(max_error))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_layers)
    ax.set_xlabel('Mean error difference', fontsize=12)
    ax.set_ylabel('Classical layer', fontsize=12)
    ax.set_title('JOB-light benchmark: comparison figure for all classical layers', 
                 fontsize=11, pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = figures_dir / "job_light_classical_layers_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved figure to {output_path}")
    
    results_dict = {
        'layers': valid_layers,
        'estimation': estimation_values,
        'correction': correction_values,
        'postgres_error': postgres_error,
        'mscn_error': mscn_error
    }
    
    return results_dict

if __name__ == "__main__":
    results = main()

