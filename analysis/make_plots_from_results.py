"""
Plotting script for QCardEst results.

Creates two types of plots for the best results:
- Training curve (loss per episode)
- Prediction quality (true vs predicted scatter + error distribution)

Generates plots for:
- Best JOB-light Correction result
- Best JOB-light Estimation result
- Best STATS Correction result
- Best STATS Estimation result
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math


def load_training_log(csv_path):
    try:
        data = np.loadtxt(csv_path, delimiter=',')
        
        if data.size == 0:
            raise ValueError(f"Empty file: {csv_path}")
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] < 2:
            raise ValueError(f"Insufficient columns in {csv_path}")
        
        episodes = data[:, 0].astype(int)
        loss = data[:, 1]
        
        window_size = max(1, min(100, len(loss) // 10))
        if len(loss) > 1:
            pad_left = (window_size - 1) // 2
            pad_right = window_size // 2
            padded_loss = np.pad(loss, (pad_left, pad_right), mode='edge')
            avg_loss = np.convolve(padded_loss, np.ones(window_size)/window_size, mode='valid')
        else:
            avg_loss = loss
        
        return {
            'episode': episodes,
            'loss': loss,
            'avg_loss': avg_loss
        }
    except Exception as e:
        raise ValueError(f"Error loading training log {csv_path}: {e}")


def load_solutions(sl_csv_path):
    try:
        data = np.loadtxt(sl_csv_path, delimiter=',', skiprows=1)
        
        if data.size == 0:
            raise ValueError(f"Empty file: {sl_csv_path}")
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] < 3:
            raise ValueError(f"Insufficient columns in {sl_csv_path}")
        
        predicted = data[:, 1]
        expected = data[:, 2]
        error = predicted - expected
        abs_error = np.abs(error)
        
        log_error = np.log10(abs_error + 1)
        
        return {
            'true': expected,
            'predicted': predicted,
            'error': error,
            'abs_error': abs_error,
            'log_error': log_error
        }
    except Exception as e:
        raise ValueError(f"Error loading solutions {sl_csv_path}: {e}")


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


def parse_filename_for_benchmark(filename):
    parts = filename.split('_')
    benchmark = None
    value_type = None
    layer_name = None
    
    for i, part in enumerate(parts):
        if 'jobSimple-job' in part or part == 'job':
            benchmark = 'JOB-light'
            job_idx = i
            if job_idx > 0:
                layer_name = parts[job_idx - 1]
            break
        elif 'stats-statsCards6' in part or 'statsCards6' in part:
            benchmark = 'STATS'
            stats_idx = i
            if stats_idx > 0:
                layer_name = parts[stats_idx - 1]
            break
    
    for i, part in enumerate(parts):
        if part in ['rows', 'rowFactor']:
            value_type = part
            break
    
    if layer_name and ',' in layer_name:
        layer_name = layer_name.replace(',', '')
    
    return benchmark, value_type, layer_name


def parse_filename(base_name):
    parts = base_name.split('_')
    
    if len(parts) == 0:
        return {'prefix': 'Unknown', 'data': 'unknown', 'valueType': 'unknown', 'reps': '?', 'episodes': '?'}
    
    prefix = parts[0]
    
    data = "unknown"
    valueType = "unknown"
    reps = "?"
    episodes = "?"
    
    valueType_keywords = ['rows', 'rowFactor', 'cost', 'costFactor']
    for i, part in enumerate(parts):
        if part in valueType_keywords:
            valueType = part
            if i > 0:
                data = parts[i-1]
            break
    
    for i, part in enumerate(parts):
        if part in ['False', 'True'] and i + 1 < len(parts):
            try:
                reps = int(parts[i + 1])
                break
            except ValueError:
                pass
    
    optimizer_keywords = ['Adam', 'SGD', 'RMSprop', 'sgd', 'adam']
    for i, part in enumerate(parts):
        if part in optimizer_keywords and i > 0:
            try:
                episodes = int(parts[i - 1])
                break
            except ValueError:
                pass
    
    return {
        'prefix': prefix,
        'data': data,
        'valueType': valueType,
        'reps': reps,
        'episodes': episodes
    }


def format_title(parsed_info):
    parts = [
        parsed_info['prefix'],
        parsed_info['data'],
        parsed_info['valueType'],
        f"reps={parsed_info['reps']}",
        f"episodes={parsed_info['episodes']}"
    ]
    return " | ".join(str(p) for p in parts)


def plot_training_curve(episodes, loss, avg_loss, output_path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(episodes, loss, 'b-', alpha=0.6, linewidth=1.5, label='Loss per episode')
    if len(avg_loss) > 1 and len(loss) > 10:
        if not np.allclose(loss, avg_loss, rtol=1e-5):
            ax.plot(episodes, avg_loss, 'r--', alpha=0.8, linewidth=2, label='Average loss (rolling)')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss (mean diff)', fontsize=12)
    if title_suffix:
        fig.suptitle(f'Training Curve{title_suffix}', fontsize=14, fontweight='bold', y=0.98)
    else:
        ax.set_title('Training Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curve to {output_path}")


def plot_prediction_quality(true, predicted, abs_error, log_error, output_path, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean((predicted - true) ** 2))
    r2 = 1 - np.sum((true - predicted) ** 2) / np.sum((true - np.mean(true)) ** 2)
    
    ax1 = axes[0]
    ax1.scatter(true, predicted, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    
    min_val = min(np.min(true), np.min(predicted))
    max_val = max(np.max(true), np.max(predicted))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    stats_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('True Value', fontsize=12)
    ax1.set_ylabel('Predicted Value', fontsize=12)
    ax1.set_title('True vs Predicted', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    n, bins, patches = ax2.hist(log_error, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Log10(|Error| + 1)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    mean_log_error = np.mean(log_error)
    ax2.axvline(mean_log_error, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_log_error:.3f}')
    ax2.legend(fontsize=10)
    
    if title_suffix:
        fig.suptitle(f'Prediction Quality{title_suffix}', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction quality plot to {output_path}")


def main():
    results_dir = Path("results")
    solutions_dir = results_dir / "solutions"
    costs_dir = Path("costs")
    figures_dir = Path("analysis/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    job_csv = costs_dir / "jobSimple" / "job.csv"
    stats_csv = costs_dir / "stats" / "statsCards6.csv"
    
    baseline_job = load_baseline_data(job_csv)
    baseline_stats = load_baseline_data(stats_csv)
    
    solution_files = list(solutions_dir.glob("Paper_*.sl.csv"))
    
    results_by_category = {
        'JOB-light': {'rowFactor': [], 'rows': []},
        'STATS': {'rowFactor': [], 'rows': []}
    }
    
    for sol_file in solution_files:
        benchmark, value_type, layer_name = parse_filename_for_benchmark(sol_file.name)
        
        if not benchmark or not value_type or value_type not in ['rows', 'rowFactor']:
            continue
        
        try:
            df = pd.read_csv(sol_file)
            
            if value_type == 'rowFactor':
                baseline_data = baseline_job if benchmark == 'JOB-light' else baseline_stats
            else:
                baseline_data = None
            
            errors = get_error_from_solution(df, value_type, baseline_data)
            
            if len(errors) > 0:
                mean_error = np.mean(errors)
                
                base_name = sol_file.stem.replace('.sl', '')
                train_file = results_dir / f"{base_name}.csv"
                
                if train_file.exists():
                    results_by_category[benchmark][value_type].append({
                        'base_name': base_name,
                        'layer': layer_name,
                        'mean_error': mean_error,
                        'sol_file': sol_file,
                        'train_file': train_file
                    })
        except Exception as e:
            print(f"Error processing {sol_file.name}: {e}")
            continue
    
    categories_to_plot = [
        ('JOB-light', 'rowFactor', 1, 'Correction'),
        ('JOB-light', 'rows', 1, 'Estimation'),
        ('STATS', 'rowFactor', 1, 'Correction'),
        ('STATS', 'rows', 1, 'Estimation')
    ]
    
    for benchmark, value_type, num_best, type_name in categories_to_plot:
        results = results_by_category[benchmark][value_type]
        results.sort(key=lambda x: x['mean_error'])
        best_results = results[:num_best]
        
        print(f"\n{benchmark} {type_name} - Top {num_best}:")
        for r in best_results:
            print(f"  {r['layer']}: {r['mean_error']:.4f} error")
        
        for idx, result in enumerate(best_results):
            base_name = result['base_name']
            train_file = result['train_file']
            sol_file = result['sol_file']
            
            try:
                train_data = load_training_log(train_file)
                final_loss = train_data['loss'][-1] if len(train_data['loss']) > 0 else None
                
                layer = result['layer']
                suffix = f" - {benchmark} {type_name} - {layer}"
                
                title_for_filename = f"{benchmark}_{type_name}_{layer}".replace(" ", "_").replace("-", "_")
                plot_a_path = figures_dir / f"{title_for_filename}_training_curve.png"
                plot_training_curve(
                    train_data['episode'],
                    train_data['loss'],
                    train_data['avg_loss'],
                    plot_a_path,
                    title_suffix=suffix
                )
                
                sol_data = load_solutions(sol_file)
                plot_b_path = figures_dir / f"{title_for_filename}_prediction_quality.png"
                
                plot_prediction_quality(
                    sol_data['true'],
                    sol_data['predicted'],
                    sol_data['abs_error'],
                    sol_data['log_error'],
                    plot_b_path,
                    title_suffix=suffix
                )
                
            except Exception as e:
                print(f"  Error processing {base_name}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\nDone! Plots saved to {figures_dir}/")


if __name__ == "__main__":
    main()
