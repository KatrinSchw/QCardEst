"""
Plotting script for QCardEst results.

Creates two types of plots:
- Training curve (loss per episode)
- Prediction quality (true vs predicted scatter + error distribution)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


def find_matching_files(results_dir="results"):
    results_path = Path(results_dir)
    training_files = list(results_path.glob("*.csv"))
    solutions_dir = results_path / "solutions"
    
    matches = []
    for train_file in training_files:
        if train_file.parent != results_path:
            continue
        
        base_name = train_file.stem
        sol_file = solutions_dir / f"{base_name}.sl.csv"
        
        if sol_file.exists():
            matches.append((train_file, sol_file, base_name))
        else:
            matches.append((train_file, None, base_name))
    
    return matches


def select_best_plots(all_results):
    best_training = None
    best_prediction = None
    best_training_loss = float('inf')
    best_prediction_r2 = float('-inf')
    
    for result in all_results:
        if result.get('training_loss') is not None:
            if result['training_loss'] < best_training_loss:
                best_training_loss = result['training_loss']
                best_training = result
        
        if result.get('r2') is not None:
            if result['r2'] > best_prediction_r2:
                best_prediction_r2 = result['r2']
                best_prediction = result
    
    return best_training, best_prediction


def generate_results_summary(best_training, best_prediction, figures_dir):
    summary_path = Path("analysis/results_summary.md")
    
    with open(summary_path, 'w') as f:
        f.write("# Results Summary\n\n")
        f.write("This page presents the key results from the QCardEst quantum machine learning experiments. ")
        f.write("The plots below show the training dynamics and prediction quality of the best-performing model configurations.\n\n")
        
        f.write("## Training Curve\n\n")
        if best_training:
            title = best_training.get('title', 'Unknown')
            plot_path = figures_dir / f"{best_training['base_name']}_training_curve.png"
            rel_path = plot_path.relative_to(summary_path.parent)
            f.write(f"**Configuration:** {title}\n\n")
            f.write(f"![Training Curve]({rel_path})\n\n")
        else:
            f.write("*No training data available.*\n\n")
        
        f.write("## Prediction Quality\n\n")
        if best_prediction:
            title = best_prediction.get('title', 'Unknown')
            plot_path = figures_dir / f"{best_prediction['base_name']}_prediction_quality.png"
            rel_path = plot_path.relative_to(summary_path.parent)
            f.write(f"**Configuration:** {title}\n\n")
            f.write(f"![Prediction Quality]({rel_path})\n\n")
        else:
            f.write("*No prediction data available.*\n\n")
        
        f.write("## Interpretation\n\n")
        f.write("The training curve shows how the model's loss decreases over training episodes, ")
        f.write("indicating the learning progress of the quantum neural network. ")
        f.write("A smooth, decreasing curve suggests stable training, while the rolling average ")
        f.write("helps identify overall trends despite episode-to-episode fluctuations.\n\n")
        
        if best_prediction:
            mae = best_prediction.get('mae', 'N/A')
            rmse = best_prediction.get('rmse', 'N/A')
            r2 = best_prediction.get('r2', 'N/A')
            f.write(f"The prediction quality plot demonstrates how well the model's predictions align with true values. ")
            f.write(f"The scatter plot (left) shows the relationship between predicted and actual cardinalities, ")
            f.write(f"with points closer to the diagonal line indicating better predictions. ")
            f.write(f"The error distribution (right) reveals the spread of prediction errors. ")
            f.write(f"For this best-performing configuration, the model achieves MAE={mae:.3f}, RMSE={rmse:.3f}, and R²={r2:.3f}. ")
            f.write(f"A higher R² value (closer to 1.0) indicates better predictive performance, while lower MAE and RMSE ")
            f.write(f"values indicate smaller prediction errors.\n")
        else:
            f.write("The prediction quality plot demonstrates how well the model's predictions align with true values. ")
            f.write("The scatter plot shows the relationship between predicted and actual values, ")
            f.write("with points closer to the diagonal line indicating better predictions. ")
            f.write("The error distribution reveals the spread of prediction errors.\n")
    
    print(f"Generated results summary at {summary_path}")


def main():
    figures_dir = Path("analysis/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    matches = find_matching_files()
    
    if not matches:
        print("No result files found in results/ directory")
        return
    
    print(f"Found {len(matches)} result file(s) to process")
    
    all_results = []
    
    for train_file, sol_file, base_name in matches:
        print(f"\nProcessing: {base_name}")
        
        try:
            parsed_info = parse_filename(base_name)
            title = format_title(parsed_info)
            
            train_data = load_training_log(train_file)
            final_loss = train_data['loss'][-1] if len(train_data['loss']) > 0 else None
            
            plot_a_path = figures_dir / f"{base_name}_training_curve.png"
            plot_training_curve(
                train_data['episode'],
                train_data['loss'],
                train_data['avg_loss'],
                plot_a_path,
                title_suffix=f" - {title}"
            )
            
            result_info = {
                'base_name': base_name,
                'title': title,
                'training_loss': final_loss,
                'plot_training': plot_a_path
            }
            
            if sol_file is not None:
                sol_data = load_solutions(sol_file)
                plot_b_path = figures_dir / f"{base_name}_prediction_quality.png"
                
                mae = np.mean(sol_data['abs_error'])
                rmse = np.sqrt(np.mean((sol_data['predicted'] - sol_data['true']) ** 2))
                r2 = 1 - np.sum((sol_data['true'] - sol_data['predicted']) ** 2) / np.sum((sol_data['true'] - np.mean(sol_data['true'])) ** 2)
                
                plot_prediction_quality(
                    sol_data['true'],
                    sol_data['predicted'],
                    sol_data['abs_error'],
                    sol_data['log_error'],
                    plot_b_path,
                    title_suffix=f" - {title}"
                )
                
                result_info.update({
                    'plot_prediction': plot_b_path,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                })
            else:
                print(f"  Warning: No solution file found for {base_name}, skipping prediction quality plot")
            
            all_results.append(result_info)
                
        except Exception as e:
            print(f"  Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()
    
    best_training, best_prediction = select_best_plots(all_results)
    generate_results_summary(best_training, best_prediction, figures_dir)
    
    print(f"\nDone! Plots saved to {figures_dir}/")


if __name__ == "__main__":
    main()

