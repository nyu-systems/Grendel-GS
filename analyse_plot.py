import matplotlib.pyplot as plt
import re

import scene

def parse_data(filename):
    """
    Parse the provided text data and extract metrics for each iteration and parameter, organized by batch size.
    """
    data = {}
    current_batch_size = None
    current_iteration = None
    
    # Iterate through each line to extract data
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if "grad stats for batch_size:" in line:
            current_batch_size = int(line.split(': ')[-1].strip())
            if current_batch_size not in data:
                data[current_batch_size] = {}
        elif line.startswith('iteration:'):
            try:
                current_iteration = int(line.split(': ')[1].strip())
            except ValueError:
                continue
            if current_iteration not in data[current_batch_size]:
                data[current_batch_size][current_iteration] = {}
        else:
            # Extract parameter name and metrics
            param_match = re.match(r'param (\w+): (.+)', line)
            if param_match:
                param_name = param_match.group(1)
                metrics_text = param_match.group(2)
                metrics = {}
                # Extract each metric and its value
                for metric in metrics_text.split(', '):
                    key, value = metric.split(': ')
                    metrics[key] = float(value) if 'nan' not in value else None
                data[current_batch_size][current_iteration][param_name] = metrics
    return data

def plot_data(data, batch_size, metric_name, scene_name, start_iter):
    """
    Plot a specific metric for all parameters across iterations for a given batch size.
    """
    plt.figure(figsize=(20, 8))
    batch_data = data.get(batch_size, {})
    for param in set(k for d in batch_data.values() for k in d.keys()):
        iterations = sorted(batch_data.keys())
        values = [batch_data[it][param][metric_name] if param in batch_data[it] else None for it in iterations]
        
        if any(v is not None for v in values):  # Only plot if there's data
            plt.plot(iterations, values, label=param)

    plt.title(f'{metric_name} over Iterations for Batch Size {batch_size}')
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f"experiments_analyse/{scene_name}_start_{start_iter}_cont_100/{metric_name}_batch_{batch_size}.png")
    plt.close()

# Usage
filename_fmt = "experiments_analyse/{scene_name}_start_{start_iter}_cont_100/python_ws=4_rk=0.log"

start_iters  = [100, 500, 1000, 2000, 3000, 5000, 9000, 10000, 15000, 20000, 25000, 30000]
scene_names = ["train", "garden"]

for scene_name in scene_names:
    for start_iter in start_iters:
        filename = filename_fmt.format(scene_name=scene_name, start_iter=start_iter)
        print(f"Analyzing {filename}")
        data = parse_data(filename)
        metrics = ['exp_avg_cosine_similarity', 'weights_delta_cosine_similarity', 'weights_delta_norm_ratio']
        batch_sizes = [2, 4, 8, 16]
        for metric in metrics:
            for batch_size in batch_sizes:
                plot_data(data, batch_size, metric, scene_name, start_iter)
