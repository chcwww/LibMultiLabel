"""
python plot.py log_dir
Export training time and memory use of experiments.
(1. Full record for each experiment)
(2. Aggregated full record for linear techniques)
(3. Aggregated training time only record for linear techniques)
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

prefix = "."
if len(sys.argv) > 1:
    prefix = sys.argv[1]
base_dir = os.path.join(prefix, 'para_log')

datasets = ['rcv1', 'EUR-Lex', 'Wiki10-31K', 'AmazonCat-13K']
linear_techs = ['tree', '1vsrest']
branches = ['master', 'no_parallel', 'ovr_thread', 'sep_ovr_thread']
bname = ['liblr-multi', 'liblr-single', 'thread', 'no-prob-thread']
bname_map = {k: v for k, v in zip(branches, bname)}

full_time = pd.read_csv(os.path.join(base_dir, 'extract_time.csv')).set_index('path')

path_list = [os.path.join(prefix, file) for file in os.listdir(prefix) if '.dat' in file]
mem_dict = {t: {d: {b: [] for b in branches} for d in datasets} for t in linear_techs}

for path in path_list:
    with open(path, 'r') as f:
        lines = f.readlines()

    header = lines[0]
    dataset = header.split('/')[1]
    linear_tech = header.split('technique')[1].split()[0]
    branch = header.replace('\n', '').split('--')[-1]
    valid_lines = [line for line in lines if len(line.split())==3]

    mem_use = [float(line.split()[1]) for line in valid_lines]
    init_time = float(valid_lines[0].split()[2])
    time_stamp = [float(line.split()[2]) - init_time for line in valid_lines]
    mem_dict[linear_tech][dataset][branch] = [np.array(mem_use), np.array(time_stamp)]


"""
1.
"""
linestyle = ['-', '--', '-.', ':']
colors = list(mcolors.TABLEAU_COLORS.keys())
for tech in linear_techs:
    os.makedirs(os.path.join(prefix, 'aggr_fig', tech), exist_ok=True)
    for data in datasets:
        local_dict = mem_dict[tech][data]
        kmeans = 0
        for i, b in enumerate(branches):
            log_path = f"{data}--{tech}--{b}.log"
            main_overhead = full_time.loc[log_path].loc[' main'] - full_time.loc[log_path].loc[' linear_run']
            data_overhead = full_time.loc[log_path].loc[' data_overhead']
            training_time = full_time.loc[log_path].loc[' training_time']
            
            plt.plot(local_dict[b][1], local_dict[b][0], 
                     label = b, linestyle = linestyle[i], color = colors[i])
            
            value = main_overhead + data_overhead + training_time
            plt.axvline(x = value,
                        color = colors[i],
                        linewidth = 1,
                        alpha = 0.4)
            
            kmeans += full_time.loc[log_path].loc[' kmeans']
    
        if tech=='tree':
            # use average for better visualize
            plt.axvline(x = kmeans / len(branches), 
                        label = 'cluster',
                        color = colors[7],
                        linewidth = 1.5,
                        alpha = 1)
        plt.title(f'Training Time and Memory Use for {data} with {tech}')
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Memory Use (MB)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(prefix, 'aggr_fig', tech, f'{data}.png'))
        plt.close()

"""
2.
"""
# two figures: tree and 1vsrest
for tech in linear_techs:
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f'Training Time and Memory Use for "{tech}"')
    for idx, data in enumerate(datasets):
        i1, i2 = idx // 2, idx % 2
        local_dict = mem_dict[tech][data]
        for i, b in enumerate(branches):
            log_path = f"{data}--{tech}--{b}.log"
            main_overhead = full_time.loc[log_path].loc[' main'] - full_time.loc[log_path].loc[' linear_run']
            data_overhead = full_time.loc[log_path].loc[' data_overhead']
            training_time = full_time.loc[log_path].loc[' training_time']
            
            axs[i1, i2].plot(local_dict[b][1], local_dict[b][0], 
                     label = b, linestyle = linestyle[i], color = colors[i])
            
            value = main_overhead + data_overhead + training_time
            axs[i1, i2].axvline(x = value,
                        color = colors[i],
                        linewidth = 1,
                        alpha = 0.4)
            
        if tech=='tree':
            kmeans = full_time.loc[log_path].loc[' kmeans']
            axs[i1, i2].axvline(x = kmeans, 
                        label = 'cluster',
                        color = colors[7],
                        linewidth = 1.5,
                        alpha = 1)
            
        axs[i1, i2].set_title(f'{data}')
        if i1 % 2 == 1:
            axs[i1, i2].set_xlabel('Training Time (seconds)')
        if i2 % 2 == 0:
            axs[i1, i2].set_ylabel('Memory Use (MB)')
        axs[i1, i2].grid(True)
    
    axs[0, 1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 0.5), title="Branch")
    fig.tight_layout()
    plt.savefig(os.path.join(prefix, 'aggr_fig', tech, f'aggr_{tech}.png'))
    plt.close()

"""
3.
"""
# two figures: tree and 1vsrest, but with only training part
for tech in linear_techs:
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f'Training Time and Memory Use for "{tech}"')
    for idx, data in enumerate(datasets):
        i1, i2 = idx // 2, idx % 2
        local_dict = mem_dict[tech][data]
        for i, b in enumerate(branches):
            log_path = f"{data}--{tech}--{b}.log"
            main_overhead = full_time.loc[log_path].loc[' main'] - full_time.loc[log_path].loc[' linear_run']
            data_overhead = full_time.loc[log_path].loc[' data_overhead']
            kmeans = full_time.loc[log_path].loc[' kmeans']
            l_train = full_time.loc[log_path].loc[' linear_train']
            total = full_time.loc[log_path].loc[' total']
            training_time = full_time.loc[log_path].loc[' training_time']
            
            x = np.array(local_dict[b][1])
            y = np.array(local_dict[b][0])
            
            start_time = main_overhead + data_overhead + kmeans
            flatten_time = 0 if tech=='1vsrest' else l_train - total
            end_time = main_overhead + data_overhead + training_time - flatten_time
            s_key = np.logical_and(x > start_time, x < end_time)
            axs[i1, i2].plot(x[s_key], y[s_key], alpha = 0.9,
                     label = bname_map[b], linestyle = linestyle[i], color = colors[i])
            if b=='master':
                axs[i1, i2].axhline(y = max(y[s_key]),
                            color = colors[5],
                            linewidth = 1.5,
                            alpha = 0.7)
            
        axs[i1, i2].set_title(f'{data}')
        if i1 % 2 == 1:
            axs[i1, i2].set_xlabel('Training Time (seconds)')
        if i2 % 2 == 0:
            axs[i1, i2].set_ylabel('Memory Use (MB)')
        axs[i1, i2].grid(True)
    
    axs[0, 1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 0.5), title="Branch")
    fig.tight_layout()
    plt.savefig(os.path.join(prefix, 'aggr_fig', tech, f'aggr_train_{tech}.png'), dpi=600)
    plt.close()
