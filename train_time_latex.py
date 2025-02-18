"""
python train_time_latex.py log_dir
Output latex code string of tables.
(1. trianing time)
(2. ovr performance difference)
(3. tree performance difference)
"""
import os
import sys
import pandas as pd

prefix = "."
if len(sys.argv) > 1:
    prefix = sys.argv[1]
base_dir = os.path.join(prefix, 'para_log')

datasets = ['rcv1', 'EUR-Lex', 'Wiki10-31K', 'AmazonCat-13K']
linear_techs = ['1vsrest', 'tree']
branches = ['master', 'no_parallel', 'ovr_thread', 'sep_ovr_thread']

bname = ['liblr-multi', 'liblr-single', 'thread', 'no-prob-thread']
bname_map = {k: v for k, v in zip(branches, bname)}

full_time = pd.read_csv(os.path.join(base_dir, 'extract_time.csv')).set_index('path')

header_template = lambda s: f"{{\\sf {s}}}"

latex_list = []
for branch in branches:
    inner_list = [header_template(bname_map[branch])]
    for data in datasets:
        for tech in linear_techs:
            log_path = f"{data}--{tech}--{branch}.log"
            key = " training_time" if tech == '1vsrest' else " training" # consider kmeans overhead or not
            value = full_time.loc[log_path].loc[key]
            inner_list.append(f'{value:.2f}')
    latex_list.append(' & '.join(inner_list) + ' \\\\')

"""
1.
"""
print('\n'.join(latex_list))


import json
perform = {k: None for k in datasets}
for data in datasets:
    with open(os.path.join(prefix, f'{data}_eval.json'), 'r') as f:
        perform[data] = json.load(f)
        
def diff_template(flo):
    color = lambda s, f: f'\\textcolor{{{s}}}{{{f}}}'
    sign = '+'
    col = 'green'
    if flo < 0:
        sign = ''
        col = 'red'
    return color(col, f'{sign}{flo:.6f}')    
        
latex_list = []
for tech in linear_techs:
    outer_list = []
    for data in datasets:
        outer_list.append(f'\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{{data}}}}}\\\\ \\hline \\hline')
        buffer = None
        for branch in branches:
            inner_list = [header_template(bname_map[branch])]
            log_path = f"{data}--{tech}--{branch}.log"
            value_dict = perform[data][tech][branch]
            if branch == 'master':
                buffer = list(value_dict.values())
                inner_list += [f'{v:.7f}' for v in buffer]
            else:
                inner_list += [f'{diff_template(v - buffer[i])}' for i, v in enumerate(value_dict.values())]
                
            outer_list.append(' & '.join(inner_list) + '\\\\')
            if branch=='master':
                outer_list.append('\\hline')
    latex_list.append('\n'.join(outer_list))

"""
2.
"""
print(latex_list[0])
"""
3.
"""
print(latex_list[1])