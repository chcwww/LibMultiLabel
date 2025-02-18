"""
python extract_time.py log_dir
Export needed time records from log_dir/para_log to log_dir/para_log/extract_time.csv  
"""
import os
import sys

prefix = "."
if len(sys.argv) > 1:
    prefix = sys.argv[1]
base_dir = os.path.join(prefix, 'para_log')

datasets = ['rcv1', 'EUR-Lex', 'Wiki10-31K', 'AmazonCat-13K']
linear_techs = ['tree', '1vsrest']
branches = ['master', 'no_parallel', 'ovr_thread', 'sep_ovr_thread']

tech_path = {tech: [f"{dataset}--{tech}--{branch}.log" for dataset in datasets for branch in branches] for tech in linear_techs}



header = "path, training_time, linear_train, linear_run, main, data_overhead, kmeans, training, total"
result_list = [header]

for tech in linear_techs:
    for path in tech_path[tech]:
        with open(os.path.join(base_dir, tech, path), 'r') as f:
            lines = f.readlines()
        
        offset = 1 if tech=='tree' else 0
            
        training_time = lines[-3].split()[-1]
        train = lines[-6 - offset].split()[-2]
        run = lines[-5 - offset].split()[-2]
        main = lines[-4 - offset].split()[-2]
        data_over_idx = 1
        while('Dataset' not in lines[data_over_idx]):
            data_over_idx += 1
        data_overhead = lines[data_over_idx].split()[-1]
        
        kmeans = i_train = i_total = '0'
        if offset:
            km, tr, to = lines[-4].split(',')
            kmeans = km.split()[-1]
            i_train = tr.split()[-1]
            i_total = to.split()[-1]
        
        local_list = [training_time, train, run, main, data_overhead, kmeans, i_train, i_total]
        result_list += [', '.join([path] + local_list)]
        
with open(os.path.join(base_dir, 'extract_time.csv'), 'w') as f:
    f.write('\n'.join(result_list))