import os
import sys
import warnings
warnings.filterwarnings("ignore")
if len(sys.argv) != 3:
    raise RuntimeError(f'type "python {sys.argv[0]} log_dir dataset_name"')

import libmultilabel.linear as linear
from itertools import combinations
import numpy as np

train_type = ['master', 'no_parallel', 'ovr_thread', 'sep_ovr_thread']
compare_combs = combinations(train_type, 2)

err_str = ""
for compare_comb in compare_combs:
    try:  
        models = [linear.load_pipeline(
            os.path.join(sys.argv[1], 'runs', 'tree', f"{sys.argv[2]}--tree--{comb}", "linear_pipeline.pickle"
                        ))[1] for comb in compare_comb]
    except FileNotFoundError as e:
        err_str += f"At least one of {compare_comb} not exist.\n"
        continue

    if models[0].name=='tree':
        models[0].weights = models[0].flat_model.weights.toarray()
        models[1].weights = models[1].flat_model.weights.toarray()
        
    if models[0].weights.shape != models[1].weights.shape:
        raise NotImplementedError("They aren't trained with the same dataset..")

    print(f'{str(compare_comb):^33} Weights Diff: {abs((models[0].weights-models[1].weights).sum()):.8f}, nnz: ({np.count_nonzero(models[0].weights)}, {np.count_nonzero(models[1].weights)})')

print(err_str)

# python compare_results.py log_dir dataset