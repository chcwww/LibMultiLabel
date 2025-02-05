import logging
from math import ceil

import numpy as np
from tqdm import tqdm
import time

import libmultilabel.linear as linear
from libmultilabel.common_utils import dump_log, is_multiclass_dataset, timer
from libmultilabel.linear.utils import LINEAR_TECHNIQUES


def linear_test(config, model, datasets, label_mapping):
    metrics = linear.get_metrics(config.monitor_metrics, datasets["test"]["y"].shape[1], multiclass=model.multiclass)
    num_instance = datasets["test"]["x"].shape[0]
    k = config.save_k_predictions
    if k > 0:
        labels = np.zeros((num_instance, k), dtype=object)
        scores = np.zeros((num_instance, k), dtype="d")
    else:
        labels = []
        scores = []

    predict_kwargs = {}
    if model.name == "tree":
        predict_kwargs["beam_width"] = config.beam_width

    for i in tqdm(range(ceil(num_instance / config.eval_batch_size))):
        slice = np.s_[i * config.eval_batch_size : (i + 1) * config.eval_batch_size]
        preds = model.predict_values(datasets["test"]["x"][slice], **predict_kwargs)
        target = datasets["test"]["y"][slice].toarray()
        metrics.update(preds, target)
        if k > 0:
            labels[slice], scores[slice] = linear.get_topk_labels(preds, label_mapping, config.save_k_predictions)
        elif config.save_positive_predictions:
            res = linear.get_positive_labels(preds, label_mapping)
            labels.append(res[0])
            scores.append(res[1])
    metric_dict = metrics.compute()
    return metric_dict, labels, scores

@timer
def linear_run(config):
    import pickle
    s = time.time()
    preprocessor = linear.Preprocessor(config.include_test_labels, config.remove_no_label_data)
    datasets = linear.load_dataset(
        config.data_format,
        config.training_file,
        config.test_file,
        config.label_file,
    )
    datasets = preprocessor.fit_transform(datasets)
    print(f'Dataset loading overhead: {time.time() - s:.4f}')
    train_type = ['master', 'no_parallel', 'ovr_thread', 'sep_ovr_thread']
    linear_techs = ['tree', '1vsrest']
    result_dict = {t: {b: None for b in train_type} for t in linear_techs}
    err_str = ""
    try:
        for t in linear_techs:
            for b in train_type:
                try:
                    import os
                    model = linear.load_pipeline(
                        os.path.join(config.result_dir, 'runs', t, 
                                    f"{config.data_name}--{t}--{b}", "linear_pipeline.pickle")
                        )[1]
                except FileNotFoundError as e:
                    err_str += f"Model of {f'{config.data_name}--{t}--{b}'} not exist.\n"
                    continue

                metric_dict, labels, scores = linear_test(config, model, datasets, preprocessor.label_mapping)

                result_dict[t][b] = metric_dict
                print(linear.tabulate_metrics(metric_dict, "test"))
    finally:
        import json
        with open(os.path.join(config.result_dir, f'{config.data_name}_eval.json'), 'w') as fp:
            json.dump(result_dict, fp, indent=4)

if __name__ == "__main__":
    linear_run()