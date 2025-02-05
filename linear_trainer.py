import logging
from math import ceil

import numpy as np
from tqdm import tqdm

import libmultilabel.linear as linear
from libmultilabel.common_utils import dump_log, is_multiclass_dataset, timer
from libmultilabel.linear.utils import LINEAR_TECHNIQUES

@timer
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
def linear_train(datasets, config):
    # detect task type
    multiclass = is_multiclass_dataset(datasets["train"], "y")
    print(f'y: {datasets["train"]["y"].shape}, x: {datasets["train"]["x"].shape}')
    # train
    if config.linear_technique == "tree":
        if multiclass:
            raise ValueError("Tree model should only be used with multilabel datasets.")

        model = LINEAR_TECHNIQUES[config.linear_technique](
            datasets["train"]["y"],
            datasets["train"]["x"],
            options=config.liblinear_options,
            K=config.tree_degree,
            dmax=config.tree_max_depth,
        )
    else:
        model = LINEAR_TECHNIQUES[config.linear_technique](
            datasets["train"]["y"],
            datasets["train"]["x"],
            multiclass=multiclass,
            options=config.liblinear_options,
        )
    return model

@timer
def linear_run(config):
    if config.seed is not None:
        np.random.seed(config.seed)

    if config.eval:
        preprocessor, model = linear.load_pipeline(config.checkpoint_path)
        datasets = linear.load_dataset(config.data_format, config.training_file, config.test_file)
        datasets = preprocessor.transform(datasets)
    else:
        import time
        s = time.time()
        preprocessor = linear.Preprocessor(config.include_test_labels, config.remove_no_label_data)
        datasets = linear.load_dataset(
            config.data_format,
            config.training_file,
            config.test_file,
            config.label_file,
        )
        datasets = preprocessor.fit_transform(datasets)
        print(f"Dataset overhead: {time.time() - s:.4f}")
        # import pickle
        # with open(f"{config.training_file}.pickle", 'rb') as fp:
        #         datasets = pickle.load(fp)
        # with open(f"{config.training_file}.preprocessor.pickle", 'rb') as fp:
        #     preprocessor = pickle.load(fp)
        s = time.time()
        model = linear_train(datasets, config)
        print(f"Training time: {time.time() - s:.4f}")
        try:
            linear.save_pipeline(config.checkpoint_dir, preprocessor, model)
        except Exception as e:
            print(e)
            print("Error when saving pipeline...")

    if config.test_file is not None:
        assert not (
            config.save_positive_predictions and config.save_k_predictions > 0
        ), """
            If save_k_predictions is larger than 0, only top k labels are saved.
            Save all labels with decision value larger than 0 by using save_positive_predictions and save_k_predictions=0."""
        s = time.time()
        metric_dict, labels, scores = linear_test(config, model, datasets, preprocessor.label_mapping)
        dump_log(config=config, metrics=metric_dict, split="test", log_path=config.log_path)
        print(linear.tabulate_metrics(metric_dict, "test"))
        if config.save_k_predictions > 0:
            with open(config.predict_out_path, "w") as fp:
                for label, score in zip(labels, scores):
                    out_str = " ".join([f"{i}:{s:.4}" for i, s in zip(label, score)])
                    fp.write(out_str + "\n")
            logging.info(f"Saved predictions to: {config.predict_out_path}")
        elif config.save_positive_predictions:
            with open(config.predict_out_path, "w") as fp:
                for batch_labels, batch_scores in zip(labels, scores):
                    for label, score in zip(batch_labels, batch_scores):
                        out_str = " ".join([f"{i}:{s:.4}" for i, s in zip(label, score)])
                        fp.write(out_str + "\n")
            logging.info(f"Saved predictions to: {config.predict_out_path}")
        print(f"Test time: {time.time() - s:.4f}")
