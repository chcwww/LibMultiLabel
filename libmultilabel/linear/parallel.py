from __future__ import annotations

import os
import threading
from queue import SimpleQueue
from tqdm import tqdm

import numpy as np
import scipy.sparse as sparse
from liblinear.liblinearutil import train, parameter, problem, solver_names

from ctypes import c_double

NO_COPY = int(os.environ.get("NO_COPY", "0"))

class ParallelTrainer(threading.Thread):
    """A trainer for parallel 1vsrest training."""
    y: sparse.csc_matrix
    x: sparse.csr_matrix
    prob: problem
    param: parameter
    weights: np.ndarray
    pbar: tqdm
    queue: SimpleQueue

    def __init__(self):
        threading.Thread.__init__(self)

    @classmethod
    def init_trainer(
        cls,
        y: sparse.csc_matrix,
        x: sparse.csr_matrix,
        options: str,
        num_class: int,
        weights: np.ndarray,
        verbose: bool,
    ):
        """Initialize the parallel trainer by setting y, x, parameter and threading related
        variables as class variable of ParallelTrainer.

        Args:
            y (sparse.csc_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
            x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
            options (str): The option string passed to liblinear.
            num_class (int): Number of class.
            weights (np.ndarray): the weights.
            verbose (bool): Output extra progress information.
        """
        cls.x = x
        cls.y = y
        cls.prob = problem(np.ones((y.shape[0],)), x)
        cls.param = parameter(options)
        if cls.param.solver_type in [solver_names.L2R_L1LOSS_SVC_DUAL, solver_names.L2R_L2LOSS_SVC_DUAL]:
            cls.param.w_recalc = True   # only works for solving L1/L2-SVM dual
        cls.weights = weights
        cls.pbar= tqdm(total=num_class, disable=not verbose)
        cls.queue = SimpleQueue()

        for i in range(num_class):
            cls.queue.put(i)

    def _do_parallel_train(self, y: np.ndarray) -> np.matrix:
        """Wrapper around liblinear.liblinearutil.train.

        Args:
            y (np.ndarray): A +1/-1 array with dimensions number of instances * 1.

        Returns:
            np.matrix: the weights.
        """
        if y.shape[0] == 0:
            return np.matrix(np.zeros((self.prob.n, 1)))

        if NO_COPY:
            prob = problem(y, self.x)
        else:
            prob = self.prob.copy()
            prob.y = (c_double * prob.l)(*y)
        model = train(prob, self.param)

        w = np.ctypeslib.as_array(model.w, (self.prob.n, 1))
        w = np.asmatrix(w)
        # Liblinear flips +1/-1 labels so +1 is always the first label,
        # but not if all labels are -1.
        # For our usage, we need +1 to always be the first label,
        # so the check is necessary.
        if model.get_labels()[0] == -1:
            return -w
        else:
            # The memory is freed on model deletion so we make a copy.
            return w.copy()

    def run(self):
        while self.queue.qsize() > 0:
            label_idx = self.queue.get()

            yi = self.y[:, label_idx].toarray().reshape(-1)
            self.weights[:, label_idx] = self._do_parallel_train(2 * yi - 1).ravel()

            self.pbar.update()

def train_parallel_1vsrest(
        y: sparse.csc_matrix,
        x: sparse.csr_matrix,
        options: str,
        num_class: int,
        weights: np.ndarray,
        num_threads: int,
        verbose: bool,
    ):
    """Parallel training on labels when using one-vs-rest strategy.

    Args:
        y (sparse.csc_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        num_class (int): Number of class.
        weights (np.ndarray): the weights.
        verbose (bool): Output extra progress information.
    """
    ParallelTrainer.init_trainer(y, x, options, num_class, weights, verbose)
    if num_threads < 0 or num_threads > os.cpu_count():
        num_threads = int(os.cpu_count() / 2)
    trainers = [ParallelTrainer() for _ in range(num_threads)]

    for trainer in trainers:
        trainer.start()
    for trainer in trainers:
        trainer.join()

    ParallelTrainer.pbar.close()
    return weights
