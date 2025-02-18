import os
import threading
from queue import Queue
from tqdm import tqdm

import numpy as np
import scipy.sparse as sparse
from liblinear.liblinearutil import train, parameter, problem

class ParallelTrainer(threading.Thread):
    """A trainer for parallel 1vsrest training."""
    y: sparse.csc_matrix
    x: sparse.csr_matrix
    param: parameter
    weights: np.ndarray
    pbar: tqdm
    queue: Queue
    lock: threading.Lock

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
            y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
            x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
            options (str): The option string passed to liblinear.
            num_class (int): Number of class.
            weights (np.ndarray): the weights.
            verbose (bool): Output extra progress information.
        """
        cls.x = x
        cls.y = y
        cls.param = parameter(options)
        cls.param.w_recalc = True   # only works for solving L1/L2-SVM dual
        cls.weights = weights
        cls.pbar= tqdm(total=num_class, disable=not verbose)
        cls.queue = Queue()
        cls.lock = threading.Lock()

        for i in range(num_class):
            cls.queue.put(i)

    @staticmethod
    def _do_parallel_train(prob: problem, param: parameter) -> np.matrix:
        """Wrapper around liblinear.liblinearutil.train.

        Args:
            prob (problem): A preprocessed liblinear.problem instance.
            param (parameter): A preprocessed liblinear.parameter instance.

        Returns:
            np.matrix: the weights.
        """
        if prob.l == 0:
            return np.matrix(np.zeros((prob.n, 1)))

        model = train(prob, param)

        w = np.ctypeslib.as_array(model.w, (prob.n, 1))
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
            weight = self._do_parallel_train(problem(2 * yi - 1, self.x), self.param).ravel()

            self.lock.acquire()
            self.weights[:, label_idx] = weight
            self.pbar.update()
            self.lock.release()

def train_parallel_1vsrest(
        y: sparse.csc_matrix,
        x: sparse.csr_matrix,
        options: str,
        num_class: int,
        weights: np.ndarray,
        verbose: bool,
    ):
    """Parallel training on labels when using one-vs-rest strategy,
    and saving trained weights by reference.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        num_class (int): Number of class.
        weights (np.ndarray): the weights.
        verbose (bool): Output extra progress information.
    """
    ParallelTrainer.init_trainer(y, x, options, num_class, weights, verbose)
    num_thread = int(os.cpu_count() / 2)
    # stderr = os.dup(2)
    trainers = [ParallelTrainer() for _ in range(num_thread)]

    for trainer in trainers:
        trainer.start()
    for trainer in trainers:
        trainer.join()

    # os.dup2(stderr, 2)
    # os.close(stderr)
    ParallelTrainer.pbar.close()
