import os
import sys
import time
import math
import torch
import torch.nn as nn
import datetime
from typing import Iterable
from typing import List, Union
from contextlib import ExitStack, contextmanager
from collections import OrderedDict, abc
from utils.misc import get_world_size, is_main_process
import logging

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, args=None, logger=None):

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    model.train()
    logger.info('Current param_groups[0].lr = {}'.format(optimizer.param_groups[0]['lr']))
    for step, batch_sample in enumerate(data_loader):

        with torch.cuda.amp.autocast(enabled=args.amp):
            if args.use_copy_paste:
                loss_dict = model(batch_sample, use_copy_paste=True, cp_threshold=args.cp_threshold)
            else:
                loss_dict = model(batch_sample, use_copy_paste=False, cp_threshold=-1)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not math.isfinite(losses.item()):
            logger.info("Loss is {}, stopping training".format(losses))
            logger.info('loss_dict = {}'.format(loss_dict))
            sys.exit(1)

        optimizer.zero_grad()
        if args.amp:
            scaler.scale(losses).backward()
            if args.clip_grad_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.clip_grad_params_max_norm,
                    norm_type=args.clip_grad_params_norm_type
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if args.clip_grad_enabled:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.clip_grad_params_max_norm,
                    norm_type=args.clip_grad_params_norm_type
                )
            optimizer.step()

        if step % args.print_fre == 0:
            metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
            logger.info(
                'epoch={}  step={}/{}  loss={}  loss_dict={}'.format(
                    epoch,
                    step,
                    len(data_loader),
                    losses.item(),
                    metrics_dict)
            )


def inference_on_dataset(model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], logger=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """

    import warnings
    warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

    num_devices = get_world_size()
    logger.info("Start inference, total {} batches".format(len(data_loader)))
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        logger.info('Evaluate template: model.inference_template = {}'.format(model.module.inference_template))
    else:
        logger.info('Evaluate template: model.inference_template = {}'.format(model.inference_template))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    start_data_time = time.perf_counter()
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs, use_copy_paste=False, cp_threshold=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx != 0 and idx % 100 == 0:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                logger.info(
                    f"Inference done {idx}/{total}  "
                    f"Dataloading: {data_seconds_per_iter:.4f} s/iter  "
                    f"Inference: {compute_seconds_per_iter:.4f} s/iter  "
                    f"Eval: {eval_seconds_per_iter:.4f} s/iter "
                    f"Total: {total_seconds_per_iter:.4f} s/iter "
                    f"ETA={eta}"
                )

            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}

    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def print_csv_format(results, logger=None):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    from collections.abc import Mapping
    assert isinstance(results, Mapping) or not len(results), results
    if logger == None:
        logger = logging.getLogger(__name__)
    current_map = -1.0
    for task, res in results.items():
        if isinstance(res, Mapping):
            # # Don't print "AP-category" metrics since they are usually not tracked.
            # important_res = [(k, v) for k, v in res.items() if "-" not in k]
            # logger.info("copypaste: Task: {}".format(task))
            # logger.info("copypaste: " + ",      ".join([k[0] for k in important_res]))
            # logger.info("copypaste: " + "  ".join(["{0:.4f}".format(k[1]) for k in important_res]))
            # logger.info("\n")
            logger.info("copypaste: Task: {}".format(task))
            logger.info('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {}'.format(round(res['AP']*0.01, 5)))
            logger.info('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {}'.format(round(res['AP50']*0.01, 5)))
            logger.info('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {}'.format(round(res['AP75']*0.01, 5)))
            logger.info('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {}'.format(round(res['APs']*0.01, 5)))
            logger.info('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {}'.format(round(res['APm']*0.01, 5)))
            logger.info('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {}'.format(round(res['APl']*0.01, 5)))
            logger.info("\n")
            current_map = res['AP']
        else:
            logger.info(f"copypaste: {task}={res}")

    return current_map
