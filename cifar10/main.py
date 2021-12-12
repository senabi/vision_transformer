from datetime import datetime
from pathlib import Path

import fire
import ignite
import ignite.distributed as idist
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from ignite.contrib.engines import common
from ignite.contrib.handlers import PiecewiseLinear
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed, setup_logger
from torch.cuda.amp import GradScaler, autocast
import gc


def training(local_rank, config):

    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name="CIFAR10-Training", distributed_rank=local_rank)

    log_basic_info(logger, config)

    output_path = config["output_path"]
    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")

        folder_name = f"{config['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)
        config["output_path"] = output_path.as_posix()
        logger.info(f"Output path: {config['output_path']}")

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)

        if config["with_clearml"]:
            try:
                from clearml import Task
            except ImportError:
                # Backwards-compatibility for legacy Trains SDK
                from trains import Task

            task = Task.init("CIFAR10-Training", task_name=output_path.stem)
            task.connect_configuration(config)
            task.connect(config)

    # Setup dataflow, model, optimizer, criterion
    train_loader = None
    test_loader = None
    model = None
    torch.cuda.empty_cache()
    gc.collect()
    train_loader, test_loader = get_dataflow(config)

    config["num_iters_per_epoch"] = len(train_loader)
    model, optimizer, criterion, lr_scheduler = initialize(config)

    logger.info(
        f"# model parameters (M): {sum([m.numel() for m in model.parameters()]) * 1e-6}"
    )

    # Create trainer for current task
    trainer = create_trainer(
        model, optimizer, criterion, lr_scheduler, train_loader.sampler, config, logger
    )

    # Let's now setup evaluator engine to perform model's validation and
    # compute metrics
    metrics = {
        "Accuracy": Accuracy(),
        "Loss": Loss(criterion),
    }

    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_evaluator(model, metrics=metrics, config=config)
    train_evaluator = create_evaluator(model, metrics=metrics, config=config)

    if config["smoke_test"]:
        logger.info(
            "Reduce the size of training and test dataloader as smoke_test=True"
        )

        def get_batches(loader):
            loader_iter = iter(loader)
            return [next(loader_iter) for _ in range(5)]

        train_loader = get_batches(train_loader)
        test_loader = get_batches(test_loader)

    if config["with_pbar"] and rank == 0:
        ProgressBar(desc="Evaluation (train)", persist=False).attach(train_evaluator)
        ProgressBar(desc="Evaluation (val)", persist=False).attach(evaluator)

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Train", state.metrics)
        state = evaluator.run(test_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED,
        run_validation,
    )

    if rank == 0:
        # Setup TensorBoard logging on trainer and evaluators. Logged values are:
        #  - Training metrics, e.g. running average loss values
        #  - Learning rate
        #  - Evaluation train/test metrics
        evaluators = {"training": train_evaluator, "test": evaluator}
        tb_logger = common.setup_tb_logging(
            output_path, trainer, optimizer, evaluators=evaluators
        )

    # Store 1 best models by validation accuracy starting from num_epochs / 2:
    best_model_handler = Checkpoint(
        {"model": model},
        get_save_handler(config),
        filename_prefix="best",
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
        score_name="test_accuracy",
        score_function=Checkpoint.get_default_score_fn("Accuracy"),
    )
    evaluator.add_event_handler(
        Events.COMPLETED(lambda *_: trainer.state.epoch > config["num_epochs"] // 2),
        best_model_handler,
    )

    try:
        trainer.run(train_loader, max_epochs=config["num_epochs"])
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()


def run(
    seed=543,
    data_path="/tmp/cifar10",
    output_path="/tmp/output-cifar10/",
    model="vit_tiny_patch4_32x32",
    rescale_size=None,
    rand_aug=None,
    rand_erasing=None,
    optimizer="adam",
    batch_size=128,
    weight_decay=1e-4,
    num_workers=4,
    num_epochs=200,
    learning_rate=0.001,
    num_warmup_epochs=0,
    validate_every=3,
    checkpoint_every=1000,
    backend=None,
    resume_from=None,
    nproc_per_node=None,
    with_pbar=False,
    with_amp=False,
    cutmix_beta=0.0,
    cutmix_prob=0.5,
    rescaled_size=None,
    with_clearml=False,
    smoke_test=False,
    dataset_name="CIFAR10",
    num_classes=10,
    **spawn_kwargs,
):
    """Main entry to train an model on CIFAR10 dataset.

    Args:
        seed (int): random state seed to set.
        data_path (str): input dataset path. Default, "/tmp/cifar10".
        output_path (str): output path. Default, "/tmp/output-cifar10".
        model (str): model name (from torchvision) to setup model to train.
        batch_size (int): total batch size.
        optimizer (str): optimizer name. Possible values: "sgd", "adam", "adamw". Default, "adam".
        weight_decay (float): weight decay.
        num_workers (int): number of workers in the data loader.
        num_epochs (int): number of epochs to train the model.
        learning_rate (float): peak of piecewise linear learning rate scheduler.
        num_warmup_epochs (int): number of warm-up epochs before learning rate decay.
        validate_every (int): run model's validation every ``validate_every`` epochs.
        checkpoint_every (int): store training checkpoint every ``checkpoint_every`` iterations.
        backend (str, optional): backend to use for distributed configuration. Possible values: None, "nccl",
            "gloo" etc.
        nproc_per_node (int, optional): optional argument to setup number of processes per node. It is useful,
            when main python process is spawning training as child processes.
        resume_from (str, optional): path to checkpoint to use to resume the training from.
        with_pbar(bool): if True adds a progress bar on training iterations.
        with_amp(bool): if True uses torch native AMP
        rescale_size (int, optional): if provided then input image will be rescaled to that value.
        cutmix_beta : beta value for the distribution of the cutmix
        cutmix_prob : cutmix probablity
        with_clearml (bool): if True, experiment ClearML logger is setup.
        smoke_test (bool): run 5 iters and quit
        dataset_name (str, optional): name of the dataset
        num_classes (int): number of classes
        **spawn_kwargs: Other kwargs to spawn run in child processes: master_addr, master_port, node_rank, nnodes

    """
    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:

        parallel.run(training, config)


def get_dataflow(config):
    # - Get train/test datasets
    if idist.get_rank() > 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    train_dataset, test_dataset = utils.get_train_test_datasets(
        config["data_path"],
        **{
            k: config[k]
            for k in [
                "rescale_size",
                "rand_aug",
                "rand_erasing",
                "dataset_name",
                "model",
            ]
        },
    )

    if idist.get_rank() == 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
    )

    test_loader = idist.auto_dataloader(
        test_dataset,
        batch_size=2 * config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )
    return train_loader, test_loader


def initialize(config):
    model = utils.get_model(config["model"], config["num_classes"])
    # Adapt model for distributed backend if provided
    model = idist.auto_model(model)

    optimizer = utils.get_optimizer(
        config["optimizer"],
        model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    # Adapt optimizer for distributed backend if provided
    optimizer = idist.auto_optim(optimizer)
    criterion = nn.CrossEntropyLoss().to(idist.device())

    le = config["num_iters_per_epoch"]
    milestones_values = [
        (0, 0.0),
        (le * config["num_warmup_epochs"], config["learning_rate"]),
        (le * config["num_epochs"], 0.0),
    ]
    lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
    )

    return model, optimizer, criterion, lr_scheduler


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


def log_basic_info(logger, config):
    logger.info(f"Train {config['model']} on CIFAR10")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(
            f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}"
        )
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def create_trainer(
    model, optimizer, criterion, lr_scheduler, train_sampler, config, logger
):

    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    cutmix_beta = config["cutmix_beta"]
    cutmix_prob = config["cutmix_prob"]
    with_amp = config["with_amp"]
    scaler = GradScaler(enabled=with_amp)

    def train_step(engine, batch):

        x, y = batch[0], batch[1]

        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        model.train()

        with autocast(enabled=with_amp):
            r = torch.rand(1).item()
            if cutmix_beta > 0 and r < cutmix_prob:
                output, loss = utils.cutmix_forward(model, x, criterion, y, cutmix_beta)
            else:
                output = model(x)
                loss = criterion(output, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if idist.backend() == "horovod":
            optimizer.synchronize()
            with optimizer.skip_synchronize():
                scaler.step(optimizer)
                scaler.update()
        else:
            scaler.step(optimizer)
            scaler.update()

        return {
            "batch loss": loss.item(),
        }

    trainer = Engine(train_step)
    trainer.logger = logger

    if config["with_pbar"] and idist.get_rank() == 0:
        ProgressBar().attach(trainer)

    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    metric_names = [
        "batch loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=metric_names,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    resume_from = config["resume_from"]
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert (
            checkpoint_fp.exists()
        ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def create_evaluator(model, metrics, config, tag="val"):
    with_amp = config["with_amp"]
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, batch):
        model.eval()
        x, y = batch[0], batch[1]
        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        with autocast(enabled=with_amp):
            output = model(x)
        return output, y

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    if idist.get_rank() == 0 and (not config["with_clearml"]):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator


def get_save_handler(config):
    if config["with_clearml"]:
        from ignite.contrib.handlers.clearml_logger import ClearMLSaver

        return ClearMLSaver(dirname=config["output_path"])

    return DiskSaver(config["output_path"], require_empty=False)


if __name__ == "__main__":
    fire.Fire({"run": run})
