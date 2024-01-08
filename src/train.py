import time
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import Logger
import tqdm
from src.logger.jam_wandb import JamWandb
from src.models.base_model import BaseModel
from src.utils import lht_utils
import numpy as np
import torch as th

from src.utils.loss_helper import loss2logz_info

from src.utils.sampling import generate_samples_loss


# from lightning import LightningModule, Callback, LightningDataModule, Trainer, seed_everything

try:
    from jammy.utils.debug import decorate_exception_hook
except ImportError:
    # pylint: disable=ungrouped-imports
    from src.utils.lht_utils import decorate_exception_hook
log = lht_utils.get_logger(__name__)


@decorate_exception_hook
def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    device = 'cuda' if th.cuda.is_available() else 'cpu'

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(
        f"Instantiating datamodule <{config.datamodule.module._target_}>"  # pylint: disable=protected-access
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule.module, config.datamodule, device
    )

    # Init lightning model
    log.info(
        f"Instantiating model <{config.model.module._target_}>"  # pylint: disable=protected-access
    )
    model: LightningModule = hydra.utils.instantiate(config.model.module, config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(
                    f"Instantiating callback <{cb_conf._target_}>"  # pylint: disable=protected-access
                )
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(
                    f"Instantiating logger <{lg_conf._target_}>"  # pylint: disable=protected-access
                )
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(
        f"Instantiating trainer <{config.trainer._target_}>"  # pylint: disable=protected-access
    )
    lht_utils.auto_gpu(config)
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    JamWandb.g_cfg = config
    lht_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    for lg in logger:
        lg.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    # reseed before training, encounter once after instantiation, randomness disappear
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    # Train the model
    log.info("Starting training!")
    t0 = time.time()
    start_density_calls = datamodule.dataset.density_calls
    trainer.fit(model=model, datamodule=datamodule)
    t1 = time.time()
    log.info(f"Total training time: {t1-t0}")
    end_density_calls = datamodule.dataset.density_calls
    log.info(f"Total density calls: {end_density_calls - start_density_calls}")
    for lg in logger:
        lg.log_metrics({'training_time': t1-t0, "density_calls": end_density_calls - start_density_calls}, step=0)
    
    # Test the model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        # trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

        best_model = BaseModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        target = hydra.utils.instantiate(config.datamodule.dataset, device=best_model.device)
        best_model.sde_model.grad_fn = target.score
        best_model.nll_target_fn = target.energy
        best_model.nll_prior_fn = best_model.sde_model.nll_prior

        best_model.eval()
        
        log_Z = np.zeros(config.num_smc_iters)
        start_time = time.time()
        for i in tqdm.tqdm(range(config.num_smc_iters)):
            _, _, loss, _ = generate_samples_loss(
                best_model.sde_model,
                best_model.nll_target_fn,
                best_model.nll_prior_fn,
                best_model.dt,
                best_model.t_end,
                2000,
                device=best_model.device,
            )
            logz_info = loss2logz_info(loss)
            log_Z[i] = logz_info["logz/loss_unbiased"]
        end_time = time.time()
        # Save normalising constant estimates (comment out when not doing a final evaluation run)
        if config.save_samples:
            np.savetxt(
                f"/data/ziz/not-backed-up/anphilli/diffusion_smc/benchmarking_data/{config.group}_{config.name}_pis_{config.num_steps}_{config.seed}.csv",
                log_Z,
            )
        # Log normalising constant estimates
        if logger:
            for lg in logger:
                lg.log_metrics(
                    {"sampling_time": (end_time - start_time) / config.num_smc_iters},
                    step=0,
                )
                lg.log_metrics(
                    {"final_log_Z": np.mean(log_Z), "var_final_log_Z": np.var(log_Z)}, 0
                )

    # Make sure everything closed properly
    log.info("Finalizing!")
    lht_utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if (
        not config.trainer.get("fast_dev_run")
        and trainer.checkpoint_callback is not None
    ):
        log.info(f"Best model ckpt: {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
    return None
