import torch
from lightning.pytorch import seed_everything

from hydra.utils import instantiate
from omegaconf import OmegaConf

from cybulde.config_schemas.config_schema import Config
from cybulde.utils.config_utils import get_config
from cybulde.utils.utils import get_logger
from cybulde.utils.torch_utils import get_local_rank

@get_config(config_path="../configs/automatically_generated", config_name="config", to_object=False, return_dict_config=True)
def run_tasks(config: Config) -> None:
    logger = get_logger(__file__)
    assert config.infrastructure.mlflow.run_id is not None, "Run id has to be set for running tasks"

    #lookup torchrun for more info here
    backend="gloo" #lib for making cpus communicate
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        backend = "nccl" #lib for making gpus communicate

    torch.distributed.init_process_group(backend = backend)

    seed_everything(seed=config.seed, workers=True)

    for task_name, task_config in config.tasks.items():
        logger.info(f"Running task: {task_name}")
        task = instantiate(task_config)
        task.run(config=config, task_config=task_config)


if __name__ == "__main__":
    run_tasks()  # type: ignore
