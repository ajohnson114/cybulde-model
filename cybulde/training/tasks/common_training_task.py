from typing import TYPE_CHECKING, Union

from lightning.pytorch import Trainer

from cybulde.data_modules.data_modules import DataModule, PartialDataModuleType
from cybulde.training.lightning_modules.bases import TrainingLightningModule
from cybulde.training.tasks.bases import TrainingTask
from cybulde.utils.io_utils import is_file
from cybulde.utils.mlflow_utils import activate_mlflow, log_artifacts_for_reproducibility

from hydra.utils import instantiate
from cybulde.utils.config_utils import get_config
from cybulde.config_schemas.config_schema import Config
from omegaconf import OmegaConf
import torch


if TYPE_CHECKING:
    from cybulde.config_schemas.config_schema import Config
    from cybulde.config_schemas.training.training_task_schemas import TrainingTaskConfig


class CommonTrainingTask(TrainingTask):
    def __init__(
        self,
        name: str,
        data_module: Union[DataModule, PartialDataModuleType],
        lightning_module: TrainingLightningModule,
        trainer: Trainer,
        best_training_checkpoint: str,
        last_training_checkpoint: str,
    ) -> None:
        super().__init__(
            name=name,
            data_module=data_module,
            lightning_module=lightning_module,
            trainer=trainer,
            best_training_checkpoint=best_training_checkpoint,
            last_training_checkpoint=last_training_checkpoint,
        )

    def run(self, config: "Config", task_config: "TrainingTaskConfig") -> None:
        experiment_name = config.infrastructure.mlflow.experiment_name
        run_id = config.infrastructure.mlflow.run_id
        run_name = config.infrastructure.mlflow.run_name

        with activate_mlflow(experiment_name=experiment_name, run_id=run_id, run_name=run_name) as _:
            if self.trainer.is_global_zero: #makes sure the code is run once for 1 gpu in the entire cluster
                log_artifacts_for_reproducibility()

            assert isinstance(self.data_module, DataModule) #this fixes an error do to the Union type enforcement 
            if is_file(self.last_training_checkpoint):
                self.logger.info("Found checkpoint here: {self.last_training_checkpoint}. Resuming training...")
                self.trainer.fit(
                    model=self.lightning_module, datamodule=self.data_module, ckpt_path=self.last_training_checkpoint
                )
            else:
                self.trainer.fit(model=self.lightning_module, datamodule=self.data_module)

            self.logger.info("training finished...")
    
    def run_laptop(self):
        self.trainer.fit(model=self.lightning_module, datamodule=self.data_module)


@get_config(config_path="../configs/automatically_generated", config_name="config", to_object=False, return_dict_config=True)
def run_tasks(config: Config) -> None:
    data_module = instantiate(config.tasks.binary_text_classification_task.data_module)
    lightning_module = instantiate(config.tasks.binary_text_classification_task.lightning_module)
    trainer = instantiate(config.tasks.binary_text_classification_task.trainer)

    trainer.fit(lightning_module,data_module)
    
    # model = CommonTrainingTask(
    #         name= "binary_text_classfication_task",
    #         data_module=data_module,
    #         lightning_module=lightning_module,
    #         trainer= trainer,
    #         best_training_checkpoint='.',
    #         last_training_checkpoint='.')
    
    # model.run_laptop()
    
    #lookup torchrun for more info here
    # backend="gloo" #lib for making cpus communicate
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(f"cuda:{get_local_rank()}")
    #     backend = "nccl" #lib for making gpus communicate

    # torch.distributed.init_process_group(backend = backend)

    # seed_everything(seed=config.seed, workers=True)
    # for task_name, task_config in config.tasks.items():
        # print(60*"#")
        # print(f"{task_name}")
        # print(60*"#")
        # task = instantiate(task_config)
        # task.run(config=config, task_config=task_config)
    

if __name__ == "__main__":
    run_tasks()

