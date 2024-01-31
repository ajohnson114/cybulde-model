from typing import Any, Callable, Optional, Protocol

from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, default_collate
from transformers import BatchEncoding

from cybulde.data_modules.datasets import TextClassificationDataset
from cybulde.models.transformations import HuggingFaceTokenizationTransformation, Transformation


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[Any], Any]] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        """
        shuffle means we should shuffle our data or not
        sampler is the order we should sample from our dataset: returns n indices
        batch_sampler is sampler but for a single batch: returns batch_size indices
        num_workers means the amount of processes (CPUs) reading the dataset
        collate_fn runs the fn on batches after each batch is created. For ex - a tokenizer
        pin_memory is for GPUs if true the samples read are directly copied directly to the GPU
        
        drop_last means that the data that is leftover after creating the batches: 
        e.g data_size % batch_size = leftover data should be kept or dropped. Dropping shouldnt 
        be that big of a deal since we always shuffle the data so those left out will be included in another run

        persistent_workers means that the dataloader wont shutdown the worker process after a daaset
        has been consumed once, lets the workers Dataset instance stay alive
        """
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers

    def initialize_dataloader(self, dataset: Dataset, is_test: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle and not is_test,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )


class PartialDataModuleType(Protocol):
    def __call__(self, transformation: Transformation) -> DataModule:
        ...


class TextClassificationDataModule(DataModule):
    def __init__(
        self,
        train_df_path: str,
        dev_df_path: str,
        test_df_path: str,
        transformation: HuggingFaceTokenizationTransformation, #this is our tokenizer
        text_column_name: str,
        label_column_name: str,
        batch_size: int,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        def tokenization_collate_fn(batch: list[tuple[str, int]]) -> tuple[BatchEncoding, Tensor]:
            texts, labels = default_collate(batch)
            encodings = transformation(texts)
            return encodings, labels

        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=tokenization_collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )

        self.train_df_path = train_df_path
        self.dev_df_path = dev_df_path
        self.test_df_path = test_df_path

        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None: #means we're training
            self.train_dataset = TextClassificationDataset(
                self.train_df_path, self.text_column_name, self.label_column_name
            )
            self.dev_dataset = TextClassificationDataset(
                self.dev_df_path, self.text_column_name, self.label_column_name
            )

        if stage == "test":
            self.test_dataset = TextClassificationDataset(
                self.test_df_path, self.text_column_name, self.label_column_name
            )

    def train_dataloader(self) -> DataLoader:
        return self.initialize_dataloader(self.train_dataset, is_test=False)

    def val_dataloader(self) -> DataLoader:
        return self.initialize_dataloader(self.dev_dataset, is_test=True)

    def test_dataloader(self) -> DataLoader:
        return self.initialize_dataloader(self.test_dataset, is_test=True)