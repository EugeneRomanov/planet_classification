import os
from typing import Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from augmentations import get_transforms
from config import DataConfig
from dataset import AmazonDataset
from dataset_splitter import stratify_shuffle_split_subsets


class AmazonDM(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self._config = config
        self._images_folder = os.path.join(self._config.data_path, 'train-jpg')

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        split_and_save_datasets(self._config.data_path, self._config.train_size)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            df_train = read_df(self._config.data_path, 'train')
            df_valid = read_df(self._config.data_path, 'valid')
            self.train_dataset = AmazonDataset(
                df_train,
                image_folder=self._images_folder,
                transforms=get_transforms(width=self._config.width, height=self._config.height),
            )
            self.valid_dataset = AmazonDataset(
                df_valid,
                image_folder=self._images_folder,
                transforms=get_transforms(width=self._config.width, height=self._config.height),
            )

        elif stage == 'test':
            df_test = read_df(self._config.data_path, 'test')
            self.test_dataset = AmazonDataset(
                df_test,
                image_folder=self._images_folder,
                transforms=get_transforms(width=self._config.width, height=self._config.height),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


def split_and_save_datasets(data_path: str, train_fraction: float = 0.8):
    print(os.path.join(data_path, 'train_classes.csv'))
    df = pd.read_csv(os.path.join(data_path, 'train_classes.csv'))
    df = df.rename(columns={'image_name': 'Id'})
    prep_df = pd.concat([df['Id'], df['tags'].str.get_dummies(sep=' ')], axis=1)
    prep_df = prep_df.fillna(0)
    prep_df.columns = prep_df.columns.str.capitalize()
    train_df, valid_df, test_df = stratify_shuffle_split_subsets(prep_df, train_fraction=train_fraction)
    train_df.to_csv(os.path.join(data_path, 'df_train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'df_valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'df_test.csv'), index=False)


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_path, f'df_{mode}.csv'))
