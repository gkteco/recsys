import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils.logger import setup_logger

class MovieLensDataset(Dataset):
    def __init__(
            self, 
            df: pd.DataFrame,
            num_negatives: int = 4,
            is_training: bool = True,
            is_dev: bool = False,
            num_test_negatives: int = 99
        ):
        self.df = df
        self.users = df["user_id"].unique()
        self.items = df["item_id"].unique()
        self.user_to_idx = {user: i for i, user in enumerate(self.users)}
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}
        self.num_negatives = num_negatives
        self.is_training = is_training
        self.num_test_negatives = num_test_negatives
        self.is_dev = is_dev

        self.user_items = {}
        # create user-item interacitons
        for user in self.users:
            self.user_items[user] = set(
                df[df["user_id"] == user]["item_id"].values
            )
        
        if not self.is_dev:
            if self.is_training:
                self.samples = self._generate_training_samples()
            else:
                self.samples = self._generate_test_samples()

    def _generate_training_samples(self):
        pylogger = setup_logger()
        pylogger.info("Generating training samples...")
        samples = []

        # First add all positive interactions
        for _, row in self.df.iterrows():
            samples.append({
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "rating": 1.0
                })
            # generate negative smaples for this positive interaction
            user_items = self.user_items[row["user_id"]]
            neg_items = list(set(self.items) - user_items) # All items user has not interacted with
            # Randomly sample negative items
            if len(neg_items) >= self.num_negatives:
                neg_samples = np.random.choice(
                    neg_items,
                    size=self.num_negatives,
                    replace=False
                )
            else:
                neg_samples = np.random.choice(
                    neg_items,
                    size=self.num_negatives,
                    replace=True
                )
            for item in neg_samples:
                samples.append({
                    "user_id": row["user_id"],
                    "item_id": item,
                    "rating": 0.0
                })
        return samples
    
    def _generate_test_samples(self):
        pylogger = setup_logger()
        pylogger.info("Generating test samples...")
        samples = []

        for _, row in self.df.iterrows():
            samples.append({
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "rating": 1.0
            })

            user_items = self.user_items[row["user_id"]]
            neg_items = list(set(self.items) - user_items)
            neg_samples = np.random.choice(
                neg_items,
                size=self.num_test_negatives,
                replace=len(neg_items) < self.num_test_negatives
            )

            for item in neg_samples:
                samples.append({
                    "user_id": row["user_id"],
                    "item_id": item,
                    "rating": 0.0
                })
        return samples
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "user_id": torch.tensor(self.user_to_idx[row["user_id"]], dtype=torch.long),
            "item_id": torch.tensor(self.item_to_idx[row["item_id"]], dtype=torch.long),
            "rating": torch.tensor(row["rating"], dtype=torch.float),
        }
    

class MovieLensDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 256, is_dev: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.is_dev = is_dev

    def setup(self, stage: str):     
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        self.df = pd.read_csv(
            '../../../../uploads/ml-1m/ratings.dat', 
            sep='::', 
            engine='python',
            names=names,
            encoding='latin-1'
        )

        # Convert all ratings to implicit feedback
        self.df['rating'] = 1.0

        # Keep one out df operations
        self.df = self.df.sort_values('timestamp')
        test_df = self.df.groupby('user_id').last().reset_index()
        train_df = self.df.merge(
            test_df[['user_id', 'item_id']],
            on=['user_id', 'item_id'],
            how='left',
            indicator=True
        )
        train_df = train_df[train_df['_merge'] == 'left_only'].drop(columns=['_merge'], axis=1)
        if stage == "bench":
            self.val_dataset = MovieLensDataset(
                test_df,
                num_negatives=99,
                is_training=False,
                is_dev=self.is_dev
            )
            return

        self.train_dataset = MovieLensDataset(
            train_df,
            num_negatives=4,
            is_training=True,
            is_dev=self.is_dev
        )
        self.val_dataset = MovieLensDataset(
            test_df,
            num_negatives=99,
            is_training=False,
            is_dev=self.is_dev
        )
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
