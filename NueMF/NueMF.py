import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import numpy as np
import torch.nn.functional as F
from typing import List
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from pathlib import Path
import yaml
from typing import Optional
import logging


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
            '../../../uploads/ml-1m/ratings.dat', 
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


class GMF(pl.LightningModule):
    def __init__(
            self, 
            num_users: int, 
            num_items: int, 
            embedding_dim: int = 1024,
        ):
        super(GMF, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)
    
    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        gmf = torch.mul(user_embedding, item_embedding)
        gmf = self.output_layer(gmf)
        o = torch.sigmoid(gmf)
        return o
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))
        acc = (y_pred.round() == labels).float().mean()
        
        self.log('GMF_train_loss', loss)
        self.log('GMF_train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))
        acc = (y_pred.round() == labels).float().mean()

        self.log('GMF_val_loss', loss, prog_bar=True)
        self.log('GMF_val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

class MLP(pl.LightningModule):
    def __init__(
            self, 
            num_users: int, 
            num_items: int, 
            embedding_dim: int = 1024,
            mlp_layers: List[int] = [1024, 512, 256, 128],
        ):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(nn.Linear(embedding_dim * 2, mlp_layers[0]))
        for i in range(1, len(mlp_layers)):
            self.mlp_layers.append(nn.Linear(mlp_layers[i-1], mlp_layers[i]))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(p=0.1))
        self.output_layer = nn.Linear(mlp_layers[-1], 1)
    
    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        for layer in self.mlp_layers:
            x = layer(x)
        x = self.output_layer(x)
        o = torch.sigmoid(x)
        return o
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))
        acc = (y_pred.round() == labels).float().mean()

        self.log('MLP_train_loss', loss, prog_bar=True)
        self.log('MLP_train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))
        acc = (y_pred.round() == labels).float().mean()

        self.log('MLP_val_loss', loss, prog_bar=True)
        self.log('MLP_val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

class NeuMF(pl.LightningModule):
    def __init__(
            self, 
            num_users: int, 
            num_items: int, 
            gmf_checkpoint: str,
            mlp_checkpoint: str,
            gmf_embedding_dim: int = 1024,
            mlp_embedding_dim: int = 1024,
            mlp_layers: List[int] = [1024, 512, 256, 128],
            batch_size: int = 512,
        ):
        super(NeuMF, self).__init__()

        self.gmf_checkpoint = gmf_checkpoint
        self.mlp_checkpoint = mlp_checkpoint

        self.gmf = GMF.load_from_checkpoint(
            self.gmf_checkpoint,
            num_users=num_users,
            num_items=num_items,
            embedding_dim=gmf_embedding_dim
        )
        self.mlp = MLP.load_from_checkpoint(
            self.mlp_checkpoint,
            num_users=num_users,
            num_items=num_items,
            embedding_dim=mlp_embedding_dim,
            mlp_layers=mlp_layers
        )

        # my output needs to be (batch_size, 1)
        self.output_layer = nn.Linear(batch_size*2, batch_size)
    
    def _pretrain_gmf(self):
        datamodule = MovieLensDataModule(batch_size=self.batch_size, is_dev=True)
        gmf_checkpoint_callback = ModelCheckpoint(
            monitor='GMF_val_loss',
            dirpath='checkpoints',
            filename='gmf-{epoch:02d}.ckpt',
            save_top_k=3,
        )
        model = GMF(num_users=self.num_users, num_items=self.num_items)
        logger = TensorBoardLogger('lightning_logs', name='GMF')
        trainer = pl.Trainer(max_epochs=self.max_epochs, callbacks=[gmf_checkpoint_callback], logger=logger)
        trainer.fit(model, datamodule=datamodule)
        return gmf_checkpoint_callback.best_model_path

    def _pretrain_mlp(self):
        datamodule = MovieLensDataModule(batch_size=self.batch_size, is_dev=True)
        mlp_checkpoint_callback = ModelCheckpoint(
            monitor='MLP_val_loss',
            dirpath='checkpoints',
            filename='mlp-{epoch:02d}.ckpt',
            save_top_k=3,
        )
        model = MLP(num_users=self.num_users, num_items=self.num_items)
        logger = TensorBoardLogger('lightning_logs', name='MLP')
        trainer = pl.Trainer(max_epochs=self.max_epochs, callbacks=[mlp_checkpoint_callback], logger=logger)
        trainer.fit(model, datamodule=datamodule)
        return mlp_checkpoint_callback.best_model_path

    def forward(self, user_ids, item_ids):
        gmf_output = self.gmf(user_ids, item_ids)
        mlp_output = self.mlp(user_ids, item_ids)
        concat = torch.cat([gmf_output, mlp_output], dim=0).view(1, -1) # (batch_size*2, 1) -> (1, batch_size*2)
        output = self.output_layer(concat).view(-1, 1)
        return output
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))
        acc = (y_pred.round() == labels).float().mean()

        self.log('NeuMF_train_loss', loss)
        self.log('NeuMF_train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))
        acc = (y_pred.round() == labels).float().mean()

        self.log('NeuMF_val_loss', loss)
        self.log('NeuMF_val_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

def load_yaml_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Matrix Factorization')
    
    # Add config file argument
    parser.add_argument('--config', type=str, default=None,
                      help='Path to YAML config file')
    
    # Parse known args first
    args, _ = parser.parse_known_args()
    
    # Load YAML config if provided
    config = load_yaml_config(args.config)
    
    # Model parameters
    parser.add_argument('--model_type', type=str, 
                      default=config.get('model_type', 'neumf'),
                      choices=['gmf', 'mlp', 'neumf'],
                      help='Type of model to train (default: neumf)')
    parser.add_argument('--num_users', type=int, 
                      default=config.get('num_users', 6040),
                      help='Number of users in the dataset')
    parser.add_argument('--num_items', type=int, 
                      default=config.get('num_items', 3706),
                      help='Number of items in the dataset')
    parser.add_argument('--gmf_embedding_dim', type=int, 
                      default=config.get('gmf_embedding_dim', 1024),
                      help='GMF embedding dimension')
    parser.add_argument('--mlp_embedding_dim', type=int, 
                      default=config.get('mlp_embedding_dim', 1024),
                      help='MLP embedding dimension')
    parser.add_argument('--mlp_layers', type=str, 
                      default=config.get('mlp_layers', '1024,512,256,128'),
                      help='Comma-separated list of MLP layer dimensions')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, 
                      default=config.get('batch_size', 1024),
                      help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, 
                      default=config.get('max_epochs', 10),
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, 
                      default=config.get('learning_rate', 0.01),
                      help='Learning rate')
    parser.add_argument('--is_dev', action='store_true',
                      help='Run in development mode')
    
    # Paths and devices
    parser.add_argument('--checkpoint_dir', type=str, 
                      default=config.get('checkpoint_dir', 'checkpoints'),
                      help='Directory to save checkpoints')
    parser.add_argument('--gmf_checkpoint', type=str, 
                      default=config.get('gmf_checkpoint', None),
                      help='Path to GMF checkpoint')
    parser.add_argument('--mlp_checkpoint', type=str, 
                      default=config.get('mlp_checkpoint', None),
                      help='Path to MLP checkpoint')
    parser.add_argument('--device', type=str, 
                      default=config.get('device', 'cuda'),
                      help='Device to use (cuda/cpu)')
    
    parser.add_argument('--pretrain_gmf', type=bool, 
                      default=config.get('pretrain_gmf', False),
                      help='Whether to pretrain GMF')
    parser.add_argument('--pretrain_mlp', type=bool, 
                      default=config.get('pretrain_mlp', False),
                      help='Whether to pretrain MLP')
    
    args = parser.parse_args()
    return args

def _pretrain_gmf(args, pylogger):
    pylogger.info("Pretraining GMF")
    datamodule = MovieLensDataModule(batch_size=args.batch_size, is_dev=True)
    gmf_checkpoint_callback = ModelCheckpoint(
        monitor='GMF_val_loss',
        dirpath='checkpoints',
        filename='gmf-{epoch:02d}.ckpt',
        save_top_k=3,
    )
    model = GMF(num_users=args.num_users, num_items=args.num_items)
    logger = TensorBoardLogger('lightning_logs', name='GMF')
    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[gmf_checkpoint_callback], logger=logger)
    trainer.fit(model, datamodule=datamodule)
    return gmf_checkpoint_callback.best_model_path

def _pretrain_mlp(args, pylogger):
    pylogger.info("Pretraining MLP")
    datamodule = MovieLensDataModule(batch_size=args.batch_size, is_dev=True)
    mlp_checkpoint_callback = ModelCheckpoint(
        monitor='MLP_val_loss',
        dirpath='checkpoints',
        filename='mlp-{epoch:02d}.ckpt',
        save_top_k=3,
    )
    model = MLP(num_users=args.num_users, num_items=args.num_items)
    logger = TensorBoardLogger('lightning_logs', name='MLP')
    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[mlp_checkpoint_callback], logger=logger)
    trainer.fit(model, datamodule=datamodule)
    return mlp_checkpoint_callback.best_model_path

def setup_logger():
    """Configure and return a logger instance."""
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handler to logger if it doesn't already have one
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger

def main():
    # Initialize logger
    pylogger = setup_logger()
    pylogger.info("Starting Neural Matrix Factorization training")
    
    args = parse_args()

    # Convert mlp_layers string to list of integers
    mlp_layers = [int(x) for x in args.mlp_layers.split(',')]
    
    # Create checkpoint directory if it doesn't exist
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if args.pretrain_gmf:
        args.gmf_checkpoint = _pretrain_gmf(args, pylogger)
    if args.pretrain_mlp:
        args.mlp_checkpoint = _pretrain_mlp(args, pylogger)
    
    # Initialize data module
    data_module = MovieLensDataModule(
        batch_size=args.batch_size,
        is_dev=args.is_dev
    )
    
    # Initialize model based on type
    if args.model_type == 'gmf':
        model = GMF(
            num_users=args.num_users,
            num_items=args.num_items,
            embedding_dim=args.gmf_embedding_dim
        )

    elif args.model_type == 'mlp':
        model = MLP(
            num_users=args.num_users,
            num_items=args.num_items,
            embedding_dim=args.mlp_embedding_dim,
            mlp_layers=mlp_layers
        )
    else:  # neumf
        model = NeuMF(
            num_users=args.num_users,
            num_items=args.num_items,
            gmf_checkpoint=args.gmf_checkpoint,
            mlp_checkpoint=args.mlp_checkpoint,
            gmf_embedding_dim=args.gmf_embedding_dim,
            mlp_embedding_dim=args.mlp_embedding_dim,
            mlp_layers=mlp_layers,
            batch_size=args.batch_size
        )
    
    # Setup trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'{args.model_type}-{{epoch:02d}}-{{val_loss:.2f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.device == 'cuda' else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger('lightning_logs', name=args.model_type)
    )
    
    # Train model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()