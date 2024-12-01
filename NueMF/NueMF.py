import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import List

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
        return o # (batch_size, 1)
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))
        
        self.log('GMF_train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))

        self.log('GMF_val_loss', loss, prog_bar=True)
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
        return o # (batch_size, 1)
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))

        self.log('MLP_train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))

        self.log('MLP_val_loss', loss, prog_bar=True)
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
        ):
        super(NeuMF, self).__init__()

        self.gmf_checkpoint = gmf_checkpoint
        self.mlp_checkpoint = mlp_checkpoint

        self.gmf = GMF.load_from_checkpoint(
            self.gmf_checkpoint,
            num_users=num_users,
            num_items=num_items,
            embedding_dim=gmf_embedding_dim
        ).requires_grad_(False).eval()


        self.mlp = MLP.load_from_checkpoint(
            self.mlp_checkpoint,
            num_users=num_users,
            num_items=num_items,
            embedding_dim=mlp_embedding_dim,
            mlp_layers=mlp_layers
        ).requires_grad_(False).eval()

        # (batch_size, gmf_embedding_dim + mlp_layers[-1]) -> (batch_size, 1)
        self.output_layer = nn.Linear(gmf_embedding_dim + mlp_layers[-1], 1)

    def forward(self, user_ids, item_ids):
        gmf_user_embs = self.gmf.user_embedding (user_ids) # (batch_size, embedding_dim)
        gmf_item_embs = self.gmf.item_embedding(item_ids)
        gmf_vector = torch.mul(gmf_user_embs, gmf_item_embs) # (batch_size, embedding_dim)

        mlp_user_embs = self.mlp.user_embedding(user_ids) # (batch_size, embedding_dim)
        mlp_item_embs = self.mlp.item_embedding(item_ids)
        x = torch.cat([mlp_user_embs, mlp_item_embs], dim=1) # (batch_size, embedding_dim * 2)
        for layer in self.mlp.mlp_layers:
            x = layer(x)
        mlp_vector = x # (batch_size, mlp_layers[-1])

        concat = torch.cat([gmf_vector, mlp_vector], dim=1) # (batch_size, mlp_vector + gmf_vector)
        output = self.output_layer(concat) # (batch_size, mlp_vector + gmf_vector) -> (batch_size, 1)
        output = torch.sigmoid(output) # (batch_size, 1)
        return output
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))

        self.log('NeuMF_train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch["user_id"], batch["item_id"], batch["rating"]
        y_pred = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(y_pred, labels.view(-1, 1))

        self.log('NeuMF_val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

