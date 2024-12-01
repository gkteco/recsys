import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from NueMF import GMF, MLP
from datamodule import MovieLensDataModule

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