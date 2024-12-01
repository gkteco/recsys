import yaml
from typing import Optional 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from NueMF import GMF, MLP, NeuMF
from datamodule import MovieLensDataModule
from utils.logger import setup_logger
from utils.pretrainers import _pretrain_gmf, _pretrain_mlp
from utils.benchmarker import run_benchmarks
import argparse
from pathlib import Path

def load_yaml_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        return {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Neural Matrix Factorization for Recommendation Systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
        ''')
    
    # Add config file argument with better help message
    parser.add_argument('--config', type=str, default=None,
                      help='Path to YAML config file containing model parameters')
    
    # Group model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--model_type', type=str, 
                      default='neumf',
                      choices=['gmf', 'mlp', 'neumf'],
                      help='Type of model to train (default: neumf)')
    model_group.add_argument('--num_users', type=int, 
                      default=6040,
                      help='Number of users in the dataset (default: 6040)')
    model_group.add_argument('--num_items', type=int, 
                      default=3706,
                      help='Number of items in the dataset (default: 3706)')
    model_group.add_argument('--gmf_embedding_dim', type=int, 
                      default=1024,
                      help='GMF embedding dimension (default: 1024)')
    model_group.add_argument('--mlp_embedding_dim', type=int, 
                      default=1024,
                      help='MLP embedding dimension (default: 1024)')
    model_group.add_argument('--mlp_layers', type=str, 
                      default='1024,512,256,128',
                      help='Comma-separated list of MLP layer dimensions (default: 1024,512,256,128)')
    
    # Group training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--batch_size', type=int, 
                      default=1024,
                      help='Batch size for training (default: 1024)')
    train_group.add_argument('--max_epochs', type=int, 
                      default=10,
                      help='Number of epochs to train (default: 10)')
    train_group.add_argument('--learning_rate', type=float, 
                      default=0.01,
                      help='Learning rate (default: 0.01)')
    train_group.add_argument('--is_dev', action='store_true',
                      help='Run in development mode with reduced negative sampling overhead')
    
    # Group paths and devices
    path_group = parser.add_argument_group('Paths and Devices')
    path_group.add_argument('--checkpoint_dir', type=str, 
                      default='checkpoints',
                      help='Directory to save checkpoints (default: checkpoints)')
    path_group.add_argument('--gmf_checkpoint', type=str, 
                      default=None,
                      help='Path to pretrained GMF checkpoint')
    path_group.add_argument('--mlp_checkpoint', type=str, 
                      default=None,
                      help='Path to pretrained MLP checkpoint')
    path_group.add_argument('--device', type=str, 
                      default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to use for training (default: cuda)')
    
    # Group pretraining options
    pretrain_group = parser.add_argument_group('Pretraining Options')
    pretrain_group.add_argument('--pretrain_gmf', type=bool, 
                      default=False,
                      help='Whether to pretrain GMF model (default: False)')
    pretrain_group.add_argument('--pretrain_mlp', type=bool, 
                      default=False,
                      help='Whether to pretrain MLP model (default: False)')
    
    # Add compilation arguments
    compile_group = parser.add_argument_group('Compilation Options')
    compile_group.add_argument('--run_bench', action='store_true',
                       help='Run benchmarking comparing torch.compile and thunder')
    compile_group.add_argument('--nuemf_checkpoint', type=str,
                      default=None,
                      help='Path to pretrained NeuMF checkpoint')

    args = parser.parse_args()
    
    # Load config file if specified
    config = load_yaml_config(args.config)
    
    # Update args with config values, command line args take precedence
    args_dict = vars(args)
    for key, value in config.items():
        if key in args_dict and args_dict[key] is None:  # Only update if not set via command line
            args_dict[key] = value
    
    return argparse.Namespace(**args_dict)

def main():
    args = parse_args()
    pylogger = setup_logger()
    
    if args.run_bench:
        data_module = MovieLensDataModule(
            batch_size=4096, # Manually setting this to a large batch size for benchmarking
            is_dev=False
        )
        pylogger.info("Setting up data module...")
        data_module.setup('bench')
        pylogger.info("Running benchmarks...")
        run_benchmarks(args, data_module)
        return
        
    # Convert mlp_layers string to list of integers
    mlp_layers = [int(x) for x in args.mlp_layers.split(',')]
    
    # Create checkpoint directory if it doesn't exist
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Pretrain models if specified
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
        filename=f'{args.model_type}-{{epoch:02d}}-{{NeuMF_val_loss:.2f}}',
        monitor='NeuMF_val_loss',
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