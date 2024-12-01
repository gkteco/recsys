import torch
import time
from typing import Literal
from torch.utils.data import DataLoader
from NueMF import NeuMF
from utils.logger import setup_logger
from datamodule import MovieLensDataModule
import thunder

def compile_model(model: torch.nn.Module, 
                 compiler: Literal['thunder', 'torch', 'none']) -> torch.nn.Module:
    """Compile the model using the specified strategy."""
    if compiler == 'thunder':
        return thunder.jit(model) 
    elif compiler == 'torch':
        return torch.compile(model)
    return model

def benchmark_inference(model: torch.nn.Module,
                       dataloader: DataLoader,
                       device: str,
                       num_runs: int = 100,
                       profile: bool = False
                       ) -> float:
    """Run inference benchmark and return average time per batch."""
    model.eval()
    total_time = 0.0
    num_batches = len(dataloader)  # Get total number of batches
    
    with torch.no_grad():
        # warmup for an entire epoch
        for batch in dataloader:
            user_ids = batch["user_id"].to(device)
            item_ids = batch["item_id"].to(device)
            _ = model(user_ids, item_ids)
            if device == 'cuda':
                torch.cuda.synchronize()

        # Actual timing runs
        for run in range(num_runs):
            batch_times = []
            
            for batch in dataloader:
                user_ids = batch["user_id"].to(device)
                item_ids = batch["item_id"].to(device)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                _ = model(user_ids, item_ids)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                batch_times.append(time.perf_counter() - start_time)
            
            total_time += sum(batch_times)
            
            # Print progress every 10 runs
            if run % 10 == 0:
                print(f"Completed {run}/{num_runs} runs. "
                      f"Average time per batch: {(sum(batch_times)/len(batch_times))*1000:.2f}ms")
    
    avg_time = total_time / (num_runs * num_batches)
    print(f"Final average time per batch: {avg_time*1000:.2f}ms")
    print(f"Total batches per run: {num_batches}")
    if profile:
        from torch.profiler import profile, record_function, ProfilerActivity
        batch = next(iter(dataloader))
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("profiler"):
                _ = model(user_ids, item_ids)
        print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    return total_time / num_runs  # Return average time per epoch

def run_benchmarks(args, data_module: MovieLensDataModule):
    """Run benchmarks comparing different compilation strategies."""
    pylogger = setup_logger()
    pylogger.info("Running inference benchmarks...")
    
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    # Initialize models
    if args.nuemf_checkpoint is None:
        return
    else:
        base_model = NeuMF.load_from_checkpoint(
                args.nuemf_checkpoint,
                num_users=args.num_users,
                num_items=args.num_items,
                gmf_checkpoint=args.gmf_checkpoint,
                mlp_checkpoint=args.mlp_checkpoint,
        ).requires_grad_(False).eval().to(device)
        # Get validation dataloader for benchmarking
        val_loader = data_module.val_dataloader()
        results = {}
        
        # Benchmark uncompiled
        pylogger.info(f"Benchmarking uncompiled model...")
        results['uncompiled'] = benchmark_inference(base_model, val_loader, device, profile=args.profile)
            
        # Benchmark torch.compile
        pylogger.info(f"Benchmarking torch.compile model...")
        torch_model = compile_model(base_model, 'torch')
        results['torch'] = benchmark_inference(torch_model, val_loader, device, profile=args.profile)
            
        # Benchmark thunder
        pylogger.info(f"Benchmarking thunder model...")
        thunder_model = compile_model(base_model, 'thunder')
        results['thunder'] = benchmark_inference(thunder_model, val_loader, device, profile=args.profile)
    
    # Print results
    pylogger.info("\nBenchmark Results (seconds per epoch):")
    pylogger.info("-" * 50)
    for name, time in results.items():
        pylogger.info(f"{name:12}: {time:.4f}s")
    
    # Calculate speedups
    baseline = results['uncompiled']
    pylogger.info("\nSpeedups vs uncompiled:")
    for name, time in results.items():
        if name != 'uncompiled':
            speedup = baseline / time
            pylogger.info(f"{name:12}: {speedup:.2f}x")
