import torch 
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter 

import numpy as np 
from omegaconf import OmegaConf 
from pathlib import Path 
from utils import (
    seed_everything,
    setup_logger,
    plot_loss_curves,
    plot_prediction_results,
    get_reconstruction_metrics,
    reconstruction_metrics_from_arrays,
    profile_model_complexity,
)
import yaml 
import time 
import shutil 
import argparse
from tqdm.auto import tqdm

from models import get_model 
from datasets import get_dataset 


def _save_training_state(
    path,
    model,
    optimizer,
    scheduler,
    step,
    best_val_loss,
    best_val_pcc,
    patience_counter,
    train_losses,
    val_losses,
    val_pccs,
    val_metric_history,
    args,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "best_val_pcc": best_val_pcc,
            "patience_counter": patience_counter,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_pccs": val_pccs,
            "val_metric_history": val_metric_history,
            "config": OmegaConf.to_container(args, resolve=True),
        },
        path,
    )


def train():     
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default="configs/ecg_baseline_wander_mecg_e.yaml")
    known_args, remaining = parser.parse_known_args()

    base_args = OmegaConf.load(known_args.config)
    cli_args = OmegaConf.from_dotlist(remaining) # load command line overrides
    args = OmegaConf.merge(base_args, cli_args)
    START_TIME = time.time() 
    separator = "-" * 100
    
    seed_everything(args.seed) 
    exp_name = args.exp_name 
    model_name = args.model_name 
    writer = None 
    metric_display_map = {
        'SSD': 'SSD (↓)',
        'MAD': 'MAD (↓)',
        'PRD': 'PRD % (↓)',
        'CosSim': 'CosSim (↑)',
        'Output_SNR_dB': 'Output SNR dB (↑)',
        'SNR_Improvement_dB': 'SNR Improvement dB (↑)',
        'LF_Reduction_dB': 'LF Reduction dB (↑)',
        'R_Peak_Timing_Error_ms': 'R-Peak Timing Error ms (↓)',
        'RR_Interval_MAE_ms': 'RR Interval MAE ms (↓)',
        'RMSE': 'RMSE (↓)',
        'Centered_CosSim': 'Centered CosSim (↑)',
        'QRS_Amplitude_Error': 'QRS Amplitude Error (↓)',
        'Parameters': 'Parameters',
        'Trainable_Parameters': 'Trainable Parameters',
        'FLOPs': 'FLOPs',
        'Inference_Time_ms': 'Inference Time ms',
        'Peak_Memory_MB': 'Peak Memory MB',
    }
    
    # Build a run tag from loss weights so each parameter combination gets its own dir
    _W_KEYS   = ['mse_w', 'pcc_w', 'spec_mag_w', 'spec_phase_w', 'grad_w', 'amp_w',
                 'stft_w', 'inst_phase_w', 'xcorr_lag_w', 'stft_mag_w', 'env_w']
    _W_ABBREV = {'mse_w': 'mse', 'pcc_w': 'pcc', 'spec_mag_w': 'smag',
                 'spec_phase_w': 'sph', 'grad_w': 'grad', 'amp_w': 'amp',
                 'stft_w': 'stft', 'inst_phase_w': 'iphase', 'xcorr_lag_w': 'xclag',
                 'stft_mag_w': 'stftmag', 'env_w': 'env'}
    model_cfg_flat = OmegaConf.to_container(args.model, resolve=True)
    weight_parts = [f"{_W_ABBREV[k]}{model_cfg_flat[k]}" for k in _W_KEYS if k in model_cfg_flat and model_cfg_flat[k] != 0]
    run_tag = "_".join(weight_parts)  # e.g. "mse0.1_pcc2.0_smag0.3_sph0.5_grad1.5_amp0.5"

    # setup directories — each unique weight combination gets its own subdirectory
    sub = Path(model_name) / run_tag if run_tag else Path(model_name)
    checkpoint_dir = Path(args.checkpoint_dir) / exp_name / sub
    results_dir    = Path(args.results_dir)    / exp_name / sub
    log_dir        = Path(args.log_dir)        / exp_name / sub

    resume = bool(args.training.get("resume", False))
    resume_checkpoint_value = args.training.get("resume_checkpoint", None)
    resume_checkpoint = (
        Path(resume_checkpoint_value)
        if resume_checkpoint_value
        else checkpoint_dir / "training_state.pt"
    )

    # delete only the current run's directory (not siblings from other weight combos)
    if not resume:
        for path in [checkpoint_dir, results_dir, log_dir]:
            if path.exists():
                shutil.rmtree(path)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)    
    
    # print config 
    logger = setup_logger(args, log_dir) 
    if args.get("log_config", False):
        logger.info("\n"+"-"*30+"Running with configs"+"-"*30+"\n")
        logger.info(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))
        logger.info(separator+"\n")
    else:
        logger.info(
            f"Run: exp_name={exp_name} | model_name={model_name} | "
            f"train_iterations={args.training.train_iterations} | batch_size={args.training.batch_size}"
        )
    
    # setup tensorboard 
    if args.get('log_tensorboard', True):
        tb_dir = checkpoint_dir / "tensorboard_logs"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard logging enabled. Logs will be saved to {tb_dir}")
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')
    # print(f'Using device {device}')
    
    # model_configs = OmegaConf.load(args.model_config_path) 
    # model_kwargs = OmegaConf.to_container(model_configs, resolve=True)  
    # logger.info("\n"+"-"*30+"Model configs"+"-"*30+"\n")
    # logger.info(yaml.dump(model_kwargs, indent=4)) 
    # logger.info(separator+"\n")
    model_kwargs = OmegaConf.to_container(args.model, resolve=True)
    model = get_model(args['model_name'], **model_kwargs).to(device) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.training.train_iterations, eta_min=1e-5
    )
    if args.get("log_model_architecture", False):
        logger.info("\n"+"-"*30+"Model Architecture"+"-"*30+"\n")
        logger.info(f'{model}\n')
    logger.info(f'model size: {sum(param.numel() for param in model.parameters())} parameters')

    dataset_kwargs = OmegaConf.to_container(args.dataset, resolve=True)
    train_dataset = get_dataset(data_mode='train', **dataset_kwargs) 
    val_dataset = get_dataset(data_mode='val', **dataset_kwargs) 
    test_dataset = get_dataset(data_mode='test', **dataset_kwargs) 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.training.batch_size, 
        shuffle=True, 
        num_workers=4, 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.training.batch_size, 
        shuffle=False, 
        num_workers=4, 
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.training.batch_size, 
        shuffle=False, 
        num_workers=4, 
    )
    # print(f'data loaders sucessfully loaded')
    logger.info(f'data loaders sucessfully loaded')
    logger.info("\n"+"-"*30+"Begin Training"+"-"*30+"\n")

    # training
    train_losses, val_losses = [], []
    val_pccs = []
    val_metric_history = []
    best_val_loss = np.inf
    best_val_pcc = -np.inf
    best_model_ckpt = checkpoint_dir / 'best_model.pt'
    best_pcc_model_ckpt = checkpoint_dir / 'best_pcc_model.pt'
    training_state_ckpt = checkpoint_dir / 'training_state.pt'

    patience = args.training.get('early_stopping_patience', 20)
    min_delta = float(args.training.get('early_stopping_min_delta', 0.0))
    validation_metrics_every = max(
        int(args.training.get('validation_metrics_every', args.training.eval_every)),
        int(args.training.eval_every),
    )
    patience_counter = 0
    stop_training = False

    step = 0
    running_train_loss = 0
    running_train_steps = 0

    if resume:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(
                f"training.resume=True but resume checkpoint was not found: {resume_checkpoint}"
            )
        state = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        step = int(state.get("step", 0))
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        best_val_pcc = float(state.get("best_val_pcc", best_val_pcc))
        patience_counter = int(state.get("patience_counter", patience_counter))
        train_losses = list(state.get("train_losses", train_losses))
        val_losses = list(state.get("val_losses", val_losses))
        val_pccs = list(state.get("val_pccs", val_pccs))
        val_metric_history = list(state.get("val_metric_history", val_metric_history))
        logger.info(f"Resumed training from {resume_checkpoint} at step {step}.")

    progress_bar = tqdm(
        total=args.training.train_iterations,
        initial=step,
        desc=f"Training {model_name}",
        unit="iter",
        dynamic_ncols=True,
        disable=not args.training.get("progress_bar", True),
    )

    try:
        while step < args.training.train_iterations and not stop_training:
            # logger.info(f'Training iteration {step+1}')
            model.train()

            for train_batch in train_loader:
                optimizer.zero_grad()
                train_loss = model.compute_loss(train_batch, device)
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                running_train_loss += train_loss.item()
                running_train_steps += 1
                current_step = step + 1

                # evaluation
                if current_step % args.training.eval_every == 0:
                    model.eval()
                    val_pcc_batches = []
                    last_metrics_step = int(val_metric_history[-1]["step"]) if val_metric_history else None
                    metrics_due = (
                        last_metrics_step is None
                        or current_step - last_metrics_step >= validation_metrics_every
                    )
                    avg_val_loss = None
                    val_noisy_batches, val_clean_batches, val_pred_batches = [], [], []

                    with torch.no_grad():
                        for val_batch in val_loader:
                            noisy = val_batch[0].to(device)
                            clean = val_batch[1].to(device)
                            pred = model(noisy)
                            pred_c = pred - pred.mean(dim=-1, keepdim=True)
                            clean_c = clean - clean.mean(dim=-1, keepdim=True)
                            pcc = (pred_c * clean_c).sum(dim=-1) / (
                                pred_c.norm(dim=-1) * clean_c.norm(dim=-1) + 1e-8
                            )
                            val_pcc_batches.append(pcc)
                            if metrics_due:
                                val_noisy_batches.append(noisy.detach().cpu().numpy())
                                val_clean_batches.append(clean.detach().cpu().numpy())
                                val_pred_batches.append(pred.detach().cpu().numpy())

                    avg_val_pcc = torch.cat([batch.flatten() for batch in val_pcc_batches]).mean().item()
                    val_metrics = None
                    if metrics_due:
                        avg_val_loss = 0
                        for val_batch in val_loader:
                            avg_val_loss += model.compute_loss(val_batch, device).item()
                        avg_val_loss /= len(val_loader)
                        fs = args.dataset.get("resample_hz", args.model.get("sampling_rate", 250))
                        low_freq_hz = args.get("evaluation", {}).get("low_frequency_high_hz", 0.5)
                        val_metrics = reconstruction_metrics_from_arrays(
                            np.concatenate(val_noisy_batches, axis=0),
                            np.concatenate(val_clean_batches, axis=0),
                            np.concatenate(val_pred_batches, axis=0),
                            fs=fs,
                            low_freq_hz=low_freq_hz,
                            eps=args.dataset.eps,
                        )
                        val_metric_history.append({"step": current_step, "val_loss": avg_val_loss, **val_metrics})
                        with open(checkpoint_dir / "validation_metrics.yaml", "w", encoding="utf-8") as f:
                            yaml.safe_dump(val_metric_history, f, sort_keys=False)

                    # Calculate the average train loss over the last 'eval_every' steps
                    current_train_loss = running_train_loss / running_train_steps
                    log_message = (
                        f'Training iter {current_step} | train loss: {current_train_loss:.4f} | '
                        f'val PCC: {avg_val_pcc:.4f}'
                    )
                    if val_metrics is not None:
                        log_message += (
                            f' | val loss: {avg_val_loss:.4f} | '
                            f'val PRD: {val_metrics["PRD"]:.4f} | '
                            f'val SNR imp: {val_metrics["SNR_Improvement_dB"]:.4f} | '
                            f'val LF red: {val_metrics["LF_Reduction_dB"]:.4f} | '
                            f'val R-peak err: {val_metrics["R_Peak_Timing_Error_ms"]:.4f} ms | '
                            f'val RR MAE: {val_metrics["RR_Interval_MAE_ms"]:.4f} ms'
                        )
                    logger.info(log_message)

                    train_losses.append(current_train_loss)
                    val_losses.append(avg_val_loss if avg_val_loss is not None else float("nan"))
                    val_pccs.append(avg_val_pcc)

                    if writer is not None:
                        writer.add_scalar('Train/Loss', current_train_loss, step)
                        writer.add_scalar('Val/PCC', avg_val_pcc, step)
                        if avg_val_loss is not None:
                            writer.add_scalar('Val/Loss', avg_val_loss, step)
                        if val_metrics is not None:
                            for key, value in val_metrics.items():
                                writer.add_scalar(f'Val/{key}', value, step)

                    # Primary selection and early stopping use validation PCC only.
                    if avg_val_pcc > best_val_pcc + min_delta:
                        logger.info(f'New best val PCC {avg_val_pcc:.4f} at iteration {current_step}! Saving model...')
                        best_val_pcc = avg_val_pcc
                        torch.save(model.state_dict(), best_pcc_model_ckpt)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        logger.info(f'No PCC improvement. Patience: {patience_counter}/{patience}')

                    # Save best validation-loss checkpoint for auxiliary analysis only.
                    if avg_val_loss is not None and avg_val_loss < best_val_loss:
                        logger.info(f'New best val loss {avg_val_loss:.4f} at iteration {current_step}! Saving model...')
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), best_model_ckpt)

                    if patience_counter >= patience:
                        logger.info(f'Early stopping triggered at iteration {current_step}.')
                        torch.save(model.state_dict(), checkpoint_dir / 'model_last.pt')
                        step = current_step
                        progress_bar.update(1)
                        _save_training_state(
                            training_state_ckpt,
                            model,
                            optimizer,
                            scheduler,
                            step,
                            best_val_loss,
                            best_val_pcc,
                            patience_counter,
                            train_losses,
                            val_losses,
                            val_pccs,
                            val_metric_history,
                            args,
                        )
                        stop_training = True
                        break

                    _save_training_state(
                        training_state_ckpt,
                        model,
                        optimizer,
                        scheduler,
                        current_step,
                        best_val_loss,
                        best_val_pcc,
                        patience_counter,
                        train_losses,
                        val_losses,
                        val_pccs,
                        val_metric_history,
                        args,
                    )

                    # Reset running training metrics
                    running_train_loss = 0
                    running_train_steps = 0
                    model.train()

                if args.training.get("save_every", 0) and current_step % args.training.save_every == 0:
                    torch.save(model.state_dict(), checkpoint_dir / f'model_step_{current_step}.pt')
                    _save_training_state(
                        training_state_ckpt,
                        model,
                        optimizer,
                        scheduler,
                        current_step,
                        best_val_loss,
                        best_val_pcc,
                        patience_counter,
                        train_losses,
                        val_losses,
                        val_pccs,
                        val_metric_history,
                        args,
                    )

                if current_step >= args.training.train_iterations:
                    logger.info(f'Saving last model')
                    torch.save(model.state_dict(), checkpoint_dir / 'model_last.pt')
                    step = current_step
                    progress_bar.update(1)
                    _save_training_state(
                        training_state_ckpt,
                        model,
                        optimizer,
                        scheduler,
                        step,
                        best_val_loss,
                        best_val_pcc,
                        patience_counter,
                        train_losses,
                        val_losses,
                        val_pccs,
                        val_metric_history,
                        args,
                    )
                    break

                step = current_step
                progress_bar.update(1)
                progress_bar.set_postfix(
                    train_loss=f"{train_loss.item():.4f}",
                    patience=f"{patience_counter}/{patience}",
                )
    finally:
        progress_bar.close()
            
    # print(f"Training complete after {(time.time()-START_TIME)/60:.2f} minutes. Best val loss: {best_val_loss:.4f}")
    logger.info(
        f"Training complete after {(time.time()-START_TIME)/60:.2f} minutes. "
        f"Best val loss: {best_val_loss:.4f} | Best val PCC: {best_val_pcc:.4f}"
    )
    # final evaluation — test best checkpoints independently
    eval_checkpoints = [
        ('best_pcc',  best_pcc_model_ckpt),
        ('best_loss', best_model_ckpt),
    ]

    for ckpt_name, ckpt_path in eval_checkpoints:
        if not ckpt_path.exists():
            logger.info(f"\n{'='*30} Skipping {ckpt_name} model ({ckpt_path} not found) {'='*30}")
            continue
        logger.info(f"\n{'='*30} Evaluating {ckpt_name} model {'='*30}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        eval_dir = results_dir / ckpt_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        loss_curve_path = plot_loss_curves(
            train_losses, val_losses,
            args.training.eval_every, eval_dir, val_pccs=val_pccs,
        )
        logger.info(f"Saved loss curves to {loss_curve_path}")

        fs = args.dataset.get("resample_hz", args.model.get("sampling_rate", 250))
        low_freq_hz = args.get("evaluation", {}).get("low_frequency_high_hz", 0.5)
        prediction_paths = plot_prediction_results(
            model, test_dataset, device, eval_dir,
            num_samples=3, seed=args.seed, eps=args.dataset.eps, fs=fs,
        )
        for prediction_path in prediction_paths:
            logger.info(f"Saved prediction plot to {prediction_path}")

        eval_loaders = [("ptbxl_fold10_test", test_loader)]
        for external_name in args.dataset.get("external_test_datasets", []):
            try:
                external_dataset = get_dataset(data_mode=external_name, **dataset_kwargs)
            except FileNotFoundError as exc:
                logger.info(f"Skipping external test dataset {external_name}: {exc}")
                continue
            external_loader = DataLoader(
                external_dataset,
                batch_size=args.training.batch_size,
                shuffle=False,
                num_workers=4,
            )
            eval_loaders.append((external_name, external_loader))

        for eval_name, eval_loader in eval_loaders:
            metrics_dict = get_reconstruction_metrics(
                model, eval_loader, device, fs=fs, low_freq_hz=low_freq_hz, eps=args.dataset.eps
            )
            logger.info('\n')
            logger.info(separator)
            logger.info(f"{ckpt_name} | {eval_name}")
            logger.info(f"{'Metric':>24} | {'Value':>12}")
            logger.info(separator)
            for key, value in metrics_dict.items():
                display_name = metric_display_map.get(key, key)
                logger.info(f"{display_name:>24} | {value:>12.4f}")
            logger.info(separator + "\n")
            with open(eval_dir / f"metrics_{eval_name}.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(metrics_dict, f, sort_keys=False)

        complexity = profile_model_complexity(
            model,
            device,
            input_length=int(args.dataset.get("window_size", 512)),
            batch_size=1,
        )
        logger.info('\n')
        logger.info(separator)
        logger.info(f"{ckpt_name} | complexity")
        logger.info(f"{'Metric':>24} | {'Value':>12}")
        logger.info(separator)
        for key, value in complexity.items():
            display_name = metric_display_map.get(key, key)
            logger.info(f"{display_name:>24} | {value:>12.4f}")
        logger.info(separator + "\n")
        with open(eval_dir / "complexity_summary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(complexity, f, sort_keys=False)

    
            

    
    


if __name__ == "__main__": 
    train() 
