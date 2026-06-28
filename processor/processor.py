import gc
import logging
import os
import time

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank
from utils.meter import AverageMeter
from utils.metrics import Evaluator


def split_prob(prob, threshold):
    if prob.min() > threshold:
        print("No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.")
        threshold = np.sort(prob)[len(prob) // 100]
    return (prob > threshold).astype(np.int64)


def _normalize_losses(losses):
    loss_min = losses.min()
    loss_max = losses.max()
    return (losses - loss_min) / (loss_max - loss_min).clamp_min(1e-8)


def _fit_two_component_gmm(input_loss, args):
    if args.noisy_rate > 0.4:
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    max_samples = 100000
    input_np = input_loss.cpu().numpy()
    if len(input_np) > max_samples:
        idx = np.random.choice(len(input_np), max_samples, replace=False)
        gmm.fit(input_np[idx])
    else:
        gmm.fit(input_np)
    return gmm, input_np


def get_loss(model, data_loader):
    logger = logging.getLogger("HHNC.train")
    model.eval()
    device = "cuda"
    data_size = len(data_loader.dataset)

    loss_bank = torch.zeros(data_size, dtype=torch.float32)
    torch.cuda.empty_cache()

    for i, batch in enumerate(data_loader):
        index = batch["index"].clone()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            per_loss = model.compute_per_loss(batch)
            loss_bank[index.cpu()] = per_loss.detach().cpu().float()

        del batch, per_loss, index
        if i % 200 == 0:
            logger.info(f"compute loss batch {i}")
            torch.cuda.empty_cache()
            gc.collect()

    input_loss = _normalize_losses(loss_bank).reshape(-1, 1)
    logger.info("\nFitting GMM ...")

    gmm, input_np = _fit_two_component_gmm(input_loss, model.args)
    prob = gmm.predict_proba(input_np)[:, gmm.means_.argmin()]
    threshold = getattr(model.args, "gmm_initial_threshold", 0.95)
    logger.info(f"[GMM] initial clean threshold: {threshold:.4f}")
    pred = split_prob(prob, threshold)

    return torch.as_tensor(pred, dtype=torch.float32), torch.as_tensor(prob, dtype=torch.float32)


def fit_gmm_on_bank(loss_bank, args):
    logger = logging.getLogger("HHNC.train")
    logger.info("\n[Fast GMM] Fitting GMM from Memory Loss Bank ...")

    input_loss = _normalize_losses(loss_bank).reshape(-1, 1)
    gmm, input_np = _fit_two_component_gmm(input_loss, args)
    prob = gmm.predict_proba(input_np)[:, gmm.means_.argmin()]
    threshold = getattr(args, "gmm_bank_threshold", 0.7)
    logger.info(f"[GMM] bank clean threshold: {threshold:.4f}")
    pred = split_prob(prob, threshold)

    return torch.as_tensor(pred, dtype=torch.float32), torch.as_tensor(prob, dtype=torch.float32)


def get_real_correspondences_from_loader(train_loader):
    dataset = train_loader.dataset

    if hasattr(dataset, "real_correspondences"):
        return torch.as_tensor(dataset.real_correspondences, dtype=torch.long)

    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "real_correspondences"):
        real = dataset.dataset.real_correspondences
        if hasattr(dataset, "indices"):
            real = np.asarray(real)[dataset.indices]
        return torch.as_tensor(real, dtype=torch.long)

    return None


def compute_clean_split_quality(label_hat, real_correspondences, eps=1e-8):
    label_hat = torch.as_tensor(label_hat).long().cpu()
    real_correspondences = torch.as_tensor(real_correspondences).long().cpu()

    pred_clean = label_hat == 1
    pred_noisy = label_hat == 0
    real_clean = real_correspondences == 1
    real_noisy = real_correspondences == 0

    tp = (pred_clean & real_clean).sum().item()
    fp = (pred_clean & real_noisy).sum().item()
    fn = (pred_noisy & real_clean).sum().item()
    tn = (pred_noisy & real_noisy).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
        "pred_clean_num": pred_clean.sum().item(),
        "pred_noisy_num": pred_noisy.sum().item(),
        "real_clean_num": real_clean.sum().item(),
        "real_noisy_num": real_noisy.sum().item(),
    }


def save_split_quality_curve(history, save_dir):
    import csv
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "clean_split_quality.csv")
    fieldnames = [
        "epoch",
        "precision",
        "recall",
        "f1",
        "acc",
        "tp",
        "fp",
        "fn",
        "tn",
        "pred_clean_num",
        "pred_noisy_num",
        "real_clean_num",
        "real_noisy_num",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow({k: item.get(k, None) for k in fieldnames})

    epochs = [x["epoch"] for x in history]
    precision = [x["precision"] for x in history]
    recall = [x["recall"] for x in history]
    f1 = [x["f1"] for x in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, precision, marker="o", linewidth=2, label="Clean Precision")
    plt.plot(epochs, recall, marker="s", linewidth=2, label="Clean Recall")
    plt.plot(epochs, f1, marker="^", linewidth=2, label="Clean F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Clean Subset Split Quality")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "clean_precision_recall_curve.png"), dpi=300)
    plt.close()


def _scalar(val):
    return val.item() if isinstance(val, torch.Tensor) else val


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {"num_epoch": num_epoch, "iteration": 0}

    logger = logging.getLogger("HHNC.train")
    logger.info("start training")

    meters = {
        "loss": AverageMeter(),
        "align_loss_global": AverageMeter(),
        "hhnc_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    scaler = torch.cuda.amp.GradScaler(enabled=getattr(args, "use_amp", True))

    best_rsum = 0.0
    best_epoch = 0
    data_size = len(train_loader.dataset)

    split_quality_history = []
    real_correspondences_all = get_real_correspondences_from_loader(train_loader)
    if real_correspondences_all is not None:
        logger.info(
            f"[SplitQuality] real clean: {(real_correspondences_all == 1).sum().item()}, "
            f"real noisy: {(real_correspondences_all == 0).sum().item()}"
        )
    else:
        logger.info("[SplitQuality] real_correspondences not found, skip split quality evaluation.")

    global_hist_weights = torch.ones(data_size, dtype=torch.float32, device=device)
    loss_bank = torch.zeros(data_size, dtype=torch.float32)

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.epoch = epoch

        if epoch == start_epoch:
            pred_clean, clean_conf = get_loss(model, train_loader)
        else:
            pred_clean, clean_conf = fit_gmm_on_bank(loss_bank, args)

        label_hat = pred_clean.clone()
        pred_clean_num = int((label_hat == 1).sum().item())
        pred_noisy_num = int((label_hat == 0).sum().item())

        use_elite_memory = not getattr(args, "disable_elite_memory", False)
        clean_conf_margin = getattr(args, "elite_clean_conf_margin", 0.0)
        if use_elite_memory and pred_clean_num > 0:
            clean_conf_in_clean_set = clean_conf[label_hat == 1]
            elite_clean_threshold = float(clean_conf_in_clean_set.mean().item())
            elite_clean_mask_all = (label_hat == 1) & (clean_conf > elite_clean_threshold + clean_conf_margin)
        else:
            elite_clean_threshold = 0.0
            elite_clean_mask_all = label_hat == 1

        elite_clean_num = int(elite_clean_mask_all.sum().item())
        logger.info(f"[Split] Epoch {epoch}: pred clean={pred_clean_num}, pred noisy={pred_noisy_num}")
        logger.info(
            f"[EliteMemory] Epoch {epoch}: enabled={use_elite_memory}, "
            f"tau_dyn={elite_clean_threshold:.4f}, margin={clean_conf_margin:.4f}, "
            f"elite clean={elite_clean_num}/{pred_clean_num}"
        )

        if real_correspondences_all is not None and get_rank() == 0:
            split_metrics = compute_clean_split_quality(label_hat, real_correspondences_all)
            split_metrics["epoch"] = epoch
            split_quality_history.append(split_metrics)
            logger.info(
                "[SplitQuality] Epoch {} | Clean Precision: {:.4f} | Clean Recall: {:.4f} | "
                "Clean F1: {:.4f} | Acc: {:.4f} | TP: {} FP: {} FN: {} TN: {}".format(
                    epoch,
                    split_metrics["precision"],
                    split_metrics["recall"],
                    split_metrics["f1"],
                    split_metrics["acc"],
                    split_metrics["tp"],
                    split_metrics["fp"],
                    split_metrics["fn"],
                    split_metrics["tn"],
                )
            )
            tb_writer.add_scalar("split_quality/clean_precision", split_metrics["precision"], epoch)
            tb_writer.add_scalar("split_quality/clean_recall", split_metrics["recall"], epoch)
            tb_writer.add_scalar("split_quality/clean_f1", split_metrics["f1"], epoch)
            tb_writer.add_scalar("split_quality/split_acc", split_metrics["acc"], epoch)
            tb_writer.add_scalar("split_quality/pred_clean_num", split_metrics["pred_clean_num"], epoch)
            tb_writer.add_scalar("split_quality/pred_noisy_num", split_metrics["pred_noisy_num"], epoch)
            tb_writer.add_scalar("split_quality/elite_clean_num", elite_clean_num, epoch)
            tb_writer.add_scalar("split_quality/elite_clean_threshold", elite_clean_threshold, epoch)
            save_split_quality_curve(split_quality_history, os.path.join(args.output_dir, "split_quality"))

        model.train()
        last_ret = None
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            index = batch["index"]
            batch["label_hat"] = label_hat[index.cpu()]
            batch["clean_conf"] = clean_conf[index.cpu()]
            batch["elite_clean_mask"] = elite_clean_mask_all[index.cpu()]
            batch["hist_weights"] = global_hist_weights[index]

            ret = model(batch, epoch, args)
            last_ret = ret
            loss_bank[index.cpu()] = ret.get("loss_A_per", torch.zeros_like(index, dtype=torch.float)).detach().cpu()

            loss_items = [v for k, v in ret.items() if "loss" in k and k != "loss_A_per"]
            if len(loss_items) == 0:
                continue

            total_loss = sum(loss_items)
            batch_size = batch["images"].shape[0]
            meters["loss"].update(total_loss.item(), batch_size)
            meters["align_loss_global"].update(_scalar(ret.get("align_loss_global", 0)), batch_size)
            meters["hhnc_loss"].update(_scalar(ret.get("hhnc_loss", 0)), batch_size)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if "cur_weights" in ret:
                with torch.no_grad():
                    global_hist_weights[index] = ret["cur_weights"].detach()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        logger.info(f"global_hist_weights mean: {global_hist_weights.mean().item():.4f}")
        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)

        if last_ret is not None and "temperature" in last_ret:
            tb_writer.add_scalar("temperature", _scalar(last_ret["temperature"]), epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()

        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / max(1, n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch,
                    time_per_batch,
                    train_loader.batch_size / time_per_batch,
                )
            )

        if epoch % eval_period == 0 and get_rank() == 0:
            logger.info("Validation Results - Epoch: {}".format(epoch))
            if args.distributed:
                rsum = evaluator.eval(model.module.eval())
            else:
                rsum = evaluator.eval(model.eval())

            torch.cuda.empty_cache()
            if best_rsum <= rsum:
                best_rsum = rsum
                best_epoch = epoch
                arguments["epoch"] = epoch
                checkpointer.save("best", **arguments)

    if get_rank() == 0:
        logger.info(f"best rSum: {best_rsum} at epoch {best_epoch}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("HHNC.test")
    logger.info("Enter inferencing")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    evaluator.eval(model.eval())