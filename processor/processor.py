
import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt
from pylab import xticks,yticks,np
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import gc
import scipy.stats as stats


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)

def get_loss(model, data_loader):
    logger = logging.getLogger("RASC.train")
    model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()

    lossA = torch.zeros(data_size, dtype=torch.float32)   # 只保留全局
    torch.cuda.empty_cache()

    for i, batch in enumerate(data_loader):
        index = batch['index'].clone()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            la = model.compute_per_loss(batch)        # 现在只返回全局 loss
            lossA[index.cpu()] = la.detach().cpu().float()

        del batch, la, index
        if i % 200 == 0:
            logger.info(f'compute loss batch {i}')
            torch.cuda.empty_cache()
            gc.collect()

        # if i==200:
        #     break

    losses_A = (lossA - lossA.min()) / (lossA.max() - lossA.min())
    input_loss_A = losses_A.reshape(-1, 1)

    logger.info('\nFitting GMM ...')

    if model.args.noisy_rate > 0.4 or model.args.dataset_name == 'RSTPReid':
        gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    max_samples = 100000
    input_A_np = input_loss_A.cpu().numpy()
    if len(input_A_np) > max_samples:
        idx = np.random.choice(len(input_A_np), max_samples, replace=False)
        gmm_A.fit(input_A_np[idx])
    else:
        gmm_A.fit(input_A_np)

    prob_A = gmm_A.predict_proba(input_A_np)[:, gmm_A.means_.argmin()]
    threshold = getattr(model.args, "gmm_initial_threshold", 0.95)
    logger.info(f"[GMM] initial clean threshold: {threshold:.4f}")
    pred_A = split_prob(prob_A, threshold)

    return torch.as_tensor(pred_A, dtype=torch.float32), torch.as_tensor(prob_A, dtype=torch.float32)

def fit_gmm_on_bank(lossA, args):
    logger = logging.getLogger("RASC.train")
    logger.info('\n[Fast GMM] Fitting GMM from Memory Loss Bank ...')

    losses_A = (lossA - lossA.min()) / (lossA.max() - lossA.min())
    input_loss_A = losses_A.reshape(-1, 1)

    if args.noisy_rate > 0.4 or args.dataset_name == 'RSTPReid':
        gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    max_samples = 100000
    input_A_np = input_loss_A.cpu().numpy()
    if len(input_A_np) > max_samples:
        idx = np.random.choice(len(input_A_np), max_samples, replace=False)
        gmm_A.fit(input_A_np[idx])
    else:
        gmm_A.fit(input_A_np)

    prob_A = gmm_A.predict_proba(input_A_np)[:, gmm_A.means_.argmin()]
    threshold = getattr(args, "gmm_bank_threshold", 0.7)
    logger.info(f"[GMM] bank clean threshold: {threshold:.4f}")
    pred_A = split_prob(prob_A, threshold)

    return torch.as_tensor(pred_A, dtype=torch.float32), torch.as_tensor(prob_A, dtype=torch.float32)

def get_real_correspondences_from_loader(train_loader):
    """
    从 train_loader.dataset 中读取真实 clean/noisy 标签。
    real_correspondences:
        1 表示真实 clean pair
        0 表示真实 noisy pair
    """
    dataset = train_loader.dataset

    # 常规情况：ImageTextDataset
    if hasattr(dataset, "real_correspondences"):
        real = dataset.real_correspondences
        return torch.as_tensor(real, dtype=torch.long)

    # 兼容 torch.utils.data.Subset
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "real_correspondences"):
        real = dataset.dataset.real_correspondences
        if hasattr(dataset, "indices"):
            real = np.asarray(real)[dataset.indices]
        return torch.as_tensor(real, dtype=torch.long)

    return None


def compute_clean_split_quality(label_hat, real_correspondences, eps=1e-8):
    """
    统计 clean 子集划分质量。

    Args:
        label_hat: Tensor[N], 1 predicted clean, 0 predicted noisy
        real_correspondences: Tensor[N], 1 real clean, 0 real noisy

    Returns:
        dict
    """
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

    clean_acc = (tp + tn) / (tp + fp + fn + tn + eps)

    pred_clean_num = pred_clean.sum().item()
    pred_noisy_num = pred_noisy.sum().item()
    real_clean_num = real_clean.sum().item()
    real_noisy_num = real_noisy.sum().item()

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": clean_acc,
        "pred_clean_num": pred_clean_num,
        "pred_noisy_num": pred_noisy_num,
        "real_clean_num": real_clean_num,
        "real_noisy_num": real_noisy_num,
    }

def save_split_quality_curve(history, save_dir):
    """
    保存每个 epoch 的 clean precision / recall / f1 曲线和 csv。
    """
    import os
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

    fig_path = os.path.join(save_dir, "clean_precision_recall_curve.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

def do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("RASC.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "align_loss_global": AverageMeter(),
        'mlm_loss': AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_rsum = 0.0
    data_size = train_loader.dataset.__len__()

    split_quality_history = []
    real_correspondences_all = get_real_correspondences_from_loader(train_loader)

    if real_correspondences_all is not None:
        logger.info(
            f"[SplitQuality] real clean: {(real_correspondences_all == 1).sum().item()}, "
            f"real noisy: {(real_correspondences_all == 0).sum().item()}"
        )
    else:
        logger.info("[SplitQuality] real_correspondences not found, skip split quality evaluation.")


    global_hist_weights = torch.ones(data_size, dtype=torch.float32, device='cuda')

    lossA_bank = torch.zeros(data_size, dtype=torch.float32)

    # evaluator.eval(model.eval())
    # train
    sims = []
    for epoch in range(start_epoch, num_epoch + 1):

        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.epoch = epoch

        if epoch == start_epoch:
            pred_A, clean_conf = get_loss(model, train_loader)
        else:
            pred_A, clean_conf = fit_gmm_on_bank(lossA_bank, args)

        label_hat = pred_A.clone()

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
            elite_clean_mask_all = (label_hat == 1)

        elite_clean_num = int(elite_clean_mask_all.sum().item())

        print(f"clean: {pred_clean_num}, noisy: {pred_noisy_num}")
        logger.info(f"[Split] Epoch {epoch}: pred clean={pred_clean_num}, pred noisy={pred_noisy_num}")
        logger.info(
            f"[EliteMemory] Epoch {epoch}: enabled={use_elite_memory}, "
            f"tau_dyn={elite_clean_threshold:.4f}, margin={clean_conf_margin:.4f}, "
            f"elite clean={elite_clean_num}/{pred_clean_num}"
        )

        if real_correspondences_all is not None and get_rank() == 0:
            split_metrics = compute_clean_split_quality(
                label_hat=label_hat,
                real_correspondences=real_correspondences_all
            )
            split_metrics["epoch"] = epoch
            split_quality_history.append(split_metrics)

            logger.info(
                "[SplitQuality] Epoch {} | "
                "Clean Precision: {:.4f} | "
                "Clean Recall: {:.4f} | "
                "Clean F1: {:.4f} | "
                "Acc: {:.4f} | "
                "TP: {} FP: {} FN: {} TN: {}".format(
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

            split_quality_dir = os.path.join(args.output_dir, "split_quality")
            save_split_quality_curve(split_quality_history, split_quality_dir)

       
        model.train()
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if (
                getattr(args, "case_vis_enable", False)
                and get_rank() == 0
                and epoch >= getattr(args, "case_vis_start_epoch", 1)
                and (n_iter % getattr(args, "case_vis_interval", 500) == 0)
            ):
                batch["vis_request"] = {
                    "save_dir": getattr(args, "vis_dir", os.path.join(args.output_dir, "noise_case_vis")),
                    "epoch": epoch,
                    "iter": n_iter,
                    "max_cases": getattr(args, "case_vis_max_per_iter", 4),
                    "token_conf_thr": getattr(args, "case_vis_token_conf_thr", 0.3),
                }

            index = batch['index']
            batch['label_hat'] = label_hat[index.cpu()]
            batch['clean_conf'] = clean_conf[index.cpu()]
            batch['elite_clean_mask'] = elite_clean_mask_all[index.cpu()]
            batch['hist_weights'] = global_hist_weights[index]
            ret = model(batch, epoch, args)

            # if n_iter == 501:
            #     break
            
            loss_A_val = ret.get('loss_A_per', torch.zeros_like(index, dtype=torch.float)).detach().cpu()
            lossA_bank[index.cpu()] = loss_A_val
            loss_items = [v for k, v in ret.items() if "loss" in k and k not in ["loss_A_per"]]
            
            if len(loss_items) == 0:
                continue
                
            total_loss = sum(loss_items)

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            def get_scalar(val):
                return val.item() if isinstance(val, torch.Tensor) else val
            meters['align_loss_global'].update(get_scalar(ret.get('align_loss_global', 0)), batch_size)
            meters['mlm_loss'].update(get_scalar(ret.get('mlm_loss', 0)), batch_size)

            # Backward
            scaler.scale(total_loss).backward()
            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if 'cur_weights' in ret:
                with torch.no_grad():
                    global_hist_weights[index] = ret['cur_weights'].detach()
        
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        print("global_hist_weights Mean:", global_hist_weights.mean())
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)

        if 'temperature' in ret:
            tb_writer.add_scalar('temperature', get_scalar(ret['temperature']), epoch)
            
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        if get_rank() == 0:
            swan_logs = {
                "train/lr": scheduler.get_lr()[0],
            }
            if 'temperature' in ret:
                swan_logs["train/temperature"] = get_scalar(ret['temperature'])
                
            for k, v in meters.items():
                if v.avg > 0:
                    swan_logs[f"train/{k}"] = v.avg

        scheduler.step()
        
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
            del ret, loss_items, total_loss, batch
                        
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    rsum = evaluator.eval(model.module.eval())
                else:
                    rsum = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_rsum <= rsum:
                    best_rsum = rsum
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

    if get_rank() == 0:
        logger.info(f"best rSum: {best_rsum} at epoch {arguments['epoch']}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)

def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("RASC.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
