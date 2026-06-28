import os
import csv
import time
import torch
import torch.nn as nn


def format_number(num):
    if num >= 1e9:
        return f"{num / 1e9:.3f}G"
    if num >= 1e6:
        return f"{num / 1e6:.3f}M"
    if num >= 1e3:
        return f"{num / 1e3:.3f}K"
    return str(num)


def count_total_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
    }


def count_module_params(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def count_active_params(model):
    model_module = model.module if hasattr(model, "module") else model

    active = 0
    details = {}

    if hasattr(model_module, "base_model"):
        n = count_module_params(model_module.base_model)
        active += n
        details["base_model"] = n

    if hasattr(model_module, "logit_scale"):
        n = model_module.logit_scale.numel()
        active += n
        details["logit_scale"] = n

    if getattr(model_module, "enable_hhnc", False) and hasattr(model_module, "hhnc_calibrator"):
        n = count_module_params(model_module.hhnc_calibrator)
        active += n
        details["hhnc_calibrator"] = n

    return {
        "active_params": active,
        "active_param_details": details,
    }

def count_memory_bank_size(model):
    """
    MemoryBank 鏄?buffer锛屼笉灞炰簬鍙傛暟閲忋€?    浣嗗彲浠ョ粺璁″叾鏄惧瓨/鍐呭瓨鍗犵敤銆?    """
    model_module = model.module if hasattr(model, "module") else model

    if not hasattr(model_module, "memory_bank"):
        return {
            "memory_bank_numel": 0,
            "memory_bank_bytes": 0,
        }

    mb = model_module.memory_bank
    total_numel = 0
    total_bytes = 0

    for _, buf in mb.named_buffers():
        total_numel += buf.numel()
        total_bytes += buf.numel() * buf.element_size()

    return {
        "memory_bank_numel": total_numel,
        "memory_bank_bytes": total_bytes,
    }


class ImageEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.module if hasattr(model, "module") else model

    def forward(self, images):
        cls_img, _ = self.model.encode_image(images)
        return cls_img


class TextEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.module if hasattr(model, "module") else model

    def forward(self, caption_ids):
        cls_txt, _ = self.model.encode_text(caption_ids)
        return cls_txt


@torch.no_grad()
def build_dummy_inputs(args, device="cuda", batch_size=1):
    img_size = getattr(args, "img_size", (384, 128))

    if isinstance(img_size, int):
        h, w = img_size, img_size
    elif isinstance(img_size, (list, tuple)):
        h, w = img_size[0], img_size[1]
    else:
        h, w = 384, 128

    text_length = getattr(args, "text_length", 77)

    images = torch.randn(batch_size, 3, h, w, device=device)

    caption_ids = torch.zeros(batch_size, text_length, dtype=torch.long, device=device)
    caption_ids[:, 0] = 49406  # <|startoftext|>
    if text_length > 2:
        caption_ids[:, 1:text_length - 1] = 100
    caption_ids[:, text_length - 1] = 49407  # <|endoftext|>

    return images, caption_ids


def compute_inference_flops(model, args, device="cuda"):
    """
    缁熻鎺ㄧ悊闃舵 FLOPs锛?    - image encoder FLOPs
    - text encoder FLOPs

    闇€瑕佸畨瑁咃細
        pip install fvcore

    娉ㄦ剰锛?    fvcore 瀵归儴鍒嗚嚜瀹氫箟 op 鍙兘鏃犳硶缁熻锛岀粨鏋滃彲浣滀负杩戜技浼拌銆?    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("[Complexity] fvcore is not installed. Please run: pip install fvcore")
        return {
            "image_flops": None,
            "text_flops": None,
            "total_inference_flops": None,
        }

    model.eval()
    images, caption_ids = build_dummy_inputs(args, device=device, batch_size=1)

    image_wrapper = ImageEncoderWrapper(model).to(device).eval()
    text_wrapper = TextEncoderWrapper(model).to(device).eval()

    try:
        image_flops = FlopCountAnalysis(image_wrapper, images).total()
    except Exception as e:
        print(f"[Complexity] Failed to compute image FLOPs: {e}")
        image_flops = None

    try:
        text_flops = FlopCountAnalysis(text_wrapper, caption_ids).total()
    except Exception as e:
        print(f"[Complexity] Failed to compute text FLOPs: {e}")
        text_flops = None

    if image_flops is not None and text_flops is not None:
        total = image_flops + text_flops
    else:
        total = None

    return {
        "image_flops": image_flops,
        "text_flops": text_flops,
        "total_inference_flops": total,
    }


@torch.no_grad()
def benchmark_inference_latency(
    model,
    args,
    device="cuda",
    batch_size=64,
    warmup=20,
    iters=100,
):
    """
    缁熻鎺ㄧ悊 latency銆?    鍒嗗埆缁熻 image encoder 鍜?text encoder 鐨?batch latency銆?    """
    model.eval()

    images, caption_ids = build_dummy_inputs(args, device=device, batch_size=batch_size)

    image_wrapper = ImageEncoderWrapper(model).to(device).eval()
    text_wrapper = TextEncoderWrapper(model).to(device).eval()

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # image warmup
    for _ in range(warmup):
        _ = image_wrapper(images)
    sync()

    start = time.time()
    for _ in range(iters):
        _ = image_wrapper(images)
    sync()
    image_time = (time.time() - start) / iters

    # text warmup
    for _ in range(warmup):
        _ = text_wrapper(caption_ids)
    sync()

    start = time.time()
    for _ in range(iters):
        _ = text_wrapper(caption_ids)
    sync()
    text_time = (time.time() - start) / iters

    return {
        "infer_batch_size": batch_size,
        "image_latency_per_batch": image_time,
        "text_latency_per_batch": text_time,
        "image_latency_per_sample": image_time / batch_size,
        "text_latency_per_sample": text_time / batch_size,
    }


def save_complexity_report(report, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = list(report.keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(report)


def profile_model_complexity(
    model,
    args,
    save_dir,
    device="cuda",
    batch_size=64,
    warmup=20,
    iters=100,
):
    total_info = count_total_params(model)
    active_info = count_active_params(model)
    mb_info = count_memory_bank_size(model)
    flops_info = compute_inference_flops(model, args, device=device)
    latency_info = benchmark_inference_latency(
        model,
        args,
        device=device,
        batch_size=batch_size,
        warmup=warmup,
        iters=iters,
    )

    report = {}

    report["dataset_name"] = getattr(args, "dataset_name", "unknown")
    report["img_size"] = str(getattr(args, "img_size", "unknown"))
    report["text_length"] = getattr(args, "text_length", 77)

    report["total_params"] = total_info["total_params"]
    report["total_params_M"] = total_info["total_params"] / 1e6
    report["trainable_params"] = total_info["trainable_params"]
    report["trainable_params_M"] = total_info["trainable_params"] / 1e6

    report["active_params"] = active_info["active_params"]
    report["active_params_M"] = active_info["active_params"] / 1e6

    report["memory_bank_bytes"] = mb_info["memory_bank_bytes"]
    report["memory_bank_MB"] = mb_info["memory_bank_bytes"] / 1024 / 1024

    image_flops = flops_info["image_flops"]
    text_flops = flops_info["text_flops"]
    total_flops = flops_info["total_inference_flops"]

    report["image_flops"] = image_flops if image_flops is not None else -1
    report["image_flops_G"] = image_flops / 1e9 if image_flops is not None else -1

    report["text_flops"] = text_flops if text_flops is not None else -1
    report["text_flops_G"] = text_flops / 1e9 if text_flops is not None else -1

    report["total_inference_flops"] = total_flops if total_flops is not None else -1
    report["total_inference_flops_G"] = total_flops / 1e9 if total_flops is not None else -1

    report.update(latency_info)

    save_path = os.path.join(save_dir, "complexity_report.csv")
    save_complexity_report(report, save_path)

    print("=" * 80)
    print("[Complexity Report]")
    print(f"Total Params: {report['total_params_M']:.3f} M")
    print(f"Active Params: {report['active_params_M']:.3f} M")
    print(f"Memory Bank: {report['memory_bank_MB']:.3f} MB")
    print(f"Image FLOPs: {report['image_flops_G']:.3f} G")
    print(f"Text FLOPs: {report['text_flops_G']:.3f} G")
    print(f"Total Inference FLOPs: {report['total_inference_flops_G']:.3f} G")
    print(f"Image latency / sample: {report['image_latency_per_sample'] * 1000:.3f} ms")
    print(f"Text latency / sample: {report['text_latency_per_sample'] * 1000:.3f} ms")
    print(f"Saved to: {save_path}")
    print("=" * 80)

    return report
