import argparse

def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")

    parser.add_argument("--warm_up", default=False, action='store_true')
    parser.add_argument("--memory_size", default=4096, type=int)
    parser.add_argument("--disable_elite_memory", default=False, action='store_true')
    parser.add_argument("--elite_clean_conf_margin", default=0.0, type=float)
    parser.add_argument("--noisy_rate", default=0.4, type=float)
    parser.add_argument("--gmm_initial_threshold", default=0.95, type=float)
    parser.add_argument("--gmm_bank_threshold", default=0.7, type=float)
    # parser.add_argument("--noisy_file", default='', type=str)
    parser.add_argument("--noisy_file", default='./noise_file/f30k/noise_inx_0.4.npy', type=str)
    parser.add_argument("--gpu", type=int, default=['0'])
    
    parser.add_argument("--mlm_loss_weight", type=float, default=0)
    parser.add_argument("--select_ratio", default=0.5, type=float)
    parser.add_argument('--retrieval_topk', type=int, default=5)

    parser.add_argument('--ambiguous_prior', type=float, default=0.5)
    parser.add_argument('--noisy_prior', type=float, default=0.25)
    
    parser.add_argument('--min_bank_size', type=int, default=1024)
    parser.add_argument('--reliability_gamma', type=float, default=1)
    parser.add_argument('--pooling_beta', type=float, default=0.0)
    parser.add_argument('--attn_entropy_weight', type=float, default=0.5)
    parser.add_argument('--global_fusion_lambda', type=float, default=0.9)
    parser.add_argument('--ntr_chunk_size', type=int, default=512)
    parser.add_argument('--ablation_mode', type=int, default=5)

    parser.add_argument('--prototype_use_calibration', type=bool, default=True)
    parser.add_argument("--use_selected_patches_for_calibration", default=False, action='store_true')
    parser.add_argument("--local_use_token_weight", default=False, action='store_true')

    parser.add_argument("--enable_visualization", action="store_true")
    parser.add_argument("--vis_dir", default="./token_vis")
    parser.add_argument("--vis_interval", type=int, default=200)

    parser.add_argument("--conf_use_logit_scale", default=False, action='store_true')
    parser.add_argument("--conf_use_retrieval_sim", default=False, action='store_true')   

    parser.add_argument("--case_vis_enable", action="store_true", help="enable noisy case visualization")
    parser.add_argument("--case_vis_interval", type=int, default=500, help="visualize every N iterations")
    parser.add_argument("--case_vis_start_epoch", type=int, default=1, help="start epoch for case visualization")
    parser.add_argument("--case_vis_max_per_iter", type=int, default=4, help="max noisy cases saved per visualization iteration")
    parser.add_argument("--case_vis_token_conf_thr", type=float, default=0.3, help="threshold for constructing reliable text from token confidence")
    parser.add_argument("--return_img_path", action="store_true")
    parser.add_argument("--return_raw_caption", action="store_true")

    parser.add_argument("--profile_complexity", action="store_true", help="only profile model complexity and exit")
    parser.add_argument("--profile_batch_size", type=int, default=64)
    parser.add_argument("--profile_warmup", type=int, default=20)
    parser.add_argument("--profile_iters", type=int, default=100)

 

    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="RASC", help="experiment name to save")
    parser.add_argument("--output_dir", default="RASC_Result/")
    parser.add_argument("--log_period", default=500)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="val") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/32')
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')
    parser.add_argument("--txt_aug", default=False, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=8, help="cross modal transformer self attn layers")
    parser.add_argument("--lr_factor", type=float, default=10.0, help="lr factor for random init self implement module")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='InfoNCE+MLM', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")

    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(224, 224))
    parser.add_argument("--stride_size", type=int, default=32)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="Flickr30k", help="[Flickr30k, COCO, CC120K]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="/home/yj/eangine/data/TIReID")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=507) # flickr: val507
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args



