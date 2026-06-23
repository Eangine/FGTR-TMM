


# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.7 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --lr 1e-6 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy"
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.7 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --lr 1e-6 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy"
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.7 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --lr 1e-6 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy"



# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.5 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --lr 6e-6 --reliability_alpha 0.25 --attn_entropy_weight 0.6 --global_fusion_lambda 0.7 --pooling_beta 0.2 --reliability_gamma 1.5 --retrieval_topk 5



# # # 消融实验
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.5 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 0
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.5 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 1
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.5 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.5 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.5 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 4
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.5 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5



# # ############################################ 0610 ############################################
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 1.0 --lr 6e-6

# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 1.0 --global_fusion_lambda 1.0 --lr 6e-6

# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.0 --lr 6e-6
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.1 --lr 6e-6
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.3 --lr 6e-6
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.5 --lr 6e-6
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 7 --lr 6e-6
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6

# # # 0610 晚上 重新设置学习率再跑一次 最好的结果是 0.2 
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.4
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.6

# # ############################################ 0611 ############################################
# # # 0611 凌晨
# # # # （1）再试一下 0.1 0.3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.1
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.3

# # # #（2）基于最好的0.2 测试一下不同的Patch筛选率
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.2 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.6 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.8 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2

# # # #（3）假设select_ratio为0.4的时候最好，测试一下不同的mlm_loss_weight
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.01 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.05 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.5 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2

# # # 0611 上午
# # # （1）测试不同类型样本的初始权重 1+0.5+0.25
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 # 阈值0.95
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 # 阈值 0.9
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM  --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --ablation_mode 5 --reliability_alpha 0.0 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 # 阈值 0.95+0.7



# # # 0611 下午
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 0
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 1
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 4
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5

# # # 0611 晚上
# # #（1）检索样本的数量对模型性能的影响 3 5 10 20
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --retrieval_topk 3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --retrieval_topk 10
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --retrieval_topk 20


# # #（2）内存大小的影响
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 70000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5

# # #（3）目前来看，消融实验的结果是mode3的效果最好，因此测试一下RASC-Flickr30k-mode3的实验结果
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 3 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy"
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 3 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy"
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 3 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy"

# # #（4）正常配置下的实验结果
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy"
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy"
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy"

# # #（5）COCO数据集效果
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name COCO --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/coco/noise_inx_0.2.npy"
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name COCO --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/coco/noise_inx_0.4.npy"
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name COCO --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/coco/noise_inx_0.6.npy"



# # ############################################ 0612 ############################################
# # # # 晚上
# # # # （1）Flickr30k在没有图像数据增广下的性能 三种数据噪声 6种消融实验 变化：memory_size=10K retrieval_topk=3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 3 --ablation_mode 0
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 3 --ablation_mode 1
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 3 --ablation_mode 2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 3 --ablation_mode 3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 3 --ablation_mode 4
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 3 --ablation_mode 5

# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 3 --ablation_mode 0
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 3 --ablation_mode 1
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 3 --ablation_mode 2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 3 --ablation_mode 3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 3 --ablation_mode 4
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 3 --ablation_mode 5

# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 3 --ablation_mode 0
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 3 --ablation_mode 1
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 3 --ablation_mode 2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 3 --ablation_mode 3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 3 --ablation_mode 4
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 10000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 3 --ablation_mode 5


# # # 白天
# # # （2）Flickr30k在没有图像数据增广下的性能 三种数据噪声 6种消融实验 变化：memory_size=40K retrieval_topk=5 【真实性能】
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 5 --ablation_mode 0
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 5 --ablation_mode 1
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 5 --ablation_mode 2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 5 --ablation_mode 3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 5 --ablation_mode 4
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 5 --ablation_mode 5

# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --ablation_mode 0
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --ablation_mode 1
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --ablation_mode 2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --ablation_mode 3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --ablation_mode 4
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --ablation_mode 5

# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 5 --ablation_mode 0
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 5 --ablation_mode 1
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 5 --ablation_mode 2
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 5 --ablation_mode 3
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 5 --ablation_mode 4
# # # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 5 --ablation_mode 5




# # 0618早上，测试gamma=1 gamma=0.8的效果（之前设置为1.5，突出最牛逼的Tokens）
# # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 3 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --reliability_gamma 1
# # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 4 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --reliability_gamma 1
# # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --reliability_gamma 1

# # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 3 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --reliability_gamma 0.8
# # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 4 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --reliability_gamma 0.8
# # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0.2 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --reliability_gamma 0.8
# # 结论：有阶梯式的性能提升，但是每次基本就是一个点，所以说gamma没啥用，其实可以删掉了。


# # pooling_beta=0 完全依赖Tokens可信度
# # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.9 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5
# # 结果：不设置可信度下限，模型性能近似，550.1（550），也是一个冗余设计？ | 74.06 | 92.98 | 96.48 | 82.47 | 82.47 | 263.52 | 88.80 | 98.30 | 99.50 | 77.17 | 55.33 | 286.60 |=550.1
# # global_fusion_lambda低一些，增加校准局部表征的作用试一下
# # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.7 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5
# # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.5 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5
# ## 总结一下：pooling_beta去掉，不需要设置下限；gamma去掉，不需要再次调节权重；
# ## global_fusion_lambda这个影响较大，过小了使用局部特征加权融合会显著影响全局特征，导致特征漂移，影响性能；
# ## 结论：图像的增广确实有必要，没有aug，性能直接衰退。


# # 性能对比实验


# # 消融实验
# # 检索语义原型-细粒度Token加权-图像Patch筛选 全局局部对齐 GMM样本划分
# # 都当做干净样本直接对齐【基础baseline】
# # gmm划分，只使用干净样本
# # （1）只使用全局相似度——分成两种
# #     a、对齐损失
# #     b、对齐损失+MLM损失
# # （2）同时使用全局、局部相似度——分成三种
# #     a、只使用
# # 全局、局部划分：
# # 模糊样本：校准，
# # 模糊样本：校准，噪声样本：检索
# # 模糊样本：校准，噪声样本：检索+校准


# # 有一个点：模糊样本是否有必要？
# # 干净样本，对齐损失+MLM损失
# # 噪声样本：内存中找语义原型，基于交叉注意力的Token加权，局部校准全局，然后在对齐。
# # 如果没有：基于全局相似度样本划分，得到干净&噪声样本。不再有局部相似度加权损失，校准特征通过全局特征优化网络。



# # python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.5 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5

# # 完全使用校准的局部表征作为全局表征进行训练
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5

# # global_fusion_lambda=0.1
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.1 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5

# # 完全使用检索到的文本全局表征进行训练
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5


# #### n-0.6
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 5

# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --retrieval_topk 5


# 0620晚上-新版的Token加权机制
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.2 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.4 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.6 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 0.8 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5

# 0621-测试一下不同的噪声先验权重的影响
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --noisy_prior 0.5
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --noisy_prior 0.75
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --noisy_prior 0

# 0621-下午-基于熵的伪文本加权
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --mlm_loss_weight 0.1 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_logit_scale --conf_use_retrieval_sim


# mlm损失权重过大了，加不加噪声样本区别不大
# python train.py --txt_aug --img_aug --loss_names InfoNCE --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --noisy_prior 0
# python train.py --txt_aug --img_aug --loss_names InfoNCE --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --noisy_prior 0.5
# python train.py --txt_aug --img_aug --loss_names InfoNCE --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --noisy_prior 1

# python train.py --txt_aug --img_aug --loss_names InfoNCE --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_logit_scale --conf_use_retrieval_sim

# python train.py --txt_aug --img_aug --loss_names InfoNCE --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --noisy_prior 0.25
# python train.py --txt_aug --img_aug --loss_names InfoNCE --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --noisy_prior 0.75


## 查找原因
# 1 测试一下noisy_prior=5
# python train.py --txt_aug --img_aug --loss_names InfoNCE --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# 2 测试一下不Warm，没有MLM头了，似乎不需要热身了
# python train.py --txt_aug --img_aug --loss_names InfoNCE --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale




# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale --enable_visualization --case_vis_enable --case_vis_interval 200 --case_vis_start_epoch 6 --case_vis_max_per_iter 4

# # 终章
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/f30k/noise_inx_0.2.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/f30k/noise_inx_0.6.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale

# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name CC120K --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 6e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale

# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name COCO --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.2 --noisy_file "./noise_file/coco/noise_inx_0.2.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name COCO --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/coco/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name COCO --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.6 --noisy_file "./noise_file/coco/noise_inx_0.6.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale

# 超参数敏感性分析

# 图片掩码比率【0.2 0.4 0.6 0.8 1.0】
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.2 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.6 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.8 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 1.0 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale

# MB大小&检索数量【20k 40k 60k 80k】【3 5 7 9】
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 20000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 60000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 80000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 100000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale

python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 1 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 3 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 5 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 7 --conf_use_retrieval_sim --conf_use_logit_scale
# python train.py --txt_aug --img_aug --loss_names InfoNCE+MLM --warm_up --memory_size 40000 --select_ratio 0.4 --gpu 0 --dataset_name Flickr30k --warmup_epochs 5 --num_epoch 10 --global_fusion_lambda 1.0 --lr 1e-6 --attn_entropy_weight 0.5 --pooling_beta 0 --ablation_mode 5 --noisy_rate 0.4 --noisy_file "./noise_file/f30k/noise_inx_0.4.npy" --retrieval_topk 9 --conf_use_retrieval_sim --conf_use_logit_scale


