#!/usr/bin/env bash
#在Cityscapes上训练

python -m torch.distributed.launch --nproc_per_node=2 tools/train.py \
        --cfg experiments/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml