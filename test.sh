#!/usr/bin/env bash
#在Cityscapes验证集vail上测试
resume="output/focal_loss_alpha0_5_gamma1/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/best.pth"

python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     TEST.MODEL_FILE $resume \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True       