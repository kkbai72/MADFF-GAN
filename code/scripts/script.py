import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

if opt.train:
	os.system("python train.py \
		--dataroot ./LLTL/train \
		--valdataroot ./LLTL/val\
		--no_dropout \
		--name ex1\
		--which_model_netG sid_unet_resize \
        --which_model_netD no_norm_4 \
        --patchD \
        --patchD_3 5 \
        --n_layers_D 6 \
        --n_layers_patchD 4 \
		--fineSize 256 \
        --patchSize 48 \
		--skip 1 \
		--batchSize 4 \
        --self_attention \
		--use_norm 1 \
		--use_wgan 0 \
        --use_ragan \
        --hybrid_loss \
        --times_residual \
		--instance_norm 0 \
		--vgg 1 \
        --vgg_choose relu5_1 \
		--gpu_ids 0 \
		--display_port=" + opt.port)


elif opt.predict:
	for i in range(1):
	        os.system("python predict.py \
	        	--dataroot ./test_datas \
	        	--name ex1 \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode unaligned \
	        	--which_model_netG sid_unet_resize \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
                --self_attention \
				--times_residual\
	        	--instance_norm 0 \
				--resize_or_crop='no'\
	        	--which_epoch 110\
	        	 --gpu_ids 1")

# ldd /home/wyx/anaconda3/envs/pytorch_wbq/lib/python3.7/site-packages/nvidia/cublas/lib/libcublas.so.11
# export LD_LIBRARY_PATH=/home/wyx/a  naconda3/envs/pytorch_wbq/lib/python3.7/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
# python scripts/script.py --predict
# tensorboard --logdir=./checkpoints/enlightening