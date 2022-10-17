NETWORK=resnet18
N_BIT=2
QUAN_DOWNSAMPLE=0
LOG_DIR=log/${NETWORK}_${N_BIT}bit_quantize_downsample_${QUAN_DOWNSAMPLE}
mkdir -p ${LOG_DIR}
### lr = original_lr*0.01*0.5 
python3 train_noise_inj.py --data=/home/sliuau/ternary_vit/dataset/imagenet-1k/imagenet \
--batch_size=512 --weight_decay=0 --student=${NETWORK} --n_bit=${N_BIT} \
--random_method 'bin_center_less_aggro' \
--random_prob 0.0 \
--boundaryRange 0.0005 \
--sample_time 20 \
--learning_rate 0.0000125 \
--epochs 1 \
--output_dir /home/sliuau/nu2uq/resnet/resnet_noise_inj/bin_center_less_aggro/ \
--quantize_downsample=${QUAN_DOWNSAMPLE} --load_pretrained_weight_student /home/sliuau/nu2uq/resnet/models/res18-2bit_star.pth.tar | tee -a ${LOG_DIR}/training.txt
### lr = original_lr*0.01*2
python3 train_noise_inj.py --data=/home/sliuau/ternary_vit/dataset/imagenet-1k/imagenet \
--batch_size=512 --weight_decay=0 --student=${NETWORK} --n_bit=${N_BIT} \
--random_method 'bin_center_less_aggro' \
--random_prob 0.0 \
--boundaryRange 0.0005 \
--sample_time 20 \
--learning_rate 0.00005 \
--epochs 1 \
--output_dir /home/sliuau/nu2uq/resnet/resnet_noise_inj/bin_center_less_aggro/ \
--quantize_downsample=${QUAN_DOWNSAMPLE} --load_pretrained_weight_student /home/sliuau/nu2uq/resnet/models/res18-2bit_star.pth.tar | tee -a ${LOG_DIR}/training.txt
