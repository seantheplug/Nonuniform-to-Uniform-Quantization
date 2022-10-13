clear
mkdir log
python3 train.py --data=/home/sliuau/ternary_vit/dataset/imagenet-1k/imagenet --batch_size=256 --learning_rate=1.25e-3 --epochs=128 --weight_decay=0 | tee -a log/training.txt
