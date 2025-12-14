IMAGE_NAME=head_pose_estimation_sixdrepnet
NOW=`date +"%Y%m%d%I%M%S"`

build:
	docker build -t ${IMAGE_NAME} .

start:
	docker run --rm -it --name ${IMAGE_NAME}-${NOW} \
						-v ${PWD}:/workspace \
						-e WANDB_API_KEY=${WANDB_API_KEY} \
						--gpus all \
						--shm-size 4gb \
						${IMAGE_NAME} bash

train:
	docker run --rm -it --name ${IMAGE_NAME}-${NOW} \
						-v ${PWD}:/workspace \
						-e WANDB_API_KEY=${WANDB_API_KEY} \
						--gpus all \
						--shm-size 4gb \
						${IMAGE_NAME} python sixdrepnet/train.py

evaluate:
	docker run --rm -it --name ${IMAGE_NAME}-eval-${NOW} \
						-v ${PWD}:/workspace \
						--gpus all \
						--shm-size 4gb \
						${IMAGE_NAME} python sixdrepnet/evaluate_epochs.py \
						--snapshot_dir output/snapshots/SixDRepNet_1765605639_bs64 \
						--aflw2000_dir datasets/AFLW2000 \
						--aflw2000_filelist datasets/AFLW2000/files.txt \
						--biwi_file datasets/BIWI/BIWI_noTrack.npz \
						--output_dir output/evaluation
