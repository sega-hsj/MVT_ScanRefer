#!/usr/bin/env bash
#SBATCH --job-name=deal
#SBATCH --output=deal.txt
#SBATCH --gres=gpu:1 -c 5

python prepare_data.py --split train --scannet_path scans/ --pointgroupinst_path /mnt/proj2/sjhuang/PointGroupInst --output_path /mnt/proj2/sjhuang/scanrefer_train
