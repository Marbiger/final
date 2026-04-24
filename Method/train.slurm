#!/bin/bash
#SBATCH -J train
#SBATCH -p normal
#SBATCH --gres=gpu:1                    
#SBATCH --nodelist=sist_gpu63
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err
#SBATCH --time=3-00:00:00         

cd /public/home/xuchg2022/GeoText-1652/Method
source ~/.bashrc
conda activate gt
export TORCH_FX_TRACEBACK=0
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8
export TORCH_FX_TRACEBACK=0
python3 run.py --task "re_bbox" --dist "gpu0" --output_dir "output/llm_train" --checkpoint "/public/home/xuchg2022/GeoText-1652/GeoText1652_model/geotext_official_checkpoint.pth"