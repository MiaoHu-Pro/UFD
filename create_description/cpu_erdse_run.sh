#!/bin/bash
#SBATCH --job-name=create_des
#SBATCH --time=3:00:00
#SBATCH --output=create_des_for_ent_rel-%j.out
#SBATCH --partition=k2-hipri




export PYTHONPATH=$PYTHONPATH:/users/40305887/gridware/share/python/3.6.4/lib/python3.6/site-packages


echo "the job start... "
python3 ERDse.py 
echo "the job end !  "




