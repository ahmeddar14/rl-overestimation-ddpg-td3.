#!/bin/zsh
cd ~/rl-project
cat jobs.txt | xargs -P 9 -L 1 zsh -c '.venv/bin/python -W ignore exp.py --algo $0 --ln $1 --depth $2 --seed $3 --steps 300000 --out results > logs/$0_ln$1_d$2_s$3.log 2>&1'
echo ALLDONE
