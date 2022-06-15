#!/bin/bash
set -x
LS=$1
NP=$2
ARCH=$3

# python -OO test/app.py -i -a resnet50  --line-size=$LS -p --noc-ports=$NP -r $ARCH -b1
# python -OO test/app.py -i -a resnet50  --line-size=$LS -p --noc-ports=$NP -r $ARCH -b2
# python -OO test/app.py -i -a bert-large-squad-avg  --line-size=$LS -p --noc-ports=$NP -r $ARCH -b1
# python -OO test/app.py -i -a bert-large-squad-avg  --line-size=$LS -p --noc-ports=$NP -r $ARCH -b8
# python -OO test/app.py -i -a ssdrn34-1200  --line-size=$LS -p --noc-ports=$NP -r $ARCH -b1
# python -OO test/app.py -i -a ssdrn34-1200  --line-size=$LS -p --noc-ports=$NP -r $ARCH -b2
python -OO test/app.py -i -a rnnt -d FP16 --line-size=$LS -p --noc-ports=$NP -r $ARCH -b1
python -OO test/app.py -i -a rnnt -d FP16 --line-size=$LS -p --noc-ports=$NP -r $ARCH -b1024
