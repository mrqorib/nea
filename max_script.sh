#!/bin/bash

mi=$1

mkdir -p log/regress/$mi

for i in {1..5}
do
	((f= $i - 1))
	for p in {1..8}; do (
		gpu=$p
		if [ $p -gt 4 ]
		then
			((gpu=8 - $p))
		fi
		CUDA_VISIBLE_DEVICES=$gpu THEANO_FLAGS="device=gpu0,floatX=float32,nvcc.flags=-D_FORCE_INLINES " nohup python train_nea.py --emb release/En_vectors.txt -tr data/fold_$f/train.tsv -tu data/fold_$f/dev.tsv -ts data/fold_$f/test.tsv -p $p -o "output/regress/${mi}/${p}_${f}" -mi $mi -rs "${p}${f}"> "log/regress/${mi}/out_${p}_${f}"
	) & done
	wait
done