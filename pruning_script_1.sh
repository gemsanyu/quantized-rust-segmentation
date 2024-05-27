#!/bin/bash

python main.py --arch fpn --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2
python main.py --arch pan --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2
python main.py --arch deeplabv3 --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2
python main.py --arch unet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2
python main.py --arch manet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2
python main.py --arch linknet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2
python main.py --arch pspnet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2
python main.py --arch unet++ --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2
python main.py --arch deeplabv3+ --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2


python main.py --arch fpn --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.5
python main.py --arch pan --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.5
python main.py --arch deeplabv3 --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.5
python main.py --arch unet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.5
python main.py --arch manet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.5
python main.py --arch linknet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.5
python main.py --arch pspnet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.5
python main.py --arch unet++ --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.5
python main.py --arch deeplabv3+ --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 500 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.5