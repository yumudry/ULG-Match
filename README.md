## Usage

### Train

```
python train.py --dataset cifar10 
```

```
python -m torch.distributed.launch --nproc_per_node 4 ./train.py 
```

### Monitoring training progress
```
tensorboard --logdir=<your out_dir>
```

## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)

