# Train ViT on CIFAR10 with [PyTorch-Ignite](https://github.com/pytorch/ignite)


We define ViT models adapted for CIFAR10 images of 32x32 size:
- vit_tiny_patch4_32x32 : 32x32 input size and patch of 4 pixels
- vit_b4_32x32 : our base ViT reimplementation with 32x32 input size and patch of 4 pixels
- vit_b3_32x32 : our base ViT reimplementation with 32x32 input size and patch of 3 pixels
- vit_b2_32x32 : our base ViT reimplementation with 32x32 input size and patch of 2 pixels
- timm_vit_b4_32x32 : timm reimplementation of base ViT with 32x32 input size and patch of 4 pixels

## Requirements:

- [pytorch-ignite](https://github.com/pytorch/ignite): `pip install pytorch-ignite`
- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`
- [tensorboardx](https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
- [python-fire](https://github.com/google/python-fire): `pip install fire`
- (Optional) pytorch-image-models: `pip install timm`

## Usage:

Run the example on a single GPU:

```bash
python main.py run
```

For details on accepted arguments:

```bash
python main.py run -- --help
```

If user would like to provide already downloaded dataset, the path can be setup in parameters as

```bash
--data_path="/path/to/cifar10/"
```

### Distributed training

#### Single node, multiple GPUs

Let's start training on a single node with 2 gpus:

```bash
# using torch.distributed.launch
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py run --backend="nccl"
```

##### Using [Horovod](https://horovod.readthedocs.io/en/latest/index.html) as distributed backend

Please, make sure to have Horovod installed before running.

Let's start training on a single node with 2 gpus:

```bash
# horovodrun
horovodrun -np=2 python -u main.py run --backend="horovod"
```

or

```bash
# using function spawn inside the code
python -u main.py run --backend="horovod" --nproc_per_node=2
```

### TODO:

- [X] Resize bicubic to 224 x 224
- [ ] Rand-Augment
- [X] CutMix 
- [ ] MixUp ?
- [X] Erasing
- [ ] Repeated Augmentations
- [ ] Stochastic Depth
- [ ] Grab HP from paper


### CIFAR10 32x32 Online logs

On TensorBoard.dev: https://tensorboard.dev/experiment/14IXJkjzT8OEagHAl5os2w

```
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py run --backend="nccl" --num_epochs=100 --model="vit_tiny_patch4_32x32" --output_path=/output/output-cifar10-vit --with_amp --with_pbar


python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py run --backend="nccl" --num_epochs=100 --model="vit_tiny_patch2_32x32" --output_path=/output/output-cifar10-vit --with_amp --with_pbar


python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py run --backend="nccl" --num_epochs=100 --model="vit_b4_32x32" --output_path=/output/output-cifar10-vit --with_amp --with_pbar


python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py run --backend="nccl" --num_epochs=100 --model="timm_vit_b4_32x32" --output_path=/output/output-cifar10-vit --with_amp --with_pbar
```

```
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py run --backend="nccl" --num_epochs=200 --model="vit_tiny_patch4_32x32" --output_path=/output/output-cifar10-vit --with_amp --with_pbar --num_warmup_epochs=100 --learning_rate=0.001 --weight_decay=1e-3
```

To debug:
```
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py run --backend="nccl" --num_epochs=1 --model="vit_tiny_patch4_32x32" --optimizer="adamw" --output_path=/output/output-cifar10-vit-debug --with_amp --with_pbar
```