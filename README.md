# Area Encoding experiments for Unofficial PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

Original [repo](https://github.com/pengzhiliang/MAE-pytorch) of unofficial MAE is built upon [BEiT](https://github.com/microsoft/unilm/tree/master/beit), thanks very much for all! (I recommend you to check out both repositories for their own issues.)

In this repo, I add some area encodings which replace the classical sinusoidal positional encoding.

## Updates
- (Mar. 17, 2022) branch created, add naive area and aaud encoding (modeling_pretrain.py)

## TODO
- [ ] parallelize naive area encoding
- [ ] 

(below setup and run section are from original [repo](https://github.com/pengzhiliang/MAE-pytorch). 
if you want to run **naive area encoding** or **aaud encoding** model, change the model parameter as below
- pretrain_mae_small_patch16_224_with_naive_ae
- pretrain_mae_base_patch16_224_with_naive_ae
- pretrain_mae_small_patch16_224_with_aaud
- pretrain_mae_base_patch16_224_with_aaud

## Setup

```
pip install -r requirements.txt
```

## Run
1. Pretrain
```bash
# Set the path to save checkpoints
OUTPUT_DIR='output/pretrain_mae_base_patch16_224'
# path to imagenet-1k train set
DATA_PATH='/path/to/ImageNet_ILSVRC2012/train'


# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_ratio 0.75 \
        --model pretrain_mae_base_patch16_224 \
        --batch_size 128 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --output_dir ${OUTPUT_DIR}
```

2. Finetune
```bash
# Set the path to save checkpoints
OUTPUT_DIR='output/'
# path to imagenet-1k set
DATA_PATH='/path/to/ImageNet_ILSVRC2012'
# path to pretrain model
MODEL_PATH='/path/to/pretrain/checkpoint.pth'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --dist_eval
```
3. Visualization of reconstruction
```bash
# Set the path to save images
OUTPUT_DIR='output/'
# path to image for visualization
IMAGE_PATH='files/ILSVRC2012_val_00031649.JPEG'
# path to pretrain model
MODEL_PATH='/path/to/pretrain/checkpoint.pth'

# Now, it only supports pretrained models with normalized pixel targets
python run_mae_vis.py ${IMAGE_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
```

