# default
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node={N_GPU} run_mae_pretraining.py \
        --data_path {DATA_DIR_FOR_ILSVRC2012_TRAIN} \
        --mask_ratio 0.75 \
        --model pretrain_mae_small_patch16_224 \
        --batch_size {BATCH_SIZE} \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --output_dir output/pretrain_mae_small_patch16_224

# learnable pos
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node={N_GPU} run_mae_pretraining.py \
        --data_path {DATA_DIR_FOR_ILSVRC2012_TRAIN} \
        --mask_ratio 0.75 \
        --model pretrain_mae_small_patch16_224_with_learnable_pos \
        --batch_size {BATCH_SIZE} \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --output_dir output/pretrain_mae_small_patch16_224_with_learnable_pos

# without pos
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node={N_GPU} run_mae_pretraining.py \
        --data_path {DATA_DIR_FOR_ILSVRC2012_TRAIN} \
        --mask_ratio 0.75 \
        --model pretrain_mae_small_patch16_224_without_pos \
        --batch_size {BATCH_SIZE} \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --output_dir output/pretrain_mae_small_patch16_224_without_learnable_pos

# aaud
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node={N_GPU} run_mae_pretraining.py \
        --data_path {DATA_DIR_FOR_ILSVRC2012_TRAIN} \
        --mask_ratio 0.75 \
        --model pretrain_mae_small_patch16_224_with_aaud \
        --batch_size {BATCH_SIZE} \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --output_dir output/pretrain_mae_small_patch16_224_with_aaud

# naive area
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node={N_GPU} run_mae_pretraining.py \
        --data_path {DATA_DIR_FOR_ILSVRC2012_TRAIN} \
        --mask_ratio 0.75 \
        --model pretrain_mae_small_patch16_224_with_naive_ae \
        --batch_size {BATCH_SIZE} \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --output_dir output/pretrain_mae_small_patch16_224_with_naive_ae





