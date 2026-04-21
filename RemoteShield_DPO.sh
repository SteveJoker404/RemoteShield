BATCH_PER_DEVICE=1 
GRAD_ACCUM_STEPS=16
MODEL_NAME="/path/to/your/model"
DATA_PATH="/path/to/your/file"
OUTPUT_DIR="RemoteShield_dpo_lora"
MAX_PIXELS=1638400

TRAIN_EPOCH=8

NPROC_PER_NODE=6 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
swift rlhf \
  --rlhf_type dpo \
  --model $MODEL_NAME \
  --dataset $DATA_PATH \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --max_pixels $MAX_PIXELS \
  --tuner_type lora \
  --target_modules all-linear \
  --lora_rank 32 \
  --lora_alpha 64 \
  --remove_unused_columns false \
  --ddp_find_unused_parameters false \
  --num_train_epochs $TRAIN_EPOCH \
  --per_device_train_batch_size $BATCH_PER_DEVICE \
  --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
  --learning_rate 3e-5 \
  --warmup_ratio 0.05 \
  --beta 0.1 \
  --loss_type sigmoid \
  --rpo_alpha 0.1 \
  --dataloader_num_workers 8 \
  --dataset_num_proc 8 \
  --max_length 4096 \
  --save_steps 100 \
  --save_total_limit 12 \
  --logging_steps 10 \
  --report_to tensorboard \
  --deepspeed zero2 \
  --seed 42 \
  --output_dir $OUTPUT_DIR \
  --gradient_checkpointing False
