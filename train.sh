
 # RefCOCOg
 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
 --master_port 12345 \
 --use_env train.py \
 --batch_size 20 \
 --lr 0.000025 \
 --lr_bert 0.000005 \
 --lr_visual 0.00001 \
 --aug_scale --aug_translate --aug_crop \
 --backbone ViTDet --is_eliminate \
 --imsize 448 \
 --bert_enc_num 12 \
 --dataset gref_umd \
 --max_query_len 40 \
 --lr_scheduler poly \
 --is_segment \
 --vl_enc_layers 3 \
 --dim_feedforward 1024 \
 --loss_alpha 0.1 \
 --epochs 150 \
 --output_dir outputs/refcocog_ViTDet >refcocog_ViTDet.txt 2>&1 &

# Pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
 --master_port 12345 \
 --use_env train.py \
 --batch_size 20 \
 --lr 0.000025 \
 --lr_bert 0.00005 \
 --lr_visual 0.00001 \
 --aug_scale --aug_translate --aug_crop \
 --backbone ViTDet \
 --imsize 448 \
 --bert_enc_num 12 \
 --dataset mixed_pretrain \
 --max_query_len 40 \
 --vl_enc_layers 3 \
 --dim_feedforward 1024 \
 --lr_scheduler poly \
 --loss_alpha 0.5 \
 --epochs 20 \
 --output_dir outputs/mixed_pretrain_decoder >mixed_pretrain_decoder.txt 2>&1 &

 # Finetune
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
 --master_port 12345 \
 --use_env train.py \
 --batch_size 20 \
 --is_segment \
 --lr 0.000025 \
 --lr_bert 0.000005 \
 --lr_visual 0.00001 \
 --aug_scale --aug_translate --aug_crop \
 --backbone ViTDet --is_eliminate \
 --imsize 448 \
 --bert_enc_num 12 \
 --dataset mixed_coco \
 --max_query_len 40 \
 --vl_enc_layers 3 \
 --dim_feedforward 1024 \
 --lr_scheduler poly \
 --loss_alpha 0.05 \
 --eliminated_threshold 0.0015 \
 --epochs 150 \
 --pretrain outputs/mixed_pretrain_decoder/checkpoint.pth \
 --output_dir outputs/mixed_coco_decoder >mixed_coco_decoder.txt 2>&1 &

