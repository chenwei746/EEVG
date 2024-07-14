
# RefCOCOg val
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
--master_port 12345 --use_env eval.py \
--batch_size 20 --num_workers 10 \
--bert_enc_num 12 \
--backbone ViTDet --imsize 448 \
--dataset gref_umd --max_query_len 40 \
--eval_set val  --vl_enc_layers 3 \
--dim_feedforward 1024 \
--eval_model ./outputs/refcocog_ViTDet/best_mask_checkpoint.pth \
--output_dir ./outputs/refcocog_ViTDet \
--is_segment

# RefCOCOg test
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
--master_port 12345 --use_env eval.py \
--batch_size 20 --num_workers 10 \
--bert_enc_num 12 \
--backbone ViTDet --imsize 448 \
--dataset gref_umd --max_query_len 40 \
--eval_set test  --vl_enc_layers 3 \
--dim_feedforward 1024 \
--eval_model ./outputs/refcocog_ViTDet/best_mask_checkpoint.pth \
--output_dir ./outputs/refcocog_ViTDet \
--is_segment


CUDA_VISIBLE_DEVICES=2,4 python -m torch.distributed.launch --nproc_per_node=2 \
--master_port 12345 --use_env eval.py \
--batch_size 20 --num_workers 10 \
--bert_enc_num 12 \
--backbone ViTDet --imsize 448 \
--dataset gref_umd --max_query_len 40 \
--eval_set test  --vl_enc_layers 3 \
--dim_feedforward 1024 \
--eval_model ./outputs/refcocog_ViTDet/best_mask_checkpoint.pth \
--output_dir ./outputs/refcocog_ViTDet \
--is_segment --is_eliminate
