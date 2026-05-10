python script/predict.py \
    --config log/Backbone/RL_hard_share/3M/small_repective_filed_strides1113.ini \
    --checkpoint /home/xli263/xli/TE_prediction/MTtrans/checkpoint/RL_hard_share_MTL/3M/small_repective_filed_strides1113-model_best.pth \
    --input /home/xli263/xli/utr_design/DRAKES/drakes_dna/new_generated_sequence/generated_sequences_short_motif_filter.csv \
    --task MPA_H \
    --out predictions_new__short_motif.csv