
# ./su_exp.sh --dataset AUDIOSET --epochs 10 --bsize_s 64 --criterion bce --train_version balanced --su_ratio 0.1 --activation sigmoid --nb_gpu 1
# ./su_exp.sh --dataset AUDIOSET --epochs 10 --bsize_s 64 --criterion bce --train_version balanced --su_ratio 1.0 --activation sigmoid --nb_gpu 1

#./mm_exp.sh --dataset AUDIOSET --epochs 10 --bsize_s 32 --bsize_u 32 --criterion_s bce --criterion_u bce --train_version balanced --su_ratio 0.1 --activation sigmoid --nb_gpu 1 --temperature 1.0 --nb_gpu 2
./fm_exp.sh \
	--dataset AUDIOSET --train_version balanced --activation sigmoid --epochs 10 --bsize_s 32 --bsize_u 32 --criterion_s bce --criterion_u bce --su_ratio 0.1 --threshold -0.1 --use_threshold_guess 1 --threshold_guess 0.5 --nb_gpu 2

# ./mm_exp.sh --dataset AUDIOSET --epochs 10 --bsize_s 32 --bsize_u 32 --criterion_s bce --criterion_u bce --train_version balanced --su_ratio 0.1 --activation sigmoid --nb_gpu 1 --use_no_mixup 1 --nb_gpu 2
# ./fm_exp.sh --dataset AUDIOSET --epochs 10 --bsize_s 32 --bsize_u 32 --criterion_s bce --criterion_u bce --train_version balanced --su_ratio 0.1 --activation sigmoid --nb_gpu 1 --use_mixup 1 --threshold 0.0 --use_threshold_guess 1 --threshold_guess 0.5 --nb_gpu 2
