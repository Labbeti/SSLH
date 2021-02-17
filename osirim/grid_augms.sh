
su_ratio=1.0
ratio=0.5
tag="augms_test1"

#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "None" --tag "${tag}_None"

# Occlusion --------------------------------------------------
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "Occlusion" --tag "${tag}_Occlusion_0.01" --occlusion_max_size 0.01
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "Occlusion" --tag "${tag}_Occlusion_0.1" --occlusion_max_size 0.1
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "Occlusion" --tag "${tag}_Occlusion_0.25" --occlusion_max_size 0.25
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "Occlusion" --tag "${tag}_Occlusion_0.5" --occlusion_max_size 0.5
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "Occlusion" --tag "${tag}_Occlusion_0.75" --occlusion_max_size 0.75
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "Occlusion" --tag "${tag}_Occlusion" # default: 1.0
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "Occlusion" --tag "${tag}_Occlusion_2.0" --occlusion_max_size 2.0
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "Occlusion" --tag "${tag}_Occlusion_4.0" --occlusion_max_size 4.0
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "Occlusion" --tag "${tag}_Occlusion_8.0" --occlusion_max_size 8.0

# CutOutSpec --------------------------------------------------
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "CutOutSpec" --tag "${tag}_CutOutSpec_0.0_0.25" --cutout_width_scale 0.0 0.25 --cutout_height_scale 0.0 0.25
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "CutOutSpec" --tag "${tag}_CutOutSpec_0.0_0.5" --cutout_width_scale 0.0 0.5 --cutout_height_scale 0.0 0.5
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "CutOutSpec" --tag "${tag}_CutOutSpec_0.0_1.0" --cutout_width_scale 0.0 1.0 --cutout_height_scale 0.0 1.0

#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "CutOutSpec" --tag "${tag}_CutOutSpec" # default: (0.1, 0.5)

#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "CutOutSpec" --tag "${tag}_CutOutSpec_0.25_0.5" --cutout_width_scale 0.25 0.5 --cutout_height_scale 0.25 0.5
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "CutOutSpec" --tag "${tag}_CutOutSpec_0.25_0.75" --cutout_width_scale 0.25 0.75 --cutout_height_scale 0.25 0.75 

#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "CutOutSpec" --tag "${tag}_CutOutSpec_0.5_0.75" --cutout_width_scale 0.5 0.75 --cutout_height_scale 0.5 0.75
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "CutOutSpec" --tag "${tag}_CutOutSpec_0.5_1.0" --cutout_width_scale 0.5 1.0 --cutout_height_scale 0.5 1.0

# RandomFreqDropout --------------------------------------------------
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "RandomFreqDropout" --tag "${tag}_RandomFreqDropout" # default: 0.01
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "RandomFreqDropout" --tag "${tag}_RandomFreqDropout_0.1" --random_freq_dropout 0.1

# RandomTimeDropout --------------------------------------------------
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "RandomTimeDropout" --tag "${tag}_RandomTimeDropout" # default: 0.01
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "RandomTimeDropout" --tag "${tag}_RandomTimeDropout_0.1" --random_time_dropout 0.1

# NoiseSpec --------------------------------------------------
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "NoiseSpec" --tag "${tag}_NoiseSpec_1.0" --noise_snr 1.0
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "NoiseSpec" --tag "${tag}_NoiseSpec_5.0" --noise_snr 5.0
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "NoiseSpec" --tag "${tag}_NoiseSpec"  # default: 10.0
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "NoiseSpec" --tag "${tag}_NoiseSpec_20.0" --noise_snr 20.0
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "NoiseSpec" --tag "${tag}_NoiseSpec_30.0" --noise_snr 30.0
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "NoiseSpec" --tag "${tag}_NoiseSpec_40.0" --noise_snr 40.0#
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "NoiseSpec" --tag "${tag}_NoiseSpec_100.0" --noise_snr 100.0

# ResizePadCrop --------------------------------------------------
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "ResizePadCrop" --tag "${tag}_ResizePadCrop" # default: (0.9, 1.1), left
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "ResizePadCrop" --tag "${tag}_ResizePadCrop_center" --resize_align "center"
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "ResizePadCrop" --tag "${tag}_ResizePadCrop_right" --resize_align "right"
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "ResizePadCrop" --tag "${tag}_ResizePadCrop_random" --resize_align "random"

#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "ResizePadCrop" --tag "${tag}_ResizePadCrop_0.5_1.5_random" --resize_align "random" --resize_rate 0.5 1.5
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "ResizePadCrop" --tag "${tag}_ResizePadCrop_0.25_1.75_random" --resize_align "random" --resize_rate 0.25 1.75

# HorizontalFlip --------------------------------------------------
#./su_augm.sh --dataset "GSC" --su_ratio "$su_ratio" --ratio "$ratio" --augm_train "HorizontalFlip" --tag "${tag}_HorizontalFlip"