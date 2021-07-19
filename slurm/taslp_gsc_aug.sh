#!/bin/sh

if [ ! "$1" = "run" ] && [ ! "$1" = "run_olympe" ] && [ ! "$1" = "run_osirim" ]; then
  echo "Usage: $0 (run|run_olympe|run_osirim)"
  exit 1
fi

dpath_scripts="."
run="$1"

epochs=300
activation="softmax"
gpus=1
cpus=4
accelerator="dp"
verbose=true
debug=false

dataset="gsc"

criterion="CE"
bsize=256
bsize_s=128
bsize_u=128

model="wideresnet28"
model_acro="wrn28"

rd="${RANDOM}"
tag_prefix="_taslp_${dataset}_${model_acro}_bsize_${bsize}_${rd}"

dataset_params="dataset=${dataset}"
common_params="${run} ${dataset_params} epochs=${epochs} expt.activation=${activation} gpus=${gpus} cpus=${cpus} trainer.accelerator=${accelerator} verbose=${verbose}"

sup_params="expt.criterion=${criterion} bsize=${bsize} model=${model}"
ssl_params="expt.criterion_s=${criterion} expt.criterion_u=${criterion} bsize_s=${bsize_s} bsize_u=${bsize_u}"

# # CutOutSpec --------------------------------------------------------------------------------------
#  Supervised 100%
# aug='weak'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='strong'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='weak2'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='strong2'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='weak3'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='strong3'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # TimeStretchPadCrop ------------------------------------------------------------------------------
# # Supervised 100%
# aug='test_cutoutspec_fill_-80'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_cutoutspec_fill_-100'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_cutoutspec_fill_0'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_cutoutspec_fill_range_-100_0'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_cutoutspec_random_-100_0'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_cutoutspec_fade_0.5'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_cutoutspec_fade_range_0_1'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_cutoutspec_addnoise_10'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_cutoutspec_addnoise_20'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_cutoutspec_subnoise_10'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_cutoutspec_subnoise_20'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # TimeStretchPadCrop ------------------------------------------------------------------------------
# # Supervised 100%
# aug='test_stretch_waveform_rates_0.5_1.5'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_stretch_spectro_rates_0.5_1.5'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_stretch_waveform_rates_1.0_1.5'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_stretch_spectro_rates_1.0_1.5'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_stretch_waveform_rates_0.5_1.0'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_stretch_spectro_rates_0.5_1.0'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # TimeStretchPadCrop 100% apply -------------------------------------------------------------------
# # Supervised 100%
# aug='test_p_1_stretch_waveform_rates_0.5_1.5'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_p_1_stretch_spectro_rates_0.5_1.5'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_p_1_stretch_waveform_rates_1.0_1.5'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_p_1_stretch_spectro_rates_1.0_1.5'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_p_1_stretch_waveform_rates_0.5_1.0'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_p_1_stretch_spectro_rates_0.5_1.0'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Occlusion ------------------------------------------------------------------------------
# # Supervised 100%
# aug='test_occlusion_waveform_scales_0_0.25'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_occlusion_spectro_scales_0_0.25'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_occlusion_waveform_scales_0.25_0.75'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_occlusion_spectro_scales_0.25_0.75'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

# # Supervised 100%
# aug='test_occlusion_spectro_scales_0.25_0.75_dim_-2'
# tag="${tag_prefix}_SUP100_${aug}"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug

for seed in 1235 1236 1237 1238 1239
do
	# Supervised 100%
	aug='test_occlusion_waveform_scales_0.25_0.75'
	tag="${tag_prefix}_SUP100_${aug}"
	$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug seed=${seed}

	# Supervised 100%
	aug='test_occlusion_spectro_scales_0.25_0.75'
	tag="${tag_prefix}_SUP100_${aug}"
	$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=$aug seed=${seed}
done
