
from typing import Callable, List, Tuple
from mlu.transforms import Occlusion, CutOutSpec, TimeStretchPadCrop, Fade, AdditiveNoise, SubtractiveNoise


def get_pool(augment_name: str) -> List[Tuple[str, Callable]]:
	if augment_name in ['weak']:
		pool = get_weak_augm_pool()
	elif augment_name in ['strong']:
		pool = get_strong_augm_pool()
	elif augment_name in ['weak2']:
		pool = get_weak2_augm_pool()
	elif augment_name in ['strong2']:
		pool = get_strong2_augm_pool()
	elif augment_name in ['weak3']:
		pool = get_weak3_augm_pool()
	elif augment_name in ['strong3']:
		pool = get_strong3_augm_pool()
	elif augment_name in ['identity']:
		pool = []
	elif augment_name.startswith("test_"):
		pool = get_pool_test(augment_name)
	else:
		raise RuntimeError(
			f'Unknown augment name "{augment_name}". '
			f'Must be one of {("weak", "strong", "weak2", "strong2", "identity")}.'
		)
	return pool


def get_weak_augm_pool() -> List[Tuple[str, Callable]]:
	common_params = dict(p=0.5)
	return [
		('waveform', Occlusion(scales=(0.0, 0.25), **common_params)),
		('waveform', TimeStretchPadCrop(rates=(0.5, 1.5), align='random', **common_params)),
		('spectrogram', CutOutSpec(freq_scales=(0.1, 0.5), time_scales=(0.1, 0.5), fill_value=-80.0, **common_params)),
	]


def get_strong_augm_pool() -> List[Tuple[str, Callable]]:
	common_params = dict(p=1.0)
	return [
		('waveform', Occlusion(scales=(0.0, 0.75), **common_params)),
		('waveform', TimeStretchPadCrop(rates=(0.25, 1.75), align='random', **common_params)),
		('spectrogram', CutOutSpec(freq_scales=(0.5, 1.0), time_scales=(0.5, 1.0), fill_value=-80.0, **common_params)),
	]


def get_weak2_augm_pool() -> List[Tuple[str, Callable]]:
	"""
		Differences with 'weak':
			- Occlusion & TimeStretchPadCrop on spec
			- fill_value from -80 to -100
			- CutOutSpec freq_scales higher and time_scales lower
	"""
	common_params = dict(p=0.5, fill_value=-100.0)
	return [
		('spectrogram', Occlusion(scales=(0.0, 0.25), **common_params)),
		('spectrogram', TimeStretchPadCrop(rates=(0.5, 1.5), align='random', **common_params)),
		('spectrogram', CutOutSpec(freq_scales=(0.5, 1.0), time_scales=(0.0, 0.5), **common_params)),
	]


def get_strong2_augm_pool() -> List[Tuple[str, Callable]]:
	"""
		Differences with 'strong':
			- Occlusion & TimeStretchPadCrop on spec
			- fill_value from -80 to -100
			- CutOutSpec freq_scales higher and time_scales lower
	"""
	common_params = dict(p=1.0, fill_value=-100.0)
	return [
		('spectrogram', Occlusion(scales=(0.0, 0.75), **common_params)),
		('spectrogram', TimeStretchPadCrop(rates=(0.25, 1.75), align='random', **common_params)),
		('spectrogram', CutOutSpec(freq_scales=(0.75, 1.0), time_scales=(0.5, 0.75), **common_params)),
	]


def get_weak3_augm_pool() -> List[Tuple[str, Callable]]:
	common_params = dict(p=0.5, fill_value=-100.0)
	return [
		('spectrogram', Occlusion(scales=(0.0, 0.25), **common_params)),
		('spectrogram', TimeStretchPadCrop(rates=(0.5, 1.5), align='random', **common_params)),
		('spectrogram', CutOutSpec(freq_scales=(0.1, 0.5), time_scales=(0.1, 0.5), **common_params)),
	]


def get_strong3_augm_pool() -> List[Tuple[str, Callable]]:
	common_params = dict(p=1.0, fill_value=-100.0)
	return [
		('spectrogram', Occlusion(scales=(0.0, 0.75), **common_params)),
		('spectrogram', TimeStretchPadCrop(rates=(0.25, 1.75), align='random', **common_params)),
		('spectrogram', CutOutSpec(freq_scales=(0.5, 1.0), time_scales=(0.5, 1.0), **common_params)),
	]


def get_pool_test(augment_name: str) -> List[Tuple[str, Callable]]:
	if augment_name.startswith('test_cutoutspec'):
		return get_pool_test_cutoutspec(augment_name)
	elif augment_name.startswith('test_stretch'):
		return get_pool_test_stretch(augment_name)
	elif augment_name.startswith('test_p_1_stretch'):
		return get_pool_test_p_1_stretch(augment_name)
	elif augment_name.startswith('test_occlusion'):
		return get_pool_test_occlusion(augment_name)
	else:
		raise ValueError(f'Unknown augment test "{augment_name}".')


def get_pool_test_cutoutspec(augment_name: str) -> List[Tuple[str, Callable]]:
	common_params = dict(
		freq_scales=(0.1, 0.5), time_scales=(0.1, 0.5), p=0.5,
	)

	if augment_name in ['test_cutoutspec_fill_-80']:
		aug = CutOutSpec(**common_params, fill_value=-80.0)

	elif augment_name in ['test_cutoutspec_fill_-100']:
		aug = CutOutSpec(**common_params, fill_value=-100.0)

	elif augment_name in ['test_cutoutspec_fill_0']:
		aug = CutOutSpec(**common_params, fill_value=0.0)

	elif augment_name in ['test_cutoutspec_fill_range_-100_0']:
		aug = CutOutSpec(**common_params, fill_value=(-100.0, 0.0))

	elif augment_name in ['test_cutoutspec_random_-100_0']:
		aug = CutOutSpec(**common_params, fill_value=(-100.0, 0.0), fill_mode='random')

	elif augment_name in ['test_cutoutspec_fade_0.5']:
		aug = CutOutSpec(**common_params, fill_mode=Fade(factor=0.5))

	elif augment_name in ['test_cutoutspec_fade_range_0_1']:
		aug = CutOutSpec(**common_params, fill_mode=Fade(factor=(0.0, 1.0)))

	elif augment_name in ['test_cutoutspec_addnoise_10']:
		aug = CutOutSpec(**common_params, fill_mode=AdditiveNoise(snr_db=10.0))

	elif augment_name in ['test_cutoutspec_addnoise_20']:
		aug = CutOutSpec(**common_params, fill_mode=AdditiveNoise(snr_db=20.0))

	elif augment_name in ['test_cutoutspec_subnoise_10']:
		aug = CutOutSpec(**common_params, fill_mode=SubtractiveNoise(snr_db=10.0))

	elif augment_name in ['test_cutoutspec_subnoise_20']:
		aug = CutOutSpec(**common_params, fill_mode=SubtractiveNoise(snr_db=20.0))

	else:
		raise ValueError(f'Unknown augment test "{augment_name}".')

	pool = [('spectrogram', aug)]
	return pool


def get_pool_test_stretch(augment_name: str) -> List[Tuple[str, Callable]]:
	common_params = dict(
		p=0.5,
		align='random',
	)

	if augment_name in ['test_stretch_waveform_rates_0.5_1.5']:
		aug = TimeStretchPadCrop(rates=(0.5, 1.5), fill_value=0.0, **common_params)
		audio_type = 'waveform'

	elif augment_name in ['test_stretch_spectro_rates_0.5_1.5']:
		aug = TimeStretchPadCrop(rates=(0.5, 1.5), fill_value=-100.0, **common_params)
		audio_type = 'spectrogram'

	elif augment_name in ['test_stretch_waveform_rates_1.0_1.5']:
		aug = TimeStretchPadCrop(rates=(1.0, 1.5), fill_value=0.0, **common_params)
		audio_type = 'waveform'

	elif augment_name in ['test_stretch_spectro_rates_1.0_1.5']:
		aug = TimeStretchPadCrop(rates=(1.0, 1.5), fill_value=-100.0, **common_params)
		audio_type = 'spectrogram'

	elif augment_name in ['test_stretch_waveform_rates_0.5_1.0']:
		aug = TimeStretchPadCrop(rates=(0.5, 1.0), fill_value=0.0, **common_params)
		audio_type = 'waveform'

	elif augment_name in ['test_stretch_spectro_rates_0.5_1.0']:
		aug = TimeStretchPadCrop(rates=(0.5, 1.0), fill_value=-100.0, **common_params)
		audio_type = 'spectrogram'

	else:
		raise ValueError(f'Unknown augment test "{augment_name}".')

	pool = [(audio_type, aug)]
	return pool


def get_pool_test_p_1_stretch(augment_name: str) -> List[Tuple[str, Callable]]:
	common_params = dict(
		p=1.0,
		align='random',
	)

	if augment_name in ['test_p_1_stretch_waveform_rates_0.5_1.5']:
		aug = TimeStretchPadCrop(rates=(0.5, 1.5), fill_value=0.0, **common_params)
		audio_type = 'waveform'

	elif augment_name in ['test_p_1_stretch_spectro_rates_0.5_1.5']:
		aug = TimeStretchPadCrop(rates=(0.5, 1.5), fill_value=-100.0, **common_params)
		audio_type = 'spectrogram'

	elif augment_name in ['test_p_1_stretch_waveform_rates_1.0_1.5']:
		aug = TimeStretchPadCrop(rates=(1.0, 1.5), fill_value=0.0, **common_params)
		audio_type = 'waveform'

	elif augment_name in ['test_p_1_stretch_spectro_rates_1.0_1.5']:
		aug = TimeStretchPadCrop(rates=(1.0, 1.5), fill_value=-100.0, **common_params)
		audio_type = 'spectrogram'

	elif augment_name in ['test_p_1_stretch_waveform_rates_0.5_1.0']:
		aug = TimeStretchPadCrop(rates=(0.5, 1.0), fill_value=0.0, **common_params)
		audio_type = 'waveform'

	elif augment_name in ['test_p_1_stretch_spectro_rates_0.5_1.0']:
		aug = TimeStretchPadCrop(rates=(0.5, 1.0), fill_value=-100.0, **common_params)
		audio_type = 'spectrogram'

	else:
		raise ValueError(f'Unknown augment test "{augment_name}".')

	pool = [(audio_type, aug)]
	return pool


def get_pool_test_occlusion(augment_name: str) -> List[Tuple[str, Callable]]:
	common_params = dict(
		p=0.5,
	)

	if augment_name in ['test_occlusion_waveform_scales_0_0.25']:
		aug = Occlusion(scales=(0.0, 0.25), fill_value=0.0, **common_params)
		audio_type = 'waveform'

	elif augment_name in ['test_occlusion_spectro_scales_0_0.25']:
		aug = Occlusion(scales=(0.0, 0.25), fill_value=-100.0, **common_params)
		audio_type = 'spectrogram'

	elif augment_name in ['test_occlusion_waveform_scales_0.25_0.75']:
		aug = Occlusion(scales=(0.25, 0.75), fill_value=0.0, **common_params)
		audio_type = 'waveform'

	elif augment_name in ['test_occlusion_spectro_scales_0.25_0.75']:
		aug = Occlusion(scales=(0.25, 0.75), fill_value=-100.0, **common_params)
		audio_type = 'spectrogram'

	elif augment_name in ['test_occlusion_spectro_scales_0.25_0.75_dim_-2']:
		aug = Occlusion(scales=(0.25, 0.75), fill_value=-100.0, dim=-2, **common_params)
		audio_type = 'spectrogram'

	else:
		raise ValueError(f'Unknown augment test "{augment_name}".')

	pool = [(audio_type, aug)]
	return pool
