
from hydra.utils import DictConfig
from torch.nn import Module

from .mobilenet import MobileNetV1, MobileNetV2
from .mobilenet_rot import MobileNetV1Rot, MobileNetV2Rot
from .wideresnet28 import WideResNet28
from .wideresnet28_rot import WideResNet28Rot


def get_model_from_cfg(cfg: DictConfig) -> Module:
	model_name = cfg.model.fullname

	if model_name == "WideResNet28":
		model = WideResNet28(
			num_classes=cfg.model.num_classes,
			num_input_channels=cfg.model.num_input_channels,
			width=cfg.model.width,
		)

	elif model_name == "MobileNetV1":
		model = MobileNetV1(num_classes=cfg.model.num_classes)

	elif model_name == "MobileNetV2":
		model = MobileNetV2(num_classes=cfg.model.num_classes)

	# Models with rotation layer and a "forward_rot()" method.
	elif model_name == "WideResNet28Rot":
		model = WideResNet28Rot(
			num_classes=cfg.model.num_classes,
			num_input_channels=cfg.model.num_input_channels,
			width=cfg.model.width,
			rot_size=cfg.model.rot_size,
		)

	elif model_name == "MobileNetV1Rot":
		model = MobileNetV1Rot(num_classes=cfg.model.num_classes, rot_size=cfg.model.rot_size)

	elif model_name == "MobileNetV2Rot":
		model = MobileNetV2Rot(num_classes=cfg.model.num_classes, rot_size=cfg.model.rot_size)

	else:
		raise RuntimeError(f"Unknown model name '{cfg.model.fullname}'.")

	return model
