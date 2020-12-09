
import torch

from argparse import Namespace
from sslh.models.checkpoint import load_state
from torch.nn import Module
from typing import List, Optional, Type


def get_model(
	name: str,
	args: Optional[Namespace] = None,
	models: Optional[list] = None,
	device: torch.device = torch.device("cuda")
) -> Module:
	"""
		Build model from a name, a argparse Namespace and a list of classes.

		:param name: The name of the model to build.
		:param args: The Namespace to use which contains argument for the model if it implements the "from_args(args: Namespace)" static method. (default: None)
			If None or if the model does not implements "from_args", use the default constructor without any argument.
		:param models: The list of classes to use for searching the model type. (default: None)
			If None, the list of classes will be initialized with all the models returns by the function "get_all_classes()".
		:param device: The device to use for the model. (default: torch.device("cuda"))
	"""

	if models is None:
		models = get_all_classes()

	model_class = get_model_class(name, models)

	if args is not None and hasattr(model_class, "from_args"):
		model = model_class.from_args(args)
	else:
		model = model_class()

	model = model.to(device)
	return model


def get_model_class(name: str, models: List[Type[Module]]) -> Type[Module]:
	for model_class in models:
		if model_class.__name__.lower() == name.lower():
			return model_class

	models_names = [model.__name__ for model in models]
	raise RuntimeError("Cannot find model \"{:s}\". Available models are {:s}".format(name, str(models_names)))


def get_all_classes() -> list:
	"""
		Returns the list of classes of the models available in this project.
	"""
	from sslh.models.cnn03 import CNN03, CNN03Rot
	from sslh.models.cnn03mish import CNN03Mish, CNN03MishRot
	from sslh.models.desed_baseline import WeakBaseline, WeakStrongBaseline
	from sslh.models.ubs8k_baseline import UBS8KBaseline, UBS8KBaselineRot
	from sslh.models.vgg import VGG, VGGRot
	from sslh.models.wrn28_2 import WideResNet28, WideResNet28Rot, WideResNet28Spec, WideResNet28RotSpec
	return [
		CNN03, CNN03Rot, CNN03Mish, CNN03MishRot, WeakBaseline, WeakStrongBaseline, UBS8KBaseline, UBS8KBaselineRot,
		VGG, VGGRot, WideResNet28, WideResNet28Rot, WideResNet28Spec, WideResNet28RotSpec
	]


def load_model_from_file(name: str, args: Optional[Namespace], filepath: Optional[str]) -> Module:
	"""
		Construct model and load it from a torch file saved by Checkpoint or CheckpointMultiple classes.

		:param name: The name of the model to build.
		:param args: The Namespace to use which contains argument for the model if it implements the "from_args(args: Namespace)" static method.
			If None or if the model does not implements "from_args", use the default constructor without any argument.
		:param filepath: The path to the .torch file saved by Checkpoint or CheckpointMultiple classes where the values of the model are stored.
	"""
	model = get_model(name, args)
	if filepath is not None:
		_best_value, _best_epoch = load_state(filepath, model, None)
	return model
