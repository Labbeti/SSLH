
import torch

from argparse import Namespace

from sslh.datasets.base import DatasetBuilder
from sslh.models.checkpoint import load_state
from sslh.models.cnn03 import CNN03, CNN03Rot
from sslh.models.cnn03mish import CNN03Mish, CNN03MishRot
from sslh.models.desed_baseline import WeakBaseline, WeakStrongBaseline
from sslh.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from sslh.models.ubs8k_baseline import UBS8KBaseline, UBS8KBaselineRot
from sslh.models.vgg import VGG, VGGRot
from sslh.models.wideresnet28 import WideResNet28, WideResNet28Rot, WideResNet28Repeat, WideResNet28RotRepeat

from torch.nn import Module
from typing import Callable, List, Optional, Type


def get_model(
	name: str,
	args: Namespace,
	builder: DatasetBuilder,
	device: torch.device = torch.device("cuda"),
	models: Optional[List[Type[Module]]] = None,
) -> Module:
	"""
		Build model from a name, a argparse Namespace and a list of classes.

		:param name: The name of the model to build.
		:param args: The Namespace to use which contains argument for the model.
		:param builder: The DatasetBuilder of the dataset.
		:param device: The device to use for the model. (default: torch.device("cuda"))
		:param models: The list of classes to use for searching the model type. (default: None)
			If None, the list of classes will be initialized with all the models returns by the function "get_all_classes()".
	"""
	name = name.lower()

	if name == CNN03.__name__.lower():
		model = CNN03(
			output_size=builder.get_nb_classes(),
			dropout=args.dropout,
		)
	elif name == CNN03Rot.__name__.lower():
		model = CNN03Rot(
			output_size=builder.get_nb_classes(),
			dropout=args.dropout,
			rot_output_size=args.nb_classes_self_supervised,
		)
	elif name == CNN03Mish.__name__.lower():
		model = CNN03Mish(
			output_size=builder.get_nb_classes(),
			dropout=args.dropout,
		)
	elif name == CNN03MishRot.__name__.lower():
		model = CNN03MishRot(
			output_size=builder.get_nb_classes(),
			dropout=args.dropout,
			rot_output_size=args.nb_classes_self_supervised,
		)
	elif name == UBS8KBaseline.__name__.lower():
		model = UBS8KBaseline(
			output_size=builder.get_nb_classes(),
			dropout=args.dropout,
		)
	elif name == UBS8KBaselineRot.__name__.lower():
		model = UBS8KBaselineRot(
			output_size=builder.get_nb_classes(),
			dropout=args.dropout,
			rot_output_size=args.nb_classes_self_supervised,
		)
	elif name == VGGRot.__name__.lower():
		model = VGGRot(
			vgg_name="VGG11",
			rot_output_size=args.nb_classes_self_supervised,
		)
	elif name == WideResNet28.__name__.lower():
		model = WideResNet28(
			num_classes=builder.get_nb_classes(),
			width=args.wrn_width,
			num_input_channels=args.wrn_num_input_channels,
		)
	elif name == WideResNet28Rot.__name__.lower():
		model = WideResNet28Rot(
			num_classes=builder.get_nb_classes(),
			width=args.wrn_width,
			rot_size=args.nb_classes_self_supervised,
			num_input_channels=args.wrn_num_input_channels,
		)
	else:
		model = get_model_default(name, device, models)

	model = model.to(device)
	return model


def get_model_default(
	name: str,
	device: torch.device = torch.device("cuda"),
	models: Optional[List[Type[Module]]] = None,
	**kwargs,
) -> Module:
	if models is None:
		models = get_all_classes()

	model_class = get_model_class(name, models)
	if model_class is None:
		models_names = str([model.__name__ for model in models])
		raise RuntimeError(f"Cannot find model '{name}'. Available models are {models_names}")

	model = model_class(**kwargs)
	model = model.to(device)
	return model


def get_model_class(name: str, models: List[Type[Module]]) -> Optional[Callable]:
	for model_class in models:
		if model_class.__name__.lower() == name.lower():
			return model_class
	return None


def get_all_classes() -> List[Type[Module]]:
	"""
		Returns the list of models classes available in this project.
	"""
	return [
		CNN03,
		CNN03Rot,
		CNN03Mish,
		CNN03MishRot,
		UBS8KBaseline,
		UBS8KBaselineRot,
		VGG,
		VGGRot,
		WeakBaseline,
		WeakStrongBaseline,
		WideResNet28,
		WideResNet28Rot,
		WideResNet28Repeat,
		WideResNet28RotRepeat,
		ResNet18,
		ResNet34,
		ResNet50,
		ResNet101,
		ResNet152,
	]


def load_model_from_file(name: str, args: Namespace, builder: DatasetBuilder, filepath: Optional[str]) -> Module:
	"""
		Construct model and load it from a torch file saved by Checkpoint or CheckpointMultiple classes.

		:param name: The name of the model to build.
		:param args: The Namespace to use which contains argument for the model.
		:param builder: The DatasetBuilder of the dataset.
		:param filepath: The path to the .torch file saved by Checkpoint or CheckpointMultiple classes where the values
			of the model are stored.
	"""
	model = get_model(name, args, builder)
	if filepath is not None:
		_best_value, _best_epoch = load_state(filepath, model, None)
	return model
