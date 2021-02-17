
import os.path as osp
import torch

from argparse import ArgumentParser
from matplotlib import pyplot as plt

from sslh.datasets.get_builder import get_dataset_builder
from sslh.utils.adversaries.fgsm import FGSM
from sslh.utils.adversaries.iterative_fgsm import IterativeFGSM
from sslh.models.get_model import load_model_from_file
from mlu.nn import CrossEntropyWithVectors


class TestAdv:
	def test_gradient_attack(self):
		builder = get_dataset_builder("ESC10")

		parser = ArgumentParser()
		parser.add_argument("--dataset_path", type=str, default=osp.join("..", "datasets"))
		parser.add_argument("--nb_classes_self_supervised", type=int, default=4)
		parser.add_argument("--label_smoothing_value", type=float, default=None)
		parser.add_argument("--augm_none", type=str, default=None)

		parser.add_argument("--epsilon", type=float, default=1e-0)
		parser.add_argument("--nb_iterations", type=int, default=10)
		parser.add_argument("--model", type=str, default=osp.join(
			"..",
			"results",
			"tensorboard",
			# "ESC10_2020-11-12_14:36:32_WideResNet28RotSpec_MixMatch_10.00%_[5]_test_for_adv",
			# "WideResNet28RotSpec_26.torch",
			"ESC10_2020-11-11_15:40:04_WideResNet28RotSpec_Supervised_10.00%_[5]_",
			"WideResNet28RotSpec_1.torch",
		))

		args = parser.parse_args()
		args.nb_classes = builder.get_nb_classes()

		model = load_model_from_file("WideResNet28RotSpec", args, args.model)

		activation = torch.softmax
		criterion = CrossEntropyWithVectors()

		dataset = builder.get_dataset_train(args)
		original_spec, label = dataset[20]

		device = torch.device("cuda")
		original_spec = torch.as_tensor(original_spec).unsqueeze(dim=0).to(device).float()
		label = torch.as_tensor(label).unsqueeze(dim=0).to(device).float()

		fgsm = FGSM(model, activation, criterion, args.epsilon)
		it_fgsm = IterativeFGSM(model, activation, criterion, args.epsilon, args.nb_iterations)

		specs = [original_spec]

		for method in (fgsm, it_fgsm):
			spec_perturbed = method(original_spec, label)
			specs.append(spec_perturbed + original_spec)

		print(f"Label : idx={label.argmax().item()}")
		for i, spec in enumerate(specs):
			pred = activation(model(spec), dim=1)[0]
			label_idx = pred.argmax().item()
			max_ = pred.max().item()
			dist = (spec - original_spec[0]).abs().mean()
			print(f"[{i}]   : idx={label_idx}, max={max_}, dist={dist}")

		print("Show...")
		for spec in specs:
			spec = spec.detach().cpu().squeeze().numpy()
			plt.figure()
			plt.imshow(spec, origin="lower")
		plt.show(block=False)
		input("Press ENTER to quit >")


if __name__ == "__main__":
	TestAdv().test_gradient_attack()
