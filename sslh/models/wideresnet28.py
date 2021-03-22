
from sslh.models.wideresnet import WideResNet


class WideResNet28(WideResNet):
	"""
		WideResNet-28 class. Expects an input of shape (bsize, 1, nb_mels, time stamps).
	"""
	def __init__(self, num_classes: int, width: int = 2, num_input_channels: int = 3):
		super().__init__(layers=[4, 4, 4], width=width, num_classes=num_classes, num_input_channels=num_input_channels)
