
from setuptools import setup, find_packages


install_requires = [
	"torch~=1.7.0",
	"torchaudio~=0.7.0",
	"torchvision~=0.8.1",
	"tensorboard",
	"matplotlib",
	"numpy",
	"librosa",
	"h5py",
	"pandas",
	"advertorch",
	"augmentation_utils @ git+https://github.com/leocances/augmentation_utils",
	"metric_utils @ git+https://github.com/leocances/pytorch_metrics",
	"ubs8k @ git+https://github.com/leocances/UrbanSound8K@new_data_management",
	"MLU @ git+https://github.com/Labbeti/MLU@dev",
]

setup(
	name="sslh",
	version="1.3.0",
	packages=find_packages(),
	url="https://github.com/Labbeti/SSLH",
	license="",
	author="Etienne Labbé",
	author_email="etienne.labbe31@gmail.com",
	description="Semi Supervised Learning with Holistic methods.",
	python_requires=">=3.8.5",
	install_requires=install_requires,
	include_package_data=True,
)
