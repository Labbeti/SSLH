#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os.path as osp

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call

from typing import List


def read_version() -> str:
	fpath = osp.join(osp.dirname(__file__), 'sslh', '__init__.py')
	version = 'unknown'
	with open(fpath, 'r') as file:
		for line in file.readlines():
			if line.startswith('__version__'):
				# Expected line : __version__ = 'VERSION'
				line_split = line.split("'")
				version = line_split[1]
				break
	return version


def read_requirements() -> List[str]:
	fpath = osp.join(osp.dirname(__file__), 'requirements.txt')
	with open(fpath) as file:
		requirements = file.read().splitlines()
		return requirements


class PostDevelopCommand(develop):
	def run(self):
		super().run()
		fpath_post_setup = osp.join(osp.dirname(__file__), 'post_setup.sh')
		check_call(['bash', fpath_post_setup])


class PostInstallCommand(install):
	def run(self):
		super().run()
		fpath_post_setup = osp.join(osp.dirname(__file__), 'post_setup.sh')
		check_call(['bash', fpath_post_setup])


setup(
	name='sslh',
	version=read_version(),
	packages=find_packages(),
	url='https://github.com/Labbeti/SSLH',
	license='',
	author='Etienne Labbé',
	author_email='etienne.labbe31@gmail.com',
	description='Semi Supervised Learning with Holistic methods.',
	python_requires='>=3.8.5',
	install_requires=read_requirements(),
	include_package_data=True,
	cmdclass={
		'develop': PostDevelopCommand,
		'install': PostInstallCommand,
	}
)
