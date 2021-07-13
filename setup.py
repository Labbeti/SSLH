#!/usr/bin/python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


class PostDevelopCommand(develop):
	def run(self):
		super().run()
		check_call(['bash', 'post_install.sh'])


class PostInstallCommand(install):
	def run(self):
		super().run()
		check_call(['bash', 'post_install.sh'])


setup(
	name='sslh',
	version='2.2.1',
	packages=find_packages(),
	url='https://github.com/Labbeti/SSLH',
	license="",
	author='Etienne Labbé',
	author_email='etienne.labbe31@gmail.com',
	description='Semi Supervised Learning with Holistic methods.',
	python_requires='>=3.8.5',
	install_requires=requirements,
	include_package_data=True,
	cmdclass={
		'develop': PostDevelopCommand,
		'install': PostInstallCommand,
	}
)
