
import subprocess


def get_current_git_hash() -> str:
	"""
		Return the current git hash in the current directory.

		:returns: The git hash. If an error occurs, returns 'UNKNOWN'.
	"""
	try:
		git_hash = subprocess.check_output(['git', 'describe', '--always'])
		git_hash = git_hash.decode('UTF-8').replace('\n', '')
		return git_hash
	except (subprocess.CalledProcessError, PermissionError):
		return 'UNKNOWN'
