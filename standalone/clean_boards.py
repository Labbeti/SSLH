
import os
import os.path as osp
import re
import shutil

from argparse import ArgumentParser
from typing import Optional


def get_size(path: str, recursive: bool = True) -> int:
	# Get size in bytes
	if osp.isfile(path):
		return osp.getsize(path)
	elif osp.islink(path):
		return 0
	elif osp.isdir(path):
		if recursive:
			size = 0
			for elt in os.listdir(path):
				size += get_size(osp.join(path, elt))
			return size
		else:
			return 0
	else:
		raise RuntimeError("Unknown element \"{:s}\".".format(osp.join(path)))


def check_if_remove(path: str, minimal_size: int, pattern: Optional[str]) -> bool:
	return osp.isdir(path) and (
		len(os.listdir(path)) == 0 or
		pattern is not None and re.search(pattern, path) is not None or
		get_size(path) < minimal_size
	)


def main():
	parser = ArgumentParser(description="Script for cleaning tensorboard results. "
										"Condition of remove are : (folder empty or pattern found or folder size < threshold)")
	parser.add_argument("--tensorboard_roots", "-tr", type=str, nargs="+",
						default=[osp.join("..", "results", "tensorboard")])
	parser.add_argument("--minimal_size", "-ms", type=int, default=0)
	parser.add_argument("--pattern", "-p", type=str, default=None)
	parser.add_argument("--all", "-a", default=False, action="store_true")
	args = parser.parse_args()

	for tensorboard_root in list(args.tensorboard_roots):
		boards = os.listdir(tensorboard_root)
		boards = [board for board in boards if osp.isdir(osp.join(tensorboard_root, board))]

		print("Check tensorboard root \"{:s}\" with {:d} elements.".format(tensorboard_root, len(boards)))

		to_remove = []
		for i, board_name in enumerate(sorted(boards)):
			board_path = osp.join(tensorboard_root, board_name)
			remove = args.all or check_if_remove(board_path, args.minimal_size, args.pattern)
			size = get_size(board_path)
			print("{:2d} - Tensorboard: (Remove:{:4s}, Size:{:6d}, Name:\"{:s}\").".format(
				i + 1, "YES" if remove else "no", size, board_name))

			if remove:
				to_remove.append(board_path)

		if len(to_remove) > 0:
			print("Remove {:d} element(s) ?".format(len(to_remove)))
			user_in = None
			while user_in not in ["yes", "no"]:
				user_in = input("(yes/no)> ")

			if user_in == "yes":
				for board_path in to_remove:
					print(f"Removing {board_path}...")
					shutil.rmtree(board_path)
				print("Done.")
		else:
			print("No element to remove found.")

	print("Terminated.")


if __name__ == "__main__":
	main()
