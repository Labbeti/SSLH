
import os
import os.path as osp
import torch

from abc import ABC
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from typing import List, Optional, Union, Tuple


class CheckPointABC(ABC):
    """
        Abstract class for CheckPoint which save best models.
    """

    def step(self, new_value: float):
        """ Method called to update the best model with the value returned by a metric. """
        raise NotImplementedError("Abstract method")

    def load_best_state(self, model: Optional[Module], optim: Optional[Optimizer]) -> (float, int):
        """ Loads the best state saved by the checkpoint. """
        raise NotImplementedError("Abstract method")

    def is_saved(self) -> bool:
        """ Return True if the checkpoint has saved at least 1 model in a file. """
        raise NotImplementedError("Abstract method")


class CheckPoint(CheckPointABC):
    """
        Main class for saving the best model and optimizer in a file.
    """

    def __init__(
        self,
        models: Union[Module, List[Module]],
        optimizer: Optimizer,
        dirpath: str,
        filename_prefix: str,
        mode: str = "max",
        save_in_file: bool = True,
        verbose: int = 1,
    ):
        """
            Save best model in a file.

            :param models: The module or a list of module to track and save.
            :param optimizer: The Optimizer to track and save.
            :param dirpath: The path to the directory where saving files.
            :filename_prefix: The prefix of the files to save.
                The files will be named "{filename_prefix}_rank_{rank}_epoch_{epoch}.torch".
            :param mode: The mode to use for comparing performances. Can be "max" or "min". (default: "max")
            :param save_in_file: Boolean for saving or not the models in files. (default: True)
            :param verbose: The verbose level to use. (default: 1)
                Use 0 for deactivate prints and
                1 for printing info when the performance is reached.
        """
        assert mode in ["min", "max"], "Available modes are \"min\" and \"max\"."
        self.models = [models] if isinstance(models, Module) else models
        self.optimizer = optimizer
        self.dirpath = dirpath
        self.filename_prefix = filename_prefix
        self.mode = mode
        self.save_in_file = save_in_file
        self.verbose = verbose

        self._best_metric = float("-inf") if self.mode == "max" else float("inf")
        self._epoch = 0
        self._prev_filepath = None

    def step(self, new_value: float):
        self._epoch += 1

        if self._check_is_better(new_value):
            self._best_metric = new_value

            if not self.save_in_file:
                if self.verbose >= 1:
                    print("Best performance reached !")
            else:
                filename = "{:s}_{:d}.torch".format(self.filename_prefix, self._epoch)
                filepath = osp.join(self.dirpath, filename)
                if self.verbose >= 1:
                    print("Best performance reached ! Saving the model in \"{:s}\"...".format(filename))
                best_state = {
                    "nb_models": len(self.models),
                    "models": [
                        {
                            "name": model.__class__.__name__,
                            "state_dict": model.state_dict()
                        }
                        for model in self.models
                    ],
                    "optimizer": self.optimizer.state_dict(),
                    "best_metric": self._best_metric,
                    "epoch": self._epoch,
                }
                if self._prev_filepath is not None:
                    os.remove(self._prev_filepath)
                torch.save(best_state, filepath)
                self._prev_filepath = filepath

    def load_best_state(self, model: Optional[Module], optim: Optional[Optimizer]) -> (float, int):
        if self.is_saved():
            return load_state(self._prev_filepath, model, optim)
        else:
            raise RuntimeError("Cannot load best state if no state has been saved for this checkpoint.")

    def is_saved(self) -> bool:
        return self._prev_filepath is not None

    def reset(self, remove_model_file: bool = False):
        """
            Reset the checkpoint state.
            :param remove_model_file: If True, remove the best model file stored by the checkpoint. (default: False)
        """
        self._best_metric = float("-inf") if self.mode == "max" else float("inf")
        self._epoch = 0

        if remove_model_file and self.is_saved():
            os.remove(self._prev_filepath)
            self._prev_filepath = None

    def _check_is_better(self, new_value: float) -> bool:
        """ Returns True if the new values must be stored with the current model. """
        if self.mode == "max":
            return new_value >= self._best_metric
        else:
            return new_value <= self._best_metric


class CheckPointMultiple(CheckPointABC):
    """
        Main class for saving the K best models and optimizer in several files.
    """

    def __init__(
        self,
        models: Union[Module, List[Module]],
        optimizer: Optimizer,
        dirpath: str,
        filename_prefix: str,
        mode: str = "max",
        save_in_file: bool = True,
        verbose: int = 1,
        nb_bests: int = 5,
    ):
        """
            Save the K best models in several files.

            Note: The best models has rank 0, the second the rank 1, ... until the best (K-1)th model.
                The first epoch is 1 and not 0 for being consistent with the logs.

            :param models: The module or a list of module to track and save.
            :param optimizer: The Optimizer to track and save.
            :param dirpath: The path to the directory where saving files.
            :filename_prefix: The prefix of the files to save.
                The files will be named "{filename_prefix}_rank_{rank}_epoch_{epoch}.torch".
            :param mode: The mode to use for comparing performances. Can be "max" or "min". (default: "max")
            :param save_in_file: Boolean for saving or not the models in files. (default: True)
            :param verbose: The verbose level to use. (default: 1)
                Use 0 for deactivate prints,
                1 for printing info when the performance is reached and
                2 for print the renames and deletes of the current files.
            :param nb_bests: Number of the best models to save. Must be >= 1. (default: 5)
        """
        assert mode in ["min", "max"], "Available modes are \"min\" and \"max\"."
        self.models = [models] if isinstance(models, Module) else models
        self.optimizer = optimizer
        self.dirpath = dirpath
        self.filename_prefix = filename_prefix
        self.mode = mode
        self.save_in_file = save_in_file
        self.verbose = verbose
        self.nb_bests = nb_bests

        self._bests = []
        self._epoch = 0

    def step(self, new_value: float):
        self._epoch += 1

        if self._check_is_better(new_value):
            new_best = {
                "value": new_value,
                "epoch": self._epoch,
                "filepath": None,
            }
            self._bests.append(new_best)
            self._bests.sort(key=lambda info: info["value"], reverse=self.mode == "max")

            # Remove files with values worst than the last one (ie self._bests[-1]["value"])
            self.__remove_worst_model()

            if not self.save_in_file:
                if self.verbose >= 1:
                    print("New performance reached !")
            else:
                # Rename files for keeping the names of the models coherent with their current ranks.
                new_filepath, new_rank = self.__rename_best_models()

                if new_filepath is None or new_rank is None:
                    raise RuntimeError("Invalid state for checkpoint : found None filepath for new model file.")

                if self.verbose >= 1:
                    print(f"New performance reached ! Saving the model of epoch {self._epoch} (rank={new_rank}/{self.nb_bests})...")

                # Save the current model
                self.__save_new_model(new_filepath, new_value)

    def load_best_state(self, model: Optional[Module], optim: Optional[Optimizer]) -> (float, int):
        if self.is_saved():
            return load_state(self._bests[0]["filepath"], model, optim)
        else:
            return 0.0, 0

    def is_saved(self) -> bool:
        return len(self._bests) > 0 and self._bests[0]["filepath"] is not None

    def get_best_performances(self) -> List[Tuple[float, int]]:
        """ Returns the list of (values, epoch) for best models list in diminishing order. """
        return [(info["value"], info["epoch"]) for info in self._bests]

    def _check_is_better(self, new_value: float) -> bool:
        """ Returns True if the new values must be stored with the current model. """
        if len(self._bests) < self.nb_bests:
            return True
        elif self.mode == "max":
            return new_value > self._bests[-1]["value"]
        else:  # "min"
            return new_value < self._bests[-1]["value"]

    def __remove_worst_model(self):
        while len(self._bests) > self.nb_bests:
            filepath_last = self._bests[-1]["filepath"]
            if filepath_last is not None and osp.isfile(filepath_last):
                if self.verbose >= 2:
                    print(f"Remove previous {osp.basename(filepath_last)}...")
                os.remove(filepath_last)
            self._bests.pop()

    def __rename_best_models(self) -> (str, int):
        new_filepath = None
        new_rank = None

        # Rename models for update their rank in their filename
        for i, best in enumerate(self._bests):
            old_filepath = best["filepath"]
            filename = "{:s}_rank_{:d}_epoch_{:d}_value_{:.6f}.torch".format(self.filename_prefix, i, best["epoch"], best["value"])
            filepath = osp.join(self.dirpath, filename)

            best["filepath"] = filepath

            if old_filepath is not None and osp.isfile(old_filepath):
                if old_filepath != filepath:
                    if self.verbose >= 2:
                        print(f"Rename {osp.basename(old_filepath)} to {osp.basename(filepath)}...")
                    os.rename(old_filepath, filepath)
            else:
                new_filepath = filepath
                new_rank = i

        return new_filepath, new_rank

    def __save_new_model(self, new_filepath: str, new_value: float):
        best_state = {
            "nb_models": len(self.models),
            "models": [
                {
                    "name": model.__class__.__name__,
                    "state_dict": model.state_dict()
                }
                for model in self.models
            ],
            "optimizer": self.optimizer.state_dict(),
            "best_metric": new_value,
            "epoch": self._epoch,
        }
        torch.save(best_state, new_filepath)
        if self.verbose >= 2:
            best_values = ", ".join(["{:.3f}".format(info["value"]) for info in self._bests])
            print(f"Current best values : {best_values}")


class NoCheckPoint(CheckPointABC):
    """
        Simple class which implements the abstract CheckpointABC for not saving model(s).
    """

    def step(self, new_value: float):
        pass

    def load_best_state(self, model: Optional[Module], optim: Optional[Optimizer]) -> (float, int):
        return 0.0, 0

    def is_saved(self) -> bool:
        return False


class CheckPointEveryKSteps(CheckPointABC):
    def __init__(
        self,
        models: Union[Module, List[Module]],
        optimizer: Optimizer,
        dirpath: str,
        filename_prefix: str,
        save_every_k_steps: int,
    ):
        self.models = [models] if isinstance(models, Module) else models
        self.optimizer = optimizer
        self.dirpath = dirpath
        self.filename_prefix = filename_prefix
        self.save_every_k_steps = save_every_k_steps

        self._counter = 0
        self._save_count = 0

    def step(self, new_value: float):
        self._counter += 1

        if self._counter >= self.save_every_k_steps:
            self._save_count += 1
            self._counter = 0

            filename = "{:s}_{:d}.torch".format(self.filename_prefix, self._save_count)
            filepath = osp.join(self.dirpath, filename)

            best_state = {
                "nb_models": len(self.models),
                "models": [
                    {
                        "name": model.__class__.__name__,
                        "state_dict": model.state_dict()
                    }
                    for model in self.models
                ],
                "optimizer": self.optimizer.state_dict(),
                "save_count": self._save_count,
            }
            torch.save(best_state, filepath)

    def load_best_state(self, model: Optional[Module], optim: Optional[Optimizer]) -> (float, int):
        raise RuntimeError("Cannot load best state for this checkpoint.")

    def is_saved(self) -> bool:
        return False


def load_state(
    filepath: str,
    models: Union[None, Module, List[Module]],
    optim: Optional[Optimizer] = None
) -> (float, int):
    """
        Update model(s) and optimizer and returns the best metric value and the best epoch.

        :param filepath: The path to the torch file where models are stored by CheckPoint or CheckPointMultiple classes.
        :param models: The module or the list of modules to update with the values stored in file.
            If None, the values of the modules will be ignored.
        :param optim: The optimizer to update with the values stored in file. (default: None)
            If None, the values of the optimizer will be ignored.
    """
    data = torch.load(filepath)

    if models is not None:
        if isinstance(models, Module):
            models = [models]
        nb_models = data["nb_models"]
        if nb_models != len(models):
            raise RuntimeError("Found {:d} in file but provide only {:d} models in argument.".format(nb_models, len(models)))

        for i, model_data in enumerate(data["models"]):
            models[i].load_state_dict(model_data["state_dict"])

    if optim is not None:
        optim.load_state_dict(data["optimizer"])

    return data["best_metric"], data["epoch"]
