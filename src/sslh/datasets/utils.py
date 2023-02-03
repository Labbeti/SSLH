#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Optional, Protocol, Sized, runtime_checkable

from torch.utils.data.dataset import Dataset


@runtime_checkable
class SizedDataset(Protocol):
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("Protocal abstract method.")

    def __len__(self) -> int:
        raise NotImplementedError("Protocal abstract method.")


class DatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self._dataset = dataset

    def __getitem__(self, idx: Any) -> Any:
        return self._dataset.__getitem__(idx)

    def __len__(self) -> int:
        if not isinstance(self._dataset, Sized):
            raise NotImplementedError(
                f'Wrapped dataset "{str(type(self._dataset))}" is not Sized (it does not have the method "__len__").'
            )
        return len(self._dataset)

    def unwrap(self, recursive: bool = False) -> Dataset:
        """
        :param recursive: If True and the dataset wrapped is another TransformDataset, unwrap until the wrapped
                element is not a TransformDataset. (default: False)
        """
        if not recursive:
            return self._dataset
        else:
            dataset = self._dataset
            while isinstance(dataset, DatasetWrapper):
                dataset = dataset.unwrap()
            return dataset


class TransformDataset(DatasetWrapper):
    def __init__(
        self,
        dataset: Dataset,
        transform: Optional[Callable],
        index: Optional[int] = None,
    ) -> None:
        """
        Wrap a dataset by applying a post-transform to item get by the method '__getitem__'.

        :param dataset: The dataset to wrap.
        :param transform: The callable transform to apply.
        :param index: The index of the element to apply the transform.
                If None, apply the transform to the complete item.
        """
        super().__init__(dataset)
        self._transform = transform
        self._index = index

        if self._transform is None:
            self._post_fn = lambda x: x
        elif self._index is None:
            self._post_fn = self._transform
        else:

            def post_fn(item: tuple) -> tuple:
                item = list(item)  # type: ignore
                item[self._index] = self._transform(item[self._index])  # type: ignore
                item = tuple(item)
                return item

            self._post_fn = post_fn

    def __getitem__(self, idx: Any) -> Any:
        return self._post_fn(self._dataset[idx])


class NoLabelDataset(DatasetWrapper):
    def __init__(
        self, dataset: Dataset, index_label: int = 1, keep_tuple: bool = False
    ) -> None:
        """
        Wrapper of Dataset which remove label from dataset by getting only the batch.

        :param dataset: The dataset to wrap.
        :param index_label: The index of the data to keep when after calling getitem() method of the dataset wrapped.
        """
        super().__init__(dataset)
        self.index_label = index_label
        self.keep_tuple = keep_tuple

    def __getitem__(self, index: Any) -> Any:
        item: tuple = super().__getitem__(index)
        item = tuple(elt for i, elt in enumerate(item) if i != self.index_label)
        if len(item) == 1 and not self.keep_tuple:
            return item[0]
        else:
            return item


def cache_feature(func):
    def decorator(*args, **kwargs):
        key = ",".join(map(str, args))

        if key not in decorator.cache:
            decorator.cache[key] = func(*args, **kwargs)

        return decorator.cache[key]

    decorator.cache = dict()
    return decorator


class CachedDataset(DatasetWrapper):
    def __init__(self, dataset: SizedDataset) -> None:
        super().__init__(dataset)  # type: ignore
        self._cache = [None for _ in range(len(dataset))]

    def __getitem__(self, index: int) -> Any:
        if self._cache[index] is None:
            item = super().__getitem__(index)
            self._cache[index] = item
        else:
            item = self._cache[index]
        return item

    def __len__(self) -> int:
        return super().__len__()
