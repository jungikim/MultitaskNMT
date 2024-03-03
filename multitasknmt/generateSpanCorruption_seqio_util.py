# the following code is an excerpt from Google's SeqIO project:
# https://github.com/google/seqio/blob/77f9be7d6be30ecfa21e8fa57e9ff5af6af22a02/seqio/utils.py
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import numpy as np
import dataclasses
import functools
import inspect
from typing import Any, Callable, Mapping, Optional


def seqio_map_over_dataset(fn=None, *, num_seeds=None, num_parallel_calls=tf.data.experimental.AUTOTUNE):
  def map_without_seeds(fn):
    @functools.wraps(fn)
    def wrapped_fn(ds, *args, **kwargs):
      return _GrainMapFn(fn, num_parallel_calls)(ds, *args, **kwargs)

    return wrapped_fn

  def map_with_seeds(fn):
    @functools.wraps(fn)
    def wrapped_fn(ds, *args, **kwargs):
      return _GrainRandomMapFn(fn, num_seeds, num_parallel_calls)(
          ds, *args, **kwargs
      )

    return wrapped_fn

  wrapper = map_without_seeds if num_seeds is None else map_with_seeds
  return wrapper if fn is None else wrapper(fn)


def seqio_stateless_shuffle(value, seed):
    flat_value = tf.reshape(value, [-1])
    indices = tf.argsort(tf.random.stateless_uniform(tf.shape(flat_value), seed=seed))
    flat_shuffle = tf.gather(flat_value, indices)
    return tf.reshape(flat_shuffle, tf.shape(value))


_MapTransform = object
_RandomMapTransform = object
_SPECIAL_KWARGS = ("sequence_length", "output_features")
_NEXT_MAP_SEED = None

def add_kwargs_to_transform(transform, **kwargs):
  is_dataclass = dataclasses.is_dataclass(transform)
  if is_dataclass:
    avaialabe_arg_names = [f.name for f in dataclasses.fields(transform)]
  else:
    avaialabe_arg_names = set(inspect.signature(transform).parameters.keys())
  kwargs = {k: v for k, v in kwargs.items() if k in avaialabe_arg_names}
  if not kwargs:
    return transform
  if is_dataclass:
    return dataclasses.replace(transform, **kwargs)
  return functools.partial(transform, **kwargs)


@dataclasses.dataclass
class _GrainRandomMapFn(_RandomMapTransform):
  map_fn: Callable[..., Any]
  num_seeds: int
  num_parallel_calls: int = tf.data.AUTOTUNE

  sequence_length: Optional[Mapping[str, int]] = None
  output_features: Optional[Mapping[str, Any]] = None

  def _map_fn_with_special_kwargs(self, *args, **kwargs):
    special_kwargs = {
        k: getattr(self, k)
        for k in _SPECIAL_KWARGS
        if getattr(self, k) is not None
    }
    map_fn = add_kwargs_to_transform(self.map_fn, **special_kwargs)
    return map_fn(*args, **kwargs)

  def random_map(self, element, rng: tf.Tensor):
    if self.num_seeds == 1:
      return self._map_fn_with_special_kwargs(element, seed=rng)
    rngs = tf.random.experimental.stateless_split(rng, self.num_seeds)
    rngs = tf.unstack(rngs)
    return self._map_fn_with_special_kwargs(element, seeds=rngs)

  def __call__(self, dataset: tf.data.Dataset, *args, **kwargs):
    global _NEXT_MAP_SEED
    if _NEXT_MAP_SEED is None:
      random_ds_seeds = ((None, None),) * self.num_seeds
    else:
      random_ds_seeds = np.arange(
          _NEXT_MAP_SEED, _NEXT_MAP_SEED + 2 * self.num_seeds
      ).reshape(-1, 2)
      random_ds_seeds = tuple(tuple(s) for s in random_ds_seeds)
      _NEXT_MAP_SEED += 2 * self.num_seeds
    seed_datasets = tf.nest.map_structure(
        tf.data.experimental.RandomDataset, random_ds_seeds
    )

    def map_fn(element, seeds):
      if self.num_seeds == 1:
        return self._map_fn_with_special_kwargs(
            element, seed=seeds[0], *args, **kwargs
        )
      return self._map_fn_with_special_kwargs(
          element, seeds=seeds, *args, **kwargs
      )

    return tf.data.Dataset.zip((dataset, seed_datasets)).map(
        map_fn, num_parallel_calls=self.num_parallel_calls
    )


@dataclasses.dataclass
class _GrainMapFn(_MapTransform):
  map_fn: Callable[..., Any]
  num_parallel_calls: int

  sequence_length: Optional[Mapping[str, int]] = None
  output_features: Optional[Mapping[str, Any]] = None

  def _map_fn_with_special_kwargs(self, *args, **kwargs):
    special_kwargs = {
        k: getattr(self, k)
        for k in _SPECIAL_KWARGS
        if getattr(self, k) is not None
    }
    map_fn = add_kwargs_to_transform(self.map_fn, **special_kwargs)
    return map_fn(*args, **kwargs)

  def map(self, element):
    return self._map_fn_with_special_kwargs(element)

  def __call__(
      self, dataset: tf.data.Dataset, *args, **kwargs
  ) -> tf.data.Dataset:
    return dataset.map(
        lambda x: self._map_fn_with_special_kwargs(x, *args, **kwargs),
        num_parallel_calls=self.num_parallel_calls,
    )
