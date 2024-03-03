# the following denoising code is from Google's text-to-text-transfer-transformer (T5)'s preprocessor
# https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/preprocessors.py#L2708
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
from opennmt import tokenizers
from multitasknmt.generateSpanCorruption_seqio_util import seqio_map_over_dataset, seqio_stateless_shuffle


_sentinelTokens = ['｟NOT_SENTINEL｠']
_sentinelTokens.extend([ '｟SENTINEL＃'+str(idx)+'｠' for idx in range(1,301) ])


class generateSpanCorruption():
    def __init__(self, noise_density=0.15,
                       mean_noise_span_length=3.0,
                       sentinelTokens = _sentinelTokens
                ):

        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.tokenizer = tokenizers.SpaceTokenizer()
        self.sentinelTokens = tf.constant(sentinelTokens)



    def t5_denoise(self, dataset, noise_mask_fn=None, inputs_fn=None, targets_fn=None, num_threads=4, training=False):
        if noise_mask_fn == None:   noise_mask_fn = self.random_spans_noise_mask
        if inputs_fn == None:       inputs_fn=self.noise_span_to_unique_sentinel
        if targets_fn == None:      targets_fn=self.nonnoise_span_to_unique_sentinel

        @seqio_map_over_dataset(num_seeds=1, num_parallel_calls=num_threads)
        def my_fn(features, seed):
            return self.t5_single_example_denoise(
                features,
                seed,
                noise_mask_fn=noise_mask_fn,
                inputs_fn=inputs_fn,
                targets_fn=targets_fn,
                training=training)
        return my_fn(dataset)#.filter(lambda x, y: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))


    def t5_single_example_denoise(self, features, seed, *,
                                  noise_mask_fn=None, inputs_fn=None, targets_fn=None, training=False):

        if noise_mask_fn == None:   noise_mask_fn = self.random_spans_noise_mask
        if inputs_fn == None:       inputs_fn=self.noise_span_to_unique_sentinel
        if targets_fn == None:      targets_fn=self.nonnoise_span_to_unique_sentinel

        seeds = tf.unstack(tf.random.experimental.stateless_split(seed, 6))

        tokens = self.tokenizer.tokenize(features, training=training)

        noise_mask = noise_mask_fn(tf.size(tokens), seeds=seeds[:2])
        inputs = inputs_fn(tokens, noise_mask, seeds=seeds[2:4])
        targets = targets_fn(tokens, noise_mask, seeds=seeds[4:6])

        return (self.tokenizer.detokenize(inputs),
                self.tokenizer.detokenize(targets))


    def noise_span_to_unique_sentinel(self, tokens, noise_mask, seeds):
        del seeds
        prev_token_is_noise     = tf.pad(noise_mask[:-1], [[1, 0]])
        first_noise_tokens      = tf.logical_and(noise_mask, tf.logical_not(prev_token_is_noise))
        subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)

        # Jungi Kim: added clipping id to len(sentinelTokens)-1
        sentinel = tf.gather(self.sentinelTokens,
                             tf.clip_by_value(
                                 tf.cumsum(tf.cast(first_noise_tokens, tf.int32)),
                                 0, # min
                                 len(self.sentinelTokens) - 1 # max
                             )
                            )

        tokens = tf.where(first_noise_tokens, sentinel, tokens)
        tokens = tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))
        return tokens


    def nonnoise_span_to_unique_sentinel(self, tokens, noise_mask, seeds):
        return self.noise_span_to_unique_sentinel(tokens, tf.logical_not(noise_mask), seeds=seeds)


    def random_spans_noise_mask(self, length, seeds, random_roll=True):
        if self.noise_density == 0.0:
            return tf.zeros(length, tf.bool)

        orig_length = length # increase length to avoid degeneracy
        length = tf.maximum(length, 2)

        def to_int(x): return tf.cast(x, tf.int32)
        def to_float(x): return tf.cast(x, tf.float32)

        num_noise_tokens = to_int(tf.round(to_float(length) * self.noise_density))
        num_noise_tokens = tf.minimum(tf.maximum(num_noise_tokens, 1), length - 1)
        num_noise_spans = to_int(tf.round(to_float(num_noise_tokens) / self.mean_noise_span_length))
        num_noise_spans = tf.maximum(num_noise_spans, 1)
        # avoid degeneracy by ensuring positive numbers of noise, nonnoise tokens, and noise spans

        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments, seed):
            first_in_segment = tf.pad(seqio_stateless_shuffle(
                                        to_int(tf.range(num_items - 1) < num_segments - 1), seed),
                                      [[1, 0]])
            segment_id = tf.cumsum(first_in_segment)
            segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans, seeds[0])
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans, seeds[1])
        interleaved_span_lengths = tf.reshape(
                                        tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
                                        [num_noise_spans * 2])
        span_starts = tf.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = tf.math.unsorted_segment_sum(tf.ones_like(span_starts), span_starts, length)
        span_num = tf.cumsum(span_start_indicator)
        is_noise = tf.equal(span_num % 2, 1)

        mask = is_noise[:orig_length]

        if random_roll:
            roll_seed = (seeds[0][0]+seeds[1][1], seeds[0][1]-seeds[1][0])  # new seed.
            # Roll the mask by a random offset e.g. for offset=2: [1,2,3,4] => [3,4,1,2]
            offset = tf.random.stateless_uniform(
                [1], seed=roll_seed, dtype=tf.int32, minval=0, maxval=length)[0]
            mask = tf.roll(mask, shift=offset, axis=0)

        return mask
