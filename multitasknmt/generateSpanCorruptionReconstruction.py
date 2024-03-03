import tensorflow as tf
from multitasknmt.generateSpanCorruption import generateSpanCorruption

class generateSpanCorruptionReconstruction(generateSpanCorruption):

    def t5_single_example_denoise(self, features, seed, *,
                                  noise_mask_fn=None, inputs_fn=None, targets_fn=None, training=False):

        if noise_mask_fn == None:   noise_mask_fn = self.random_spans_noise_mask
        if inputs_fn == None:       inputs_fn=self.noise_span_to_unique_sentinel

        seeds = tf.unstack(tf.random.experimental.stateless_split(seed, 6))

        tokens = self.tokenizer.tokenize(features, training=training)
        noise_mask = noise_mask_fn(tf.size(tokens), seeds=seeds[:2])
        inputs = inputs_fn(tokens, noise_mask, seeds=seeds[2:4])

        return (tf.strings.join(['reconstruct: ', self.tokenizer.detokenize(inputs)]),
                features)
