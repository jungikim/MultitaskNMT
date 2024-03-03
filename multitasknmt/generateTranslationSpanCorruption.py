import tensorflow as tf
from multitasknmt.generateSpanCorruption import generateSpanCorruption

class generateTranslationSpanCorruption(generateSpanCorruption):

    def t5_single_example_denoise(self, features, seed, *,
                                  noise_mask_fn=None, inputs_fn=None, targets_fn=None, training=False):

        if noise_mask_fn == None:   noise_mask_fn = self.random_spans_noise_mask
        if inputs_fn == None:       inputs_fn=self.noise_span_to_unique_sentinel
        if targets_fn == None:      targets_fn=self.nonnoise_span_to_unique_sentinel

        seeds = tf.unstack(tf.random.experimental.stateless_split(seed, 6))

        srcFeatures, tgtFeatures = features
        srcTokens = self.tokenizer.tokenize(srcFeatures, training=training)

        noise_mask = noise_mask_fn(tf.size(srcTokens), seeds=seeds[:2])
        inputs = inputs_fn(srcTokens, noise_mask, seeds=seeds[2:4])
        targets = targets_fn(srcTokens, noise_mask, seeds=seeds[4:6])

        inputs = tf.strings.join([self.tokenizer.detokenize(inputs), tgtFeatures], separator=' ')

        return (inputs,
                self.tokenizer.detokenize(targets))
