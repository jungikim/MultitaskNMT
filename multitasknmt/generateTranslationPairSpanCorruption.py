import tensorflow as tf
from multitasknmt.generateSpanCorruption import generateSpanCorruption

class generateTranslationPairSpanCorruption(generateSpanCorruption):

    def t5_single_example_denoise(self, features, seed, *,
                                  noise_mask_fn=None, inputs_fn=None, targets_fn=None, training=False):

        if noise_mask_fn == None:   noise_mask_fn = self.random_spans_noise_mask
        if inputs_fn == None:       inputs_fn=self.noise_span_to_unique_sentinel
        if targets_fn == None:      targets_fn=self.nonnoise_span_to_unique_sentinel

        seeds = tf.unstack(tf.random.experimental.stateless_split(seed, 6))

        srcFeatures, tgtFeatures = features
        text = tf.strings.join([srcFeatures, tgtFeatures], separator=' ')
        tokens = self.tokenizer.tokenize(text, training=training)

        noise_mask = noise_mask_fn(tf.size(tokens), seeds=seeds[:2])
        inputs = inputs_fn(tokens, noise_mask, seeds=seeds[2:4])
        targets = targets_fn(tokens, noise_mask, seeds=seeds[4:6])

        return (self.tokenizer.detokenize(inputs),
                self.tokenizer.detokenize(targets))
