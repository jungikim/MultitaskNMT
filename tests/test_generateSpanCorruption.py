import numpy as np
import tensorflow as tf
from multitasknmt.generateSpanCorruption import generateSpanCorruption


class GenerateSpanCorruptionTest(tf.test.TestCase):
    def _get_processor(self):
        processor = generateSpanCorruption()
        return processor

    def _get_tokenized(self, tokenizer):
        tokens=tokenizer.tokenize('A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 1 2 3 4')
        return tokens

    def test_noise_span(self):
        processor = self._get_processor()
        tokens = self._get_tokenized(processor.tokenizer)
        print('tokenize()->random_spans_noise_mask()->noise_span_to_unique_sentinel()->nonnoise_span_to_unique_sentinel()')
        seeds = np.zeros(shape=(2,2),dtype=np.int32)
        noiseMask = processor.random_spans_noise_mask(length=30, seeds=seeds)
        twns = processor.noise_span_to_unique_sentinel(tokens, noiseMask, seeds=seeds)
        twnns = processor.nonnoise_span_to_unique_sentinel(tokens, noiseMask, seeds=seeds)
        twns = processor.tokenizer.detokenize(twns).numpy().decode('utf-8')
        twnns = processor.tokenizer.detokenize(twnns).numpy().decode('utf-8')
        print('noise mask: %s' % (noiseMask))
        print('noise_span_to_unique_sentinel: %s' % (twns))
        print('nonnoise_span_to_unique_sentinel: %s' % (twnns))


    def test_t5_denoise(self):
        processor = self._get_processor()
        tokens = self._get_tokenized(processor.tokenizer)
        print('t5_denoise("%s")' % (tokens))
        dataset = tf.data.Dataset.from_tensor_slices([processor.tokenizer.detokenize(tokens)])
        dataset = processor.t5_denoise(dataset)
        for (src,tgt) in dataset:
            print(src.numpy().decode('utf-8'))
            print(tgt.numpy().decode('utf-8'))


    def test_t5_single_example_denoise(self):
        processor = self._get_processor()
        tokens = self._get_tokenized(processor.tokenizer)
        print('t5_single_example_denoise()')
        (src,tgt) = processor.t5_single_example_denoise(processor.tokenizer.detokenize(tokens),
                                                        seed=tf.zeros(shape=(2),dtype=tf.dtypes.int32))
        print(src.numpy().decode('utf-8'))
        print(tgt.numpy().decode('utf-8'))
        print()


    def test_t5_denoise_file(self):
        processor = self._get_processor()
        file='data/testinput.txt'
        print('test_t5_denoise_file(%s)' % (file))
        dataset = tf.data.TextLineDataset(filenames=file)
        dataset = processor.t5_denoise(dataset)
        for (s,t) in dataset:
            print(s.numpy().decode('utf-8'))
            print(t.numpy().decode('utf-8'))
        print()


if __name__ == "__main__":
    tf.test.main()
 