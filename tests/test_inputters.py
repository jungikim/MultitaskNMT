import yaml
import tensorflow as tf
import opennmt as onmt
from multitasknmt.spanCorruption_inputter import SpanCorruptionInputter
from multitasknmt.translationPairSpanCorruption_inputter import TranslationPairSpanCorruptionInputter
from multitasknmt.translationSpanCorruption_inputter import TranslationSpanCorruptionInputter
from multitasknmt.examplesMTPrompt_inputter import ExamplesMTPromptInputter
from multitasknmt.sc_tpsc_tsc_mt_inputter import ScTpscTscMtInputter

def _get_config():
  config = yaml.safe_load("""
  data:
    train_features_file:
      - data/testinput_en.txt
      - data/testinput_ko.txt
    train_labels_file:
      - data/testinput_ko.txt
      - data/testinput_en.txt
    eval_features_file: data/testinput_ko.txt
    eval_labels_file: data/testinput_en.txt

    source_vocabulary: data/enkovocab/pyonmt_bpe_enko_32k_min2_casemarkup.vocab
    target_vocabulary: data/enkovocab/pyonmt_bpe_enko_32k_min2_casemarkup.vocab

    source_tokenization:
      type: OpenNMTTokenizer
      params:
        joiner_annotate: true
        mode: aggressive
        no_substitution: true
        segment_alphabet_change: true
        segment_alphabet: ["Han", "Kanbun", "Katakana", "Hiragana"]
        bpe_model_path: data/enkovocab/pyonmt_bpe_enko_32k_min2.model
        segment_numbers: true
        preserve_segmented_tokens: true
        support_prior_joiners: true

    target_tokenization:
      type: OpenNMTTokenizer
      params:
        joiner_annotate: true
        mode: aggressive
        no_substitution: true
        segment_alphabet_change: true
        segment_alphabet: ["Han", "Kanbun", "Katakana", "Hiragana"]
        bpe_model_path: data/enkovocab/pyonmt_bpe_enko_32k_min2.model
        segment_numbers: true
        preserve_segmented_tokens: true
        support_prior_joiners: true
  """)
  return config


class InputterTest():
  def _build_inputter(self, config):
    raise NotImplementedError()

  def testMakeTrainingDataset(self):
    config = _get_config()
    inputter = self._build_inputter(config)
    dataset = inputter.make_training_dataset(
      config['data']['train_features_file'], #source_file
      config['data']['train_labels_file'],   #target_file
      batch_size=1,
      batch_type="examples",
      shuffle_buffer_size=int(1e3),
      length_bucket_width=1,
      maximum_features_length=None,
      maximum_labels_length=None,
      single_pass=True
    )
    print(dataset)
    for source, target in dataset:
      print(' '.join([t.decode('UTF-8') for t in source['tokens'][0].numpy()]))
      print(' '.join([t.decode('UTF-8') for t in target['tokens'][0].numpy()]))
      print()


  def testMakeEvaluationDataset(self):
    config = _get_config()
    inputter = self._build_inputter(config)
    dataset = inputter.make_evaluation_dataset(
      config['data']['eval_features_file'], #source_file
      config['data']['eval_labels_file'],   #target_file
      batch_size=1,
      batch_type="examples",
      length_bucket_width=1,
    )
    print(dataset)
    for source, target in dataset:
      print(' '.join([t.decode('UTF-8') for t in source['tokens'][0].numpy()]))
      print(' '.join([t.decode('UTF-8') for t in target['tokens'][0].numpy()]))
      print()


class ExampleInputterTest(tf.test.TestCase, InputterTest):
  def _build_inputter(self, config):
    examplesInputter = onmt.inputters.ExampleInputter(
      onmt.inputters.WordEmbedder(embedding_size=1),
      onmt.inputters.WordEmbedder(embedding_size=1),
    )
    examplesInputter.initialize(config['data'])
    return examplesInputter


class ExampleInputterTest(tf.test.TestCase, InputterTest):
  def _build_inputter(self, config):
    spanCorruptionInputter = SpanCorruptionInputter(
      onmt.inputters.WordEmbedder(embedding_size=1),
      onmt.inputters.WordEmbedder(embedding_size=1),
      num_threads=4,
      span_corruption_noise_density=0.15,
      span_corruption_mean_noise_span_length=3.0,
    )
    spanCorruptionInputter.initialize(config['data'])
    return spanCorruptionInputter


class translationPairSpanCorruptionInputterTest(tf.test.TestCase, InputterTest):
  def _build_inputter(self, config):
    inputter = TranslationPairSpanCorruptionInputter(
      onmt.inputters.WordEmbedder(embedding_size=1),
      onmt.inputters.WordEmbedder(embedding_size=1),
      num_threads=4,
      span_corruption_noise_density=0.50,
      span_corruption_mean_noise_span_length=3.0,
    )
    inputter.initialize(config['data'])
    return inputter


class translationSpanCorruptionInputterTest(tf.test.TestCase, InputterTest):
  def _build_inputter(self, config):
    inputter = TranslationSpanCorruptionInputter(
      onmt.inputters.WordEmbedder(embedding_size=1),
      onmt.inputters.WordEmbedder(embedding_size=1),
      num_threads = 4,
      span_corruption_noise_density=0.50,
      span_corruption_mean_noise_span_length=3.0,
    )
    inputter.initialize(config['data'])
    return inputter


class ExamplesMTPromptTest(tf.test.TestCase, InputterTest):
  def _build_inputter(self, config):
    examplesMTPromptInputter = ExamplesMTPromptInputter(
      onmt.inputters.WordEmbedder(embedding_size=1),
      onmt.inputters.WordEmbedder(embedding_size=1),
      num_threads = 4,
    )
    examplesMTPromptInputter.initialize(config['data'])
    return examplesMTPromptInputter


class sc_tpsc_tsc_mt_InputterTest(tf.test.TestCase, InputterTest):
  def _build_inputter(self, config):
    scTpscTscMtInputter = ScTpscTscMtInputter(
      onmt.inputters.WordEmbedder(embedding_size=1),
      onmt.inputters.WordEmbedder(embedding_size=1),
      num_threads=4,
      span_corruption_noise_density=0.15,
      span_corruption_mean_noise_span_length=3.0,
      translation_span_corruption_noise_density=0.50,
      translation_span_corruption_mean_noise_span_length=3.0,
    )
    scTpscTscMtInputter.initialize(config['data'])
    return scTpscTscMtInputter


if __name__ == "__main__":
    tf.test.main()
