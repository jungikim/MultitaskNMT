import tensorflow as tf
from opennmt.utils import misc

from multitasknmt.sc_mt_inputter import ScMtInputter
from multitasknmt.generateSpanCorruptionReconstruction import generateSpanCorruptionReconstruction

class ScScrMtInputter(ScMtInputter):
    def __init__(
        self,
        features_inputter,
        labels_inputter,
        share_parameters=False,
        accepted_annotations=None,
        num_threads = 4,
        span_corruption_noise_density=0.15,
        span_corruption_mean_noise_span_length=3.0,
    ):
        super().__init__(
            features_inputter,
            labels_inputter,
            share_parameters=share_parameters,
            accepted_annotations=accepted_annotations,
            num_threads = num_threads,
            span_corruption_noise_density = span_corruption_noise_density,
            span_corruption_mean_noise_span_length = span_corruption_mean_noise_span_length,
        )
        self.spanCorruptionReconstructionGenerator = generateSpanCorruptionReconstruction(
                                        noise_density = span_corruption_noise_density,
                                        mean_noise_span_length = span_corruption_mean_noise_span_length)


    def _make_dataset(self, data_file, training=None):
        if not isinstance(data_file, list):
            data_file = [data_file]

        if not training:
            try:
                data_file = tf.nest.pack_sequence_as(
                    self._structure(), tf.nest.flatten(data_file)
                )
            except ValueError:
                data_file = []  # This will raise the error below.

        if not isinstance(data_file[0], list):
            data_file[0] = [data_file[0]]
        if len(data_file) == 1:
            data_file.append([None for _ in range(len(data_file[0]))])
        if not isinstance(data_file[1], list):
            data_file[1] = [data_file[1]]

        if len(self.inputters) != 2 or len(data_file) != 2:
            raise ValueError(
                "TranslationPairSpanCorruption takes exactly two inputters and two sets of data files"
                "saw %d inputters but got %d sets of data files"
                % (len(self.inputters), len(data_file))
            )
        if len(data_file[0]) != len(data_file[1]):
            raise ValueError(
                "All parallel inputs must have the same number of data files, "
                "saw %d files for input 0 but got %d files for input 1"
                % (len(data_file[0]), len(data_file[1]))
            )

        monolingual_data_file=[[],[]]
        bilingual_data_file=[[],[]]
        monolingual_data_file[0], monolingual_data_file[1], \
        bilingual_data_file[0], bilingual_data_file[1] = \
                                    ScMtInputter._separate_into_mono_bi(data_file)

        monolingual_data_file[0] = [ d for d in monolingual_data_file[0] if d is not None]
        monolingual_data_file[1] = [ d for d in monolingual_data_file[1] if d is not None]

        bilingual_data_langpairs = ScMtInputter._separate_bi_into_langpair(bilingual_data_file)

        tf.get_logger().info('len(monolingual_data_file[0]): %s' % (len(monolingual_data_file[0])))
        tf.get_logger().info('len(monolingual_data_file[1]): %s' % (len(monolingual_data_file[1])))
        tf.get_logger().info('len(bilingual_data_file[0]): %s' % (len(bilingual_data_file[0])))
        tf.get_logger().info('len(bilingual_data_file[1]): %s' % (len(bilingual_data_file[1])))
        tf.get_logger().info('len(bilingual_data_langpairs): %s' % (len(bilingual_data_langpairs)))

        sc_parallel_datasets=[]
        scr_parallel_datasets=[]
        if monolingual_data_file and len(monolingual_data_file) == 2 and len(monolingual_data_file[0]) > 0:
            sc_srcDatasets = self.inputters[0].make_dataset(monolingual_data_file[0], training=training)
            if not isinstance(sc_srcDatasets, list): sc_srcDatasets = [sc_srcDatasets]
            for sc_srcDataset in sc_srcDatasets:
                sc_parallel_datasets.append(self.spanCorruptionGenerator.t5_denoise(sc_srcDataset,
                                                                            num_threads=self.num_threads,
                                                                            training=training))
        if monolingual_data_file and len(monolingual_data_file) == 2 and len(monolingual_data_file[1]) > 0:
            scr_tgtDatasets = self.inputters[1].make_dataset(monolingual_data_file[1], training=training)
            if not isinstance(scr_tgtDatasets, list): scr_tgtDatasets = [scr_tgtDatasets]
            for scr_tgtDataset in scr_tgtDatasets:
                scr_parallel_datasets.append(self.spanCorruptionReconstructionGenerator.t5_denoise(scr_tgtDataset,
                                                                                           num_threads=self.num_threads,
                                                                                           training=training))

        mt_parallel_datasets = []
        if len(bilingual_data_langpairs) > 0:
            for (srclang, tgtlang) in bilingual_data_langpairs.keys():
                srcFs, tgtFs = bilingual_data_langpairs[(srclang, tgtlang)]
                mt_srcDataset = self.inputters[0].make_dataset(srcFs, training=training)
                mt_tgtDataset = self.inputters[1].make_dataset(tgtFs, training=training)
                if not isinstance(mt_srcDataset, list): mt_srcDataset = [mt_srcDataset]
                if not isinstance(mt_tgtDataset, list): mt_tgtDataset = [mt_tgtDataset]
                mt_datasets = [mt_srcDataset, mt_tgtDataset]
                mt_dataset = [
                    tf.data.Dataset.zip(tuple(mt_parallel_dataset))
                    for mt_parallel_dataset in zip(*mt_datasets)
                ]
                for mt_d in mt_dataset:
                    mt_parallel_datasets.append(self.MTPromptGenerator.generateMTPrompt(mt_d,
                                                                            srcLang=srclang, tgtLang=tgtlang,
                                                                            num_threads=self.num_threads,
                                                                            training=training))

        tf.get_logger().info('len(sc_parallel_datasets): %s' % (len(sc_parallel_datasets)))
        tf.get_logger().info('len(scr_parallel_datasets): %s' % (len(scr_parallel_datasets)))
        tf.get_logger().info('len(mt_parallel_datasets): %s' % (len(mt_parallel_datasets)))

        parallel_datasets = []
        parallel_datasets.extend(sc_parallel_datasets)
        parallel_datasets.extend(scr_parallel_datasets)
        parallel_datasets.extend(mt_parallel_datasets)
        tf.get_logger().info('len(parallel_datasets): %s' % (len(parallel_datasets)))

        if len(parallel_datasets) == 1:
            return parallel_datasets[0]
        if not training:
            # raise ValueError("Only training data can be configured to multiple files")
            return mt_parallel_datasets[0]
        return parallel_datasets


    def get_dataset_size(self, data_file):
        monolingual_data_file=[[],[]]
        bilingual_data_file=[[],[]]
        monolingual_data_file[0], monolingual_data_file[1], \
        bilingual_data_file[0], bilingual_data_file[1] = \
                                    ScMtInputter._separate_into_mono_bi(data_file)

        monolingual_data_file[0] = [ d for d in monolingual_data_file[0] if d is not None]
        monolingual_data_file[1] = [ d for d in monolingual_data_file[1] if d is not None]

        mono_src_size = sum(self.inputters[0].get_dataset_size(monolingual_data_file[0]))
        mono_tgt_size = sum(self.inputters[1].get_dataset_size(monolingual_data_file[1]))

        bi_size = None
        for inputter, data in zip(self.inputters, bilingual_data_file):
            _size = sum(inputter.get_dataset_size(data))
            if _size is not None:
                if bi_size is None:
                    bi_size = _size
                elif _size != bi_size:
                    raise RuntimeError("Parallel datasets do not have the same size %s, %s" % (bi_size, _size))

        # mono: sc, scr, bi: mt
        size = mono_src_size + mono_tgt_size + bi_size

        if size is not None:
            for annotation, path in self.annotation_files.items():
                annotation_size = tf.nest.map_structure(misc.count_lines, path)
                if size != annotation_size:
                    raise RuntimeError(
                        "Annotation dataset '%s' does not have the same size as "
                        "the examples dataset" % annotation
                    )
        return size
