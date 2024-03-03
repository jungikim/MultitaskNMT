import tensorflow as tf
from opennmt.utils import misc

from multitasknmt.spanCorruption_inputter import SpanCorruptionInputter
from multitasknmt.sc_mt_inputter import ScMtInputter
from multitasknmt.generateTranslationSpanCorruption import generateTranslationSpanCorruption
from multitasknmt.generateTranslationPairSpanCorruption import generateTranslationPairSpanCorruption
from multitasknmt.generateMTPrompt import generateMTPrompt

class TpscTscMtInputter(SpanCorruptionInputter):
    def __init__(
        self,
        features_inputter,
        labels_inputter,
        share_parameters=False,
        accepted_annotations=None,
        num_threads = 4,
        translation_span_corruption_noise_density=0.50,
        translation_span_corruption_mean_noise_span_length=3.0,
    ):
        super().__init__(
            features_inputter,
            labels_inputter,
            share_parameters=share_parameters,
            accepted_annotations=accepted_annotations,
        )
        self.num_threads = num_threads

        self.translationSpanCorruptionGenerator = generateTranslationSpanCorruption(
                                        noise_density = translation_span_corruption_noise_density,
                                        mean_noise_span_length = translation_span_corruption_mean_noise_span_length)
        self.translationPairSpanCorruptionGenerator = generateTranslationPairSpanCorruption(
                                        noise_density = translation_span_corruption_noise_density,
                                        mean_noise_span_length = translation_span_corruption_mean_noise_span_length)
        self.MTPromptGenerator = generateMTPrompt()


    # modified from ParallelInputter.make_dataset()
    def _make_dataset(self, data_file, training=None):

        if not isinstance(data_file, list):
            data_file = [data_file]

        # For evaluation and inference, accept a flat list of data files for nested inputters.
        # This is needed when nesting can't easily be represented (e.g. on the command line).
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

        tsc_parallel_datasets = []
        tpsc_parallel_datasets = []
        # mt_parallel_datasets = []
        if bilingual_data_file and len(bilingual_data_file) == 2 and len(bilingual_data_file[0]) > 0:
            # translation span corruption
            tsc_srcDataset = self.inputters[0].make_dataset(bilingual_data_file[0], training=training)
            tsc_tgtDataset = self.inputters[1].make_dataset(bilingual_data_file[1], training=training)
            if not isinstance(tsc_srcDataset, list): tsc_srcDataset = [tsc_srcDataset]
            if not isinstance(tsc_tgtDataset, list): tsc_tgtDataset = [tsc_tgtDataset]
            tsc_datasets = [tsc_srcDataset, tsc_tgtDataset]

            tsc_dataset = [
                tf.data.Dataset.zip(tuple(tsc_parallel_dataset))
                for tsc_parallel_dataset in zip(*tsc_datasets)
            ]
            for tsc_d in tsc_dataset:
                tsc_parallel_datasets.append(self.translationSpanCorruptionGenerator.t5_denoise(tsc_d,
                                                                                        num_threads=self.num_threads,
                                                                                        training=training))


            # translation pair span corruption
            tpsc_srcDataset = self.inputters[0].make_dataset(bilingual_data_file[0], training=training)
            tpsc_tgtDataset = self.inputters[1].make_dataset(bilingual_data_file[1], training=training)
            if not isinstance(tpsc_srcDataset, list): tpsc_srcDataset = [tpsc_srcDataset]
            if not isinstance(tpsc_tgtDataset, list): tpsc_tgtDataset = [tpsc_tgtDataset]
            tpsc_datasets = [tpsc_srcDataset, tpsc_tgtDataset]
            tpsc_dataset = [
                tf.data.Dataset.zip(tuple(tpsc_parallel_dataset))
                for tpsc_parallel_dataset in zip(*tpsc_datasets)
            ]
            for tpsc_d in tpsc_dataset:
                tpsc_parallel_datasets.append(self.translationPairSpanCorruptionGenerator.t5_denoise(tpsc_d,
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


        tf.get_logger().info('len(tsc_parallel_datasets): %s' % (len(tsc_parallel_datasets)))
        tf.get_logger().info('len(tpsc_parallel_datasets): %s' % (len(tpsc_parallel_datasets)))
        tf.get_logger().info('len(mt_parallel_datasets): %s' % (len(mt_parallel_datasets)))

        parallel_datasets = []
        parallel_datasets.extend(tsc_parallel_datasets)
        parallel_datasets.extend(tpsc_parallel_datasets)
        parallel_datasets.extend(mt_parallel_datasets)
        tf.get_logger().info('len(parallel_datasets): %s' % (len(parallel_datasets)))

        # single_parallel_dataset = parallel_datasets[0]
        # for i in range(1,len(parallel_datasets)):
        #     single_parallel_dataset = single_parallel_dataset.concatenate(parallel_datasets[i])
        # tf.get_logger().info('single_parallel_dataset: %s' % (single_parallel_dataset))
        # parallel_datasets = [single_parallel_dataset]

        if len(parallel_datasets) == 1:
            return parallel_datasets[0]
        if not training:
            # raise ValueError("Only training data can be configured to multiple files")
            return mt_parallel_datasets[0]
        return parallel_datasets


    # modified from ExampleInputter.make_dataset()
    def make_dataset(self, data_file, training=None):
        dataset = self._make_dataset(data_file, training=training)
        if not training or not self.annotation_files:
            return dataset

        # Some annotations are configured and should be zipped to the training dataset.
        all_annotation_datasets = tf.nest.map_structure(
            tf.data.TextLineDataset, self.annotation_files
        )

        # Common case of a non-weighted dataset.
        if not isinstance(dataset, list):
            return tf.data.Dataset.zip({"examples": dataset, **all_annotation_datasets})

        # Otherwise, there should be as many annotations datasets as input datasets.
        datasets = dataset
        for name, annotation_datasets in all_annotation_datasets.items():
            num_annotation_datasets = (
                len(annotation_datasets) if isinstance(annotation_datasets, list) else 1
            )
            if num_annotation_datasets != len(datasets):
                raise ValueError(
                    "%d '%s' files were provided, but %d were expected to match the "
                    "number of data files"
                    % (num_annotation_datasets, name, len(datasets))
                )

        # Convert dict of lists to list of dicts.
        all_annotation_datasets = [
            dict(zip(all_annotation_datasets, t))
            for t in zip(*all_annotation_datasets.values())
        ]

        return [
            tf.data.Dataset.zip({"examples": dataset, **annotation_datasets})
            for dataset, annotation_datasets in zip(datasets, all_annotation_datasets)
        ]


    def get_dataset_size(self, data_file):
        monolingual_data_file=[[],[]]
        bilingual_data_file=[[],[]]
        monolingual_data_file[0], monolingual_data_file[1], \
        bilingual_data_file[0], bilingual_data_file[1] = \
                                    ScMtInputter._separate_into_mono_bi(data_file)

        monolingual_data_file[0] = [ d for d in monolingual_data_file[0] if d is not None]
        monolingual_data_file[1] = [ d for d in monolingual_data_file[1] if d is not None]

        bi_size = None
        for inputter, data in zip(self.inputters, bilingual_data_file):
            _size = sum(inputter.get_dataset_size(data))
            if _size is not None:
                if bi_size is None:
                    bi_size = _size
                elif _size != bi_size:
                    raise RuntimeError("Parallel datasets do not have the same size %s, %s" % (bi_size, _size))

        size = 3 * (bi_size)

        if size is not None:
            for annotation, path in self.annotation_files.items():
                annotation_size = tf.nest.map_structure(misc.count_lines, path)
                if size != annotation_size:
                    raise RuntimeError(
                        "Annotation dataset '%s' does not have the same size as "
                        "the examples dataset" % annotation
                    )
        return size
