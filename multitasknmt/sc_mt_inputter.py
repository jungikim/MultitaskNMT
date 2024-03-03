import tensorflow as tf
from opennmt.utils import misc

from multitasknmt.spanCorruption_inputter import SpanCorruptionInputter
from multitasknmt.generateSpanCorruption import generateSpanCorruption
from multitasknmt.generateMTPrompt import generateMTPrompt

class ScMtInputter(SpanCorruptionInputter):
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
        )
        self.num_threads = num_threads

        self.spanCorruptionGenerator = generateSpanCorruption(
                                        noise_density = span_corruption_noise_density,
                                        mean_noise_span_length = span_corruption_mean_noise_span_length)
        self.MTPromptGenerator = generateMTPrompt()


    @staticmethod
    def _separate_into_mono_bi(data_file):
        srcDataFiles, tgtDataFiles = data_file[0], data_file[1]
        mono_src, mono_tgt = [], []
        bi_src, bi_tgt = [], []

        if not tgtDataFiles:
            for src in srcDataFiles:
                mono_src.append(src)
                mono_tgt.append(None)
            return mono_src, mono_tgt, bi_src, bi_tgt

        if not srcDataFiles:
            for tgt in tgtDataFiles:
                mono_src.append(None)
                mono_tgt.append(tgt)
            return mono_src, mono_tgt, bi_src, bi_tgt

        for src,tgt in zip(srcDataFiles, tgtDataFiles):
            if (tgt == None or len(tgt) == 0) or \
               (src == None or len(src) == 0):
                mono_src.append(src)
                mono_tgt.append(tgt)
            else:
                bi_src.append(src)
                bi_tgt.append(tgt)
        return mono_src, mono_tgt, bi_src, bi_tgt


    @staticmethod
    def _separate_bi_into_langpair(bidata_file):
        srcFiles, tgtFiles = bidata_file[0], bidata_file[1]
        langpairMap = {}
        for srcF, tgtF in zip(srcFiles, tgtFiles):
            srcLang = generateMTPrompt.parseLang(srcF)
            tgtLang = generateMTPrompt.parseLang(tgtF)
            if (srcLang,tgtLang) not in langpairMap:
                langpairMap[(srcLang,tgtLang)] = ([],[])
            langpairMap[(srcLang,tgtLang)][0].append(srcF)
            langpairMap[(srcLang,tgtLang)][1].append(tgtF)

        return langpairMap


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

        # for span corruption, we assume we only have the source file
        # The returned dataset is a parallel dataset
        sc_parallel_datasets=[]
        if monolingual_data_file and len(monolingual_data_file) == 2 and len(monolingual_data_file[0]) > 0:
            sc_srcDatasets = self.inputters[0].make_dataset(monolingual_data_file[0], training=training)
            if not isinstance(sc_srcDatasets, list): sc_srcDatasets = [sc_srcDatasets]
            for sc_srcDataset in sc_srcDatasets:
                sc_parallel_datasets.append(self.spanCorruptionGenerator.t5_denoise(sc_srcDataset,
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
        tf.get_logger().info('len(mt_parallel_datasets): %s' % (len(mt_parallel_datasets)))

        parallel_datasets = []
        parallel_datasets.extend(sc_parallel_datasets)
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

        mono_size = sum(self.inputters[0].get_dataset_size(monolingual_data_file[0]))

        bi_size = None
        for inputter, data in zip(self.inputters, bilingual_data_file):
            _size = sum(inputter.get_dataset_size(data))
            if _size is not None:
                if bi_size is None:
                    bi_size = _size
                elif _size != bi_size:
                    raise RuntimeError("Parallel datasets do not have the same size %s, %s" % (bi_size, _size))

        size = mono_size + 1 * (bi_size)

        if size is not None:
            for annotation, path in self.annotation_files.items():
                annotation_size = tf.nest.map_structure(misc.count_lines, path)
                if size != annotation_size:
                    raise RuntimeError(
                        "Annotation dataset '%s' does not have the same size as "
                        "the examples dataset" % annotation
                    )
        return size
