import tensorflow as tf
from opennmt.inputters import ExampleInputter
from opennmt.data import dataset as dataset_util
from opennmt.utils import misc
from multitasknmt.generateMTPrompt import generateMTPrompt
from multitasknmt.Util import _get_dataset_transforms

class ExamplesMTPromptInputter(ExampleInputter):

    def __init__(
        self,
        features_inputter,
        labels_inputter,
        share_parameters=False,
        accepted_annotations=None,
        num_threads = 4,
    ):
        super().__init__(
            features_inputter,
            labels_inputter,
            share_parameters=share_parameters,
            accepted_annotations=accepted_annotations,
        )
        self.num_threads = num_threads
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


        if len(self.inputters) != 2 or len(data_file) != 2:
            raise ValueError(
                "TranslationSpanCorruption takes exactly two inputters and two sets of data files"
                "saw %d inputters but got %d sets of data files"
                % (len(self.inputters), len(data_file))
            )
        # if len(data_file[0]) != len(data_file[1]):
        #     print(data_file[0], flush=True)
        #     print(data_file[1], flush=True)
        #     raise ValueError(
        #         "All parallel inputs must have the same number of data files, "
        #         "saw %d files for input 0 but got %d files for input 1"
        #         % (len(data_file[0]), len(data_file[1]))
        #     )

        # Add MT Prompt

        def _separate_bi_into_langpair(bidata_file):
            srcFiles, tgtFiles = bidata_file[0], bidata_file[1]
            if not isinstance(srcFiles, list): srcFiles = [srcFiles]
            if not isinstance(tgtFiles, list): tgtFiles = [tgtFiles]
            langpairMap = {}
            for srcF, tgtF in zip(srcFiles, tgtFiles):
                srcLang = generateMTPrompt.parseLang(srcF)
                tgtLang = generateMTPrompt.parseLang(tgtF)
                if (srcLang,tgtLang) not in langpairMap:
                    langpairMap[(srcLang,tgtLang)] = ([],[])
                langpairMap[(srcLang,tgtLang)][0].append(srcF)
                langpairMap[(srcLang,tgtLang)][1].append(tgtF)


            return langpairMap

        bilingual_data_langpairs = _separate_bi_into_langpair(data_file)


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
                mt_single_dataset = mt_dataset[0]
                for i in range(1,len(mt_dataset)):
                    mt_single_dataset = mt_single_dataset.concatenate(mt_dataset[i])
                mt_parallel_datasets.append(self.MTPromptGenerator.generateMTPrompt(mt_single_dataset,
                                                                            srcLang=srclang, tgtLang=tgtlang,
                                                                            num_threads=self.num_threads,
                                                                            training=training))

        # single_dataset = mt_parallel_datasets[0]
        # for i in range(1,len(mt_parallel_datasets)):
        #     single_dataset = single_dataset.concatenate(mt_parallel_datasets[i])

        # single_dataset = [single_dataset]

        if len(mt_parallel_datasets) == 1:
            return mt_parallel_datasets[0]
        if not training:
            raise ValueError("Only training data can be configured to multiple files")
        return mt_parallel_datasets


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


    def make_inference_dataset(
            self,
            features_file,
            batch_size,
            srcLang = None,
            tgtLang = None,
            batch_type="examples",
            length_bucket_width=None,
            num_threads=1,
            prefetch_buffer_size=None,
    ):
        transform_fns = _get_dataset_transforms(
            self.features_inputter, num_threads=num_threads, training=False
        )
        dataset = self.features_inputter.make_dataset(features_file, training=False)
        dataset = self.MTPromptGenerator.generateMTPrompt(dataset,
                                                          srcLang=srcLang,
                                                          tgtLang=tgtLang,
                                                          num_threads=self.num_threads,
                                                          training=False)
        dataset = dataset.apply(
            dataset_util.inference_pipeline(
                batch_size,
                batch_type=batch_type,
                transform_fns=transform_fns,
                length_bucket_width=length_bucket_width,
                length_fn=self.features_inputter.get_length,
                num_threads=num_threads,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )
        return dataset
