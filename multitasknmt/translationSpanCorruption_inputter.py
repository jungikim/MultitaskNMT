import tensorflow as tf
from opennmt.inputters import ExampleInputter

from multitasknmt.generateTranslationSpanCorruption import generateTranslationSpanCorruption


class TranslationSpanCorruptionInputter(ExampleInputter):

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
        self.translationSpanCorruptionGenerator = generateTranslationSpanCorruption(
                                        noise_density = span_corruption_noise_density,
                                        mean_noise_span_length = span_corruption_mean_noise_span_length)


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
        if len(data_file[0]) != len(data_file[1]):
            raise ValueError(
                "All parallel inputs must have the same number of data files, "
                "saw %d files for input 0 but got %d files for input 1"
                % (len(data_file[0]), len(data_file[1]))
            )

        # translation span corruption
        srcDataset = self.inputters[0].make_dataset(data_file[0], training=training)
        tgtDataset = self.inputters[1].make_dataset(data_file[1], training=training)
        if not isinstance(srcDataset, list): srcDataset = [srcDataset]
        if not isinstance(tgtDataset, list): tgtDataset = [tgtDataset]
        datasets = [srcDataset, tgtDataset]

        dataset = [
            tf.data.Dataset.zip(tuple(parallel_dataset))
            for parallel_dataset in zip(*datasets)
        ]
        parallel_datasets = []
        for d in dataset:
            parallel_datasets.append(self.translationSpanCorruptionGenerator.t5_denoise(d,
                                                                                num_threads=self.num_threads,
                                                                                training=training))

        if len(parallel_datasets) == 1:
            return parallel_datasets[0]
        if not training:
            raise ValueError("Only training data can be configured to multiple files")
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
