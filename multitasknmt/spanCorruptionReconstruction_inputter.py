import tensorflow as tf
from opennmt.data import dataset as dataset_util
from opennmt.inputters import ExampleInputter
from opennmt.utils import misc

from multitasknmt.generateSpanCorruptionReconstruction import generateSpanCorruptionReconstruction
from multitasknmt.Util import _get_dataset_transforms

class SpanCorruptionReconstructionInputter(ExampleInputter):

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

        self.spanCorruptionReconstructionGenerator = generateSpanCorruptionReconstruction(
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

        # for span corruption reconstruction, we assume we only have the target file
        # The returned dataset is a parallel dataset
        tgtDatasets = self.inputters[1].make_dataset(data_file[1], training=training)
        if not isinstance(tgtDatasets, list): tgtDatasets = [tgtDatasets]
        parallel_datasets = []
        for d in tgtDatasets:
            parallel_datasets.append(self.spanCorruptionReconstructionGenerator.t5_denoise(d,
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


    def get_dataset_size(self, data_file):
        # size = sum(self.inputters[0].get_dataset_size(data_file[0]))
        size = self.inputters[1].get_dataset_size(data_file[1])
        if size is not None:
            for annotation, path in self.annotation_files.items():
                annotation_size = tf.nest.map_structure(misc.count_lines, path)
                if size != annotation_size:
                    raise RuntimeError(
                        "Annotation dataset '%s' does not have the same size as "
                        "the examples dataset" % annotation
                    )
        return size


    def make_evaluation_dataset(
        self,
        features_file,
        labels_file,
        batch_size,
        batch_type="examples",
        length_bucket_width=None,
        num_threads=1,
        prefetch_buffer_size=None,
    ):
        data_files = [features_file, labels_file]
        length_fn = [
            self.features_inputter.get_length,
            self.labels_inputter.get_length,
        ]

        transform_fns = _get_dataset_transforms(
            self, num_threads=num_threads, training=False
        )
        dataset = self.make_dataset(data_files, training=False)
        dataset = dataset.apply(
            dataset_util.inference_pipeline(
                batch_size,
                batch_type=batch_type,
                transform_fns=transform_fns,
                length_bucket_width=length_bucket_width,
                length_fn=length_fn,
                num_threads=num_threads,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )
        return dataset

    def make_training_dataset(
        self,
        features_file,
        labels_file,
        batch_size,
        batch_type="examples",
        batch_multiplier=1,
        batch_size_multiple=1,
        shuffle_buffer_size=None,
        length_bucket_width=None,
        pad_to_bucket_boundary=False,
        maximum_features_length=None,
        maximum_labels_length=None,
        single_pass=False,
        num_shards=1,
        shard_index=0,
        num_threads=4,
        prefetch_buffer_size=None,
        cardinality_multiple=1,
        weights=None,
        batch_autotune_mode=False,
        # skip_prob = 0.95,
    ):
        data_files = [features_file, labels_file]
        maximum_length = [maximum_features_length, maximum_labels_length]
        features_length_fn = self.features_inputter.get_length
        labels_length_fn = self.labels_inputter.get_length

        dataset = self.make_dataset(data_files, training=True)

        filter_fn = lambda *arg: (
            self.keep_for_training(
                misc.item_or_tuple(arg), maximum_length=maximum_length
            )
        )

        transform_fns = _get_dataset_transforms(
            self, num_threads=num_threads, training=True
        )
        transform_fns.append(lambda dataset: dataset.filter(filter_fn))

        # # apply random skip
        # tf.get_logger().info(f'Applying training data skip with probability {skip_prob}')
        # filter_fn = lambda *arg: (
        #     # keep if rand (0~1) is greater than skip_prob
        #     # for skip_prob 0.95, only 1/20 of instances will have True and be kept
        #     tf.math.greater_equal(tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32), skip_prob)
        # )
        # transform_fns.insert(0, lambda dataset: dataset.filter(filter_fn))

        if batch_autotune_mode:
            if isinstance(dataset, list):  # Ignore weighted dataset.
                dataset = dataset[0]

            # We repeat the dataset now to ensure full batches are always returned.
            dataset = dataset.repeat()
            for transform_fn in transform_fns:
                dataset = dataset.apply(transform_fn)

            # length_fn returns the maximum length instead of the actual example length so
            # that batches are built as if each example has the maximum length.
            if labels_file is not None:
                constant_length_fn = [
                    lambda x: maximum_features_length,
                    lambda x: maximum_labels_length,
                ]
            else:
                constant_length_fn = lambda x: maximum_features_length

            # The length dimension is set to the maximum length in the padded shapes.
            padded_shapes = self.get_padded_shapes(
                dataset.element_spec, maximum_length=maximum_length
            )

            # Dynamically pad each sequence to the maximum length.
            def _pad_to_shape(tensor, padded_shape):
                if tensor.shape.rank == 0:
                    return tensor
                tensor_shape = misc.shape_list(tensor)
                paddings = [
                    [0, padded_dim - tensor_dim]
                    if tf.is_tensor(tensor_dim) and padded_dim is not None
                    else [0, 0]
                    for tensor_dim, padded_dim in zip(tensor_shape, padded_shape)
                ]
                return tf.pad(tensor, paddings)

            dataset = dataset.map(
                lambda *arg: tf.nest.map_structure(
                    _pad_to_shape, misc.item_or_tuple(arg), padded_shapes
                )
            )
            dataset = dataset.apply(
                dataset_util.batch_sequence_dataset(
                    batch_size,
                    batch_type=batch_type,
                    batch_multiplier=batch_multiplier,
                    length_bucket_width=1,
                    length_fn=constant_length_fn,
                )
            )
            return dataset

        if weights is not None:
            dataset = (dataset, weights)
        dataset = dataset_util.training_pipeline(
            batch_size,
            batch_type=batch_type,
            batch_multiplier=batch_multiplier,
            batch_size_multiple=batch_size_multiple,
            transform_fns=transform_fns,
            length_bucket_width=length_bucket_width,
            pad_to_bucket_boundary=pad_to_bucket_boundary,
            features_length_fn=features_length_fn,
            labels_length_fn=labels_length_fn,
            maximum_features_length=maximum_features_length,
            maximum_labels_length=maximum_labels_length,
            single_pass=single_pass,
            num_shards=num_shards,
            shard_index=shard_index,
            num_threads=num_threads,
            dataset_size=self.get_dataset_size(data_files),
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size,
            cardinality_multiple=cardinality_multiple,
        )(dataset)
        return dataset
