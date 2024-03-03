from opennmt.utils import misc

def _get_dataset_transforms(
    inputter,
    num_threads=None,
    training=None,
    prepare_batch_size=128,
):
    transform_fns = []

    if inputter.has_prepare_step():
        prepare_fn = lambda *arg: inputter.prepare_elements(
            misc.item_or_tuple(arg), training=training
        )
        transform_fns.extend(
            [
                lambda dataset: dataset.batch(prepare_batch_size),
                lambda dataset: dataset.map(prepare_fn, num_parallel_calls=num_threads),
                lambda dataset: dataset.unbatch(),
            ]
        )

    map_fn = lambda *arg: inputter.make_features(
        element=misc.item_or_tuple(arg), training=training
    )
    transform_fns.append(
        lambda dataset: dataset.map(map_fn, num_parallel_calls=num_threads)
    )
    return transform_fns
