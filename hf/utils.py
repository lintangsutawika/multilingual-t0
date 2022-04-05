import re
import functools

from typing import Dict, List, Optional, Tuple


def _interleave_map_style_datasets(
    datasets: List["Dataset"],
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[Any] = None,
    split: Optional[Any] = None,
    stop: Optional[str] = 'first_exhausted',
    **kwargs,
) -> "Dataset":
    """
    Interleave several map-style datasets (sources) into a single map-style dataset.
    The new dataset is constructed by alternating between the sources to get the examples.
    If `probabilities = None` (default) the new dataset is constructed by cycling between each source to get the examples.
    If `probabilities` is not `None, the new dataset is constructed by getting examples from a random source at a time according to the provided probabilities.

    Args:
        datasets (:obj:`List[Dataset]`): list of datasets to interleave
        probabilities (:obj:`List[float]`, optional, default None): If specified, the new dataset is constructued by sampling
            examples from one source at a time according to these probabilities.
        seed (:obj:`int`, optional, default None): The random seed used to choose a source for each example.
        stop (:obj:`str`, optional, default 'first_exhausted'): If `stop = 'first_exhausted'`, the sampling ends when one of the source datasets runs out of examples.
        **kwargs: Keyword arguments to be passed to :meth:`datasets.Datasets.select` when selecting the indices used to interleave the datasets.

    Output:
        :class:`datasets.Dataset`
    """
    from datasets import concatenate_datasets

    # To interleave the datasets, we concatenate them and then we re-order the indices
    concatenated_datasets = concatenate_datasets(datasets, info=info, split=split)

    # Let's now build the indices to pass to .select()
    # Here we create the length that will be sampled from each dataset based on its probability
    lengths = [len(dset) for dset in datasets]
    offsets = np.cumsum([0] + lengths[:-1])
    if probabilities is None:
        # Example: If lengths of the datasets are [3, 4, 5]
        # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 6, 9]
        # Note that we only have 3 examples per dataset since the first dataset ran out of examples
        indices = (offsets.reshape(1, -1) + np.arange(min(lengths)).reshape(-1, 1)).flatten().tolist()
    else:

        def iter_random_indices():
            """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
            rng = np.random.default_rng(seed)
            while True:
                yield from (int(i) for i in rng.choice(len(datasets), size=1000, p=probabilities))

        current_index = [0] * len(datasets)
        runout = []
        indices = []
        if stop== "first_exhausted":
            for source_idx in iter_random_indices():
                # we ran out of examples, let's stop
                if current_index[source_idx] >= lengths[source_idx]:
                    break
                # let's add the example at the current index of the `source_idx`-th dataset
                indices.append(current_index[source_idx] + offsets[source_idx])
                current_index[source_idx] += 1
        else:
            # stop == "all_exhausted"
            # This approach oversamples from runs out dataset
            for source_idx in iter_random_indices():
                # we ran out of examples from one of the datasets so we add the source_idx to the runout list
                # we keep doing it until we run out of examples from all the datasets
        
                if current_index[source_idx] >= lengths[source_idx]:
                    if source_idx not in runout:
                        runout.append(source_idx)
                    
                    current_index[source_idx]=0

                if len(runout)==len(probabilities):
                    break


                # let's add the example at the current index of the `source_idx`-th dataset
                indices.append(current_index[source_idx] + offsets[source_idx])
                current_index[source_idx] += 1
    return concatenated_datasets.select(indices, **kwargs)
