import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


def simple_random_sampling(
    cur_entity: Tuple[int, int],
    num_neg_sample: int,
    true_triplet: Dict,
    num_entity: int = None,
    go_terms: List = None,
    **kwargs
) -> List:

    is_go = False if go_terms is None else True
    
    negative_sample_list = []
    negative_sample_size = 0

    while negative_sample_size < num_neg_sample:
        if not is_go:
            negative_sample = np.random.randint(num_entity, size=num_neg_sample)
        else:
            if len(go_terms) < num_neg_sample:
                negative_sample = np.random.choice(go_terms, size=num_neg_sample, replace=True)
            else:
                negative_sample = np.random.choice(go_terms, size=num_neg_sample, replace=False)

        mask = np.in1d(
            negative_sample,
            true_triplet[cur_entity],
            assume_unique=True,
            invert=True
        )
        negative_sample = negative_sample[mask]
        negative_sample_list.append(negative_sample)
        negative_sample_size += negative_sample.size
    
    negative_sample = np.concatenate(negative_sample_list)[:num_neg_sample]
    return negative_sample


negative_sampling_strategy = {
    'simple_random': simple_random_sampling,
}