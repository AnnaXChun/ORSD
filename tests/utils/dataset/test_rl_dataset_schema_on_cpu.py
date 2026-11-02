# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import datasets

from verl.utils.dataset.rl_dataset import normalize_top_level_string_features


def test_normalize_top_level_string_features_allows_concatenation():
    large_string_dataset = datasets.Dataset.from_dict(
        {"data_source": ["aime24"]},
        features=datasets.Features({"data_source": datasets.Value("large_string")}),
    )
    string_dataset = datasets.Dataset.from_dict(
        {"data_source": ["aime25"]},
        features=datasets.Features({"data_source": datasets.Value("string")}),
    )

    normalized = [
        normalize_top_level_string_features(large_string_dataset),
        normalize_top_level_string_features(string_dataset),
    ]
    combined = datasets.concatenate_datasets(normalized)

    assert combined.features["data_source"].dtype == "string"
    assert combined["data_source"] == ["aime24", "aime25"]
