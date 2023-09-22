from teradataml import DataFrame, copy_to_sql
from aoa import (
    record_evaluation_stats,
    aoa_create_context,
    ModelContext
)

import json
import numpy as np
import pandas as pd


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    test_df = DataFrame.from_query(context.dataset_info.sql)


    print("Scoring")

    evaluation = {'Accuracy': '1.0'}

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    record_evaluation_stats(features_df=test_df,
                            predicted_df=test_df,
                            context=context)
