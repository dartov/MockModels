from teradataml import copy_to_sql, DataFrame
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

import pandas as pd


def score(context: ModelContext, **kwargs):

    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    features_tdf = DataFrame.from_query(context.dataset_info.sql)

    print("Scoring")

    predictions_df = features_tdf.assign(**{target_name: 1})

    print("Finished Scoring")

    # calculate stats


    record_scoring_stats(features_df=features_tdf, predicted_df=predictions_df, context=context)
