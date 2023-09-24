from teradataml import DataFrame
from aoa import (
    record_training_stats,
    aoa_create_context,
    get_feature_stats_summary,
    ModelContext
)

def train(context: ModelContext, **kwargs):
    aoa_create_context()

    target_name = context.dataset_info.target_names[0]

    feature_list = [x.lower() for x in context.hyperparams["features"].split(",")]
    feature_summary = get_feature_stats_summary(context.dataset_info.get_feature_metadata_fqtn())
    continuous_features = [f for f in feature_list if feature_summary[f]=='continuous']
    categorical_features = [f for f in feature_list if feature_summary[f]=='categorical']

    train_df = DataFrame.from_query(context.dataset_info.sql)

    print("Starting dummy training...")

    print("Continuous features are: ", continuous_features)
    print("Categorical features are: ", categorical_features)
    print("Targets are:", context.dataset_info.target_names)

    print("Finished dummy training")


    record_training_stats(train_df,
                          features=continuous_features + categorical_features,
                          targets=[target_name],
                          categorical=categorical_features + [target_name],
                          context=context)
