from teradataml import DataFrame
from aoa import (
    record_training_stats,
    aoa_create_context,
    ModelContext
)

def train(context: ModelContext, **kwargs):
    aoa_create_context()

    target_name = context.dataset_info.target_names[0]

    train_df = DataFrame.from_query(context.dataset_info.sql)

    print("Starting dummy training...")

    continuous_features = context.hyperparams["continuous_features"].split(",")
    print("Continuous features are: ", continuous_features)
    categorical_features = context.hyperparams["categorical_features"].split(",")
    print("Categorical features are: ", categorical_features)
    print("Targets are:", context.dataset_info.target_names)

    print("Finished dummy training")


    record_training_stats(train_df,
                          features=continuous_features + categorical_features,
                          targets=[target_name],
                          categorical=categorical_features + [target_name],
                          context=context)
