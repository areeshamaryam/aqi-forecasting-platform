# import pandas as pd

# df = pd.read_parquet("ml_pipeline/data/processed/features.parquet")

# print(df.head())

# print("\nColumns:")
# print(df.columns)

# print("\nData Types:")
# print(df.dtypes)
from ml_pipeline.feature_store.hopsworks_client import get_feature_store
from ml_pipeline.feature_store.config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

fs = get_feature_store()
fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
fg.delete()
print("Deleted.")