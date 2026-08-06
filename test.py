from ml_pipeline.feature_store.hopsworks_client import get_feature_store

feature_store = get_feature_store()

print("\n🎉 Successfully connected to Hopsworks!")
print(feature_store)