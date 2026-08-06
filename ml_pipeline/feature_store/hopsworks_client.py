import os
import tempfile

temp_dir = tempfile.gettempdir()

os.environ["TMP"] = temp_dir
os.environ["TEMP"] = temp_dir
os.environ["TMPDIR"] = temp_dir
import hopsworks
from pathlib import Path

from .config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT,
)


def get_feature_store():
    """
    Connect to Hopsworks and return the Feature Store.
    """

    CERT_FOLDER = Path(__file__).resolve().parents[2] / "certs"

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        cert_folder=str(CERT_FOLDER),
    )

    feature_store = project.get_feature_store()

    print("✅ Connected to Hopsworks!")

    return feature_store