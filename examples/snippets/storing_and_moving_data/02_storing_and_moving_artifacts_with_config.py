# We can make our workflow entirely modular, by abstracting how we define the functions we are using, and resolving
# them via an input config file

# create dummy data
import pandas as pd
import os
df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

# resolve storage and copy functions
config = {
    "store_artifact_locally": "store_pandas_pickle",
    "copy_model": "copy_from_disk_to_disk"
}
from mercury_ml.common import ArtifactCopiers, LocalArtifactStorers
store_artifact_locally = getattr(LocalArtifactStorers, config["store_artifact_locally"])
copy_from_local_to_remote = getattr(ArtifactCopiers, config["copy_model"])

# use task api to perform storage and copying in one step
path = store_artifact_locally(data=df, directory="./stored_data/local", filename="my_dummy_data")
copy_from_local_to_remote(source_dir="./stored_data/local",
                          target_dir="./stored_data/remote",
                          filename=os.path.basename(path))


