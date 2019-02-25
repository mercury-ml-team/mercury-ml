# History

## 0.1.1

- Keras Model fitting methods at `mercury_ml.keras.providers.model_fitting` now allow for keyboard interrupts (either last, or last_best model will be returned)
- `mercury_ml.common.utils.recursively_update_config` will now correctly iterate through both dictionaries and lists
- In `mercury_ml.common.providers.artifact_copying.from_disk.copy_from_disk_to_s3` it is now possible to specify whether or not an existing S3 file should be overwritten 
- A `copy_from_s3_to_s3` method has been added `mercury_ml.common.providers.artifact_copying.from_s3`

## 0.1.0

Initial (pre-release) version. Most currently functionality has been tested and works. Lot's of functionality still to come.
Only Keras and H2O currently supported, but the aim is to eventually support all major (and some minor) machine learning
frameworks.