from setuptools import setup, find_packages
setup(name="mercury-ml",
      version="0.1.4",
      description="A library for managing Machine Learning workflows",
      url="https://github.com/mercury-ml-team/mercury-ml",
      author="Karl Schriek",
      author_email="kschriek@gmail.com",
      license="MIT",
      packages=find_packages(
          exclude=["*.tests", "*.tests.*", "tests.*", "tests",
                   "*.examples", "*.examples.*", "examples.*", "examples"]),
      include_package_data=True,
      install_requires=["numpy", "pandas", "sklearn", "jsonref"],
      extras_require={
            "keras": ["tensorflow", "keras", "pillow"],
            "keras-gpu": ["tensorflow-gpu", "keras", "pillow"],
            "h2o": ["h2o"],
            "h2o-sparkling": ["h2o", "pyspark", "h2o-pysparkling"],
            "s3": ["boto3"],
            "gcs": ["google-cloud-storage"],
            "mongo": ["pymongo"]
            },
      python_requires=">=3.5",
      zip_safe=False)



