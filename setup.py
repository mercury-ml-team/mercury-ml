from setuptools import setup, find_packages
setup(name="mercury-ml",
      version="0.0.2",
      description="A library for managing Machine Learning workflows",
      url="https://github.com/mercury-ml-team/mercury-ml",
      author="Karl Schriek",
      author_email="kschriek@gmail.com",
      license="MIT",
      packages=find_packages(
          exclude=["*.tests", "*.tests.*", "tests.*", "tests",
                   "*.examples", "*.examples.*", "examples.*", "examples"]),
      include_package_data=True,
      install_requires=["jsonref"],
      extras_require={
            "keras": ["tensorflow", "keras", "pillow"],
            },
      python_requires=">=3.6",
      zip_safe=False)



