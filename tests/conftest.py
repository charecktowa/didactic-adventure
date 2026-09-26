from importlib.util import find_spec

# These tests need ZenML, which only the optional `mlops` dependency group installs.
collect_ignore = [] if find_spec("zenml") else ["test_training_pipeline.py"]
