import pytest
import os

from data_preprocessing.data_preparation import PrepareFiles


def test_class_definition():
    with pytest.raises(ValueError):
        data = PrepareFiles(dataset_dir="")
    dataset = PrepareFiles(data_available=True)
    assert os.path.isdir(dataset.dataset_dir) is True


# def test_load_images():
