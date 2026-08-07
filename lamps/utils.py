import os
import importlib


def slugify(string):
    """
    Convert a string to a slug format (lowercase, spaces replaced with underscores).
    """
    return string.lower().replace("/", "__")


def get_checkpoint_save_path(experiment, dataset, mode):
    base_save_path = os.path.join("checkpoints", experiment, mode, dataset)

    i = 1
    save_path = f"{base_save_path}_{i}"

    while os.path.exists(save_path):
        i += 1
        save_path = f"{base_save_path}_{i}"

    return save_path


def load_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
