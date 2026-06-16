import os
import importlib.util
import numpy as np

# Load stage1_preproc by file path so the test works even if the package
# isn't installed or doesn't have an __init__.py
HERE = os.path.dirname(__file__)
MODULE_PATH = os.path.abspath(os.path.join(HERE, "..", "stage1_preproc.py"))
spec = importlib.util.spec_from_file_location("stage1_preproc", MODULE_PATH)
stage1_preproc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage1_preproc)


def _print(title, arr):
    if arr is None:
        print(f"{title}: None")
    else:
        print(f"{title}: shape={arr.shape}, dtype={arr.dtype}")


def main():
    # create a dummy color image (BGR) 300x400
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    blurred = stage1_preproc.gaussian_blur(img, ksize=(6, 6))
    resized = stage1_preproc.resize_to_size(img, size=(256, 256))

    _print("orig", img)
    _print("blurred", blurred)
    _print("resized", resized)


if __name__ == "__main__":
    main()

