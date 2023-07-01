import os
import shutil
import sys
from pathlib import Path

if __name__ == "__main__":
    filename = sys.argv[1]
    input_dir = sys.argv[2]
    unmatch_dir = Path(input_dir).parent / "un_match"
    match_dir = Path(input_dir).parent / "match"

    with open(filename, "rt") as f:
        rows = [row.strip() for row in f.readlines()]

    un_match = [name.split(os.sep)[-1] for name in rows]

    if not os.path.exists(str(unmatch_dir)):
        unmatch_dir.mkdir(exist_ok=True)
    if not match_dir.exists():
        match_dir.mkdir(exist_ok=True)

    for image in un_match:
        if os.path.exists(os.path.join(input_dir, image)):
            image_path = os.path.join(input_dir, image)
            shutil.copy(image_path, unmatch_dir.joinpath(image))
    print("Done")

    match = list(set(os.listdir(input_dir)) - set(un_match))
    for image in match:
        if os.path.exists(os.path.join(input_dir, image)):
            image_path = os.path.join(input_dir, image)
            shutil.copy(image_path, match_dir.joinpath(image))
    print("Done")

    shutil.rmtree(input_dir)
