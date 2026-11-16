# Model checkpoints

All 26 existing `.pt` files under the `fire-harmonization*` folders are included
using Git LFS. This includes training checkpoints (`best.pt`, `last.pt`) and
inference exports from PLSWORK, full-resolution training, teacher/student training,
YESSS, and saturation-aware V2. The original `fire-harmonization-research` folder
currently contains no `.pt` files.

After cloning with Git LFS installed, retrieve the checkpoint contents with:

```sh
git lfs install
git lfs pull
```

[model-checkpoints.json](model-checkpoints.json) lists every included path, byte
size, and SHA-256 checksum. The model files are preserved as produced by training.
They are research artifacts; including a checkpoint does not establish its quality.

Existing ignore rules for generated training artifacts remain in place. To publish
a new checkpoint, explicitly add its path with `git add -f path/to/checkpoint.pt`;
the repository's attributes apply Git LFS automatically. Update the inventory when
adding or replacing checkpoint files.
