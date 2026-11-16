# Model checkpoints

All 32 existing `.pt` files under the `fire-harmonization*` folders are included
using Git LFS, with lossless parts for two oversized files. This includes training checkpoints (`best.pt`, `last.pt`) and
inference exports from PLSWORK, full-resolution training, teacher/student training,
YESSS, saturation-aware V2, and wide-diverse V1/V2. The original `fire-harmonization-research` folder
currently contains no `.pt` files.

After cloning with Git LFS installed, retrieve the checkpoint contents with:

```sh
git lfs install
git lfs pull
python restore-model-checkpoints.py
```

[model-checkpoints.json](model-checkpoints.json) lists every included path, byte
size, and SHA-256 checksum. The model files are preserved as produced by training.
They are research artifacts; including a checkpoint does not establish its quality.

The wide-diverse V2 `large200_d1/best.pt` and `last.pt` checkpoints are each about
2.42 GB. GitHub's individual-account LFS limit is 2 GB per file, so these two are
stored in adjacent `.pt.parts/` directories as chunks of at most 1 GiB. The restore
script verifies every chunk and the complete original SHA-256 before publishing
the reconstructed `.pt` file. An existing checkpoint is verified and left intact;
a different existing file produces an error instead of being overwritten.

The four other wide-diverse checkpoints, including both inference exports, are
stored directly as `.pt` LFS files. Total original checkpoint size is 7,254,779,362
bytes. A fresh clone needs additional disk space to retain the parts and restore
the two larger checkpoints. To verify all checkpoint contents without creating
the two reconstructed files, run `python restore-model-checkpoints.py --verify-only`.

See [GitHub's LFS size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

Existing ignore rules for generated training artifacts remain in place. To publish
a new checkpoint, explicitly add its path with `git add -f path/to/checkpoint.pt`;
the repository's attributes apply Git LFS automatically. Update the inventory when
adding or replacing checkpoint files.
