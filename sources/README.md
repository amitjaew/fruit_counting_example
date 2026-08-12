# Sources

## Video files

```
sources/L{n}.mp4
```

Where `{n}` is a sequence number (e.g., `L0.mp4`, `L1.mp4`, `L2.mp4`).

## Extracted frames

```
sources/L{n}/00001.png
sources/L{n}/00002.png
...
```

Each `L{n}/` directory contains all frames extracted from the corresponding video.
Frame filenames are zero-padded to 5 digits.

Extract with:

```bash
pipenv run python helpers/extract_frames.py sources/L{n}.mp4
```
