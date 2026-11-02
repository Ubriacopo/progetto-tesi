# AMIGOS adapter

The AMIGOS adapter handles physiological signals, frontal videos, stimulus media, and assessments. Raw data must be obtained under the dataset's access terms.

## Download helper

The repository includes a helper for authorized downloads:

```bash
python -m main.dataset.amigos.scripts.download_script --path /path/to/amigos
```

The current arguments are listed by `--help`.

## Face video naming

- `P<participant>_<video>_face.mov` identifies a short-video trial.
- Long-video files include an `L` marker.
- Group recordings list all participant identifiers in left-to-right order.

Group recordings require a consistent participant-cropping policy before preprocessing.

## Storage estimate

An early estimate placed the processed face-video corpus near 100 GB. A 120 GB allocation allows for temporary files and failed runs. The estimate remains provisional until the video pipeline is finalized.

The source archive layout is described in [`AMIGOS.md`](AMIGOS.md).
