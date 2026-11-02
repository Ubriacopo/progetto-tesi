# Open tasks

This file tracks unresolved implementation and experiment work. Thesis writing tasks belong in [`report/tesi/note.md`](report/tesi/note.md).

## Data

- Define a consistent policy for AMIGOS group recordings.
- Decide whether to crop each participant from group videos.
- Extract long recordings into aligned 20-second segments.
- Avoid repeated transcript extraction for shared stimulus videos.
- Complete MAHNOB-HCI preprocessing and integration.
- Verify the DEAP preprocessing pipeline.
- Run the remaining dataset preprocessing jobs.

## Model

- Restore modality dropout.
- Evaluate a unimodal EEG path.
- Review loss balancing.
- Confirm the safe video token limit during training.

Video without pooling is currently impractical. Precomputed features exceed the available memory.
