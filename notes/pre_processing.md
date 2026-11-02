# Preprocessing notes

## Shared goals

- Preserve participant and trial identifiers.
- Convert every modality to a documented sampling rate.
- Align all streams on a common time axis.
- Store validity masks for missing or padded data.
- Split by participant before generating overlapping windows.
- Keep raw, extracted, and model-ready data separate.

## AMIGOS

EEG was recorded with an EMOTIV EPOC at 128 Hz. The source package contains:

- original and preprocessed physiological recordings;
- frontal face, RGB, and depth videos;
- frame timestamps;
- stimulus metadata;
- self-assessments and external annotations;
- personality, mood, and participant questionnaires.

The long-video experiment includes individual and group sessions. Group video handling must be fixed before the final preprocessing run.

## DEAP

DEAP uses valence, arousal, and dominance as its main affective dimensions. Ratings use a 1–9 scale. Participant ratings also include liking and familiarity.

The original metadata provides alignment between each physiological trial and its one-minute stimulus excerpt. Frontal participant videos do not contain the stimulus audio.

## DREAMER

The available data contains physiological streams and assessments. No stimulus video is available in the current copy. Downstream experiments therefore use physiological inputs only.
