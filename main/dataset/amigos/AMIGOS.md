# AMIGOS source layout

AMIGOS separates the short- and long-video experiments.

- Short trials are distributed in one archive per participant. A file such as `PXX_VideoID_face.mov` contains the frontal recording for one stimulus.
- Long trials are distributed by recording session and condition. Individual archives contain one participant. Group archives contain several participants in a single frame.
- Participant identifiers in group filenames follow their left-to-right seating order.

The source package also contains physiological signals, self-assessments, external annotations, questionnaires, and stimulus metadata. Raw archives remain immutable. Extracted and processed data are stored in separate directories.
