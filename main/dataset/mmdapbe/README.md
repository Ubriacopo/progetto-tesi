# MMDAPBE exploration notes

MMDAPBE is not part of the current EEGAVI pretraining corpus. \
Its source package appears to contain physiological archives, camera recordings, audio files, synchronization data, NIRS signals, and a behaviour matrix.

Before adding an adapter, verify:

- the contents and channel layout of the physiological archives;
- the relationship between original and reduced audio files;
- the synchronization format;
- whether repeated sessions belong to the same participant;
- which fields contain self-assessments;
- whether NIRS is in scope.
