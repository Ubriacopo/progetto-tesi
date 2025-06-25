## Now let's think!

To avoid connecting all the time remotely (I can start by working here) I sample some of the data to check.<br>
What I believe is todo:

- Find structural similarities between the datasets
- Pre-process the data to be all in same freq domain etc. (Approfondisci le tue lacune)

### AMIGOS:

> EEG_DATA: EEG data was recorded using the EMOTIV Epoc, with a sampling frequency of 128 Hz.
> EEG recordings are stored in the variable EEG_DATA as a list of 20 matrixes of 25 columns, one for each of the videos.

- Participant_questionnaire: (xls, ods spreadsheet)    
  The answers participants gave to the questionnaire before the experiment.
- Experiment_data:    (xls, ods spreadsheet)    
  Order of the videos for both short and long videos experiments.
- Participant_Personality    (xls, ods spreadsheet)   
  The answer participants gave to the personality traits
  questionnaire, and calculated estimated personality traits.
- Participant_PANAS:    (xls, ods spreadsheet)
  The answer participants gave to the mood (PANAS) questionnaire, and calculated Positive Affect (PA) and Negative
  Affect (NA).
- Video_List:    (xls, ods spreadsheet)    
  Information of all the videos used in both experiments.
- Face_video: The frontal face video recordings, through an HD camera from both experiments.
- RGB_kinect: The frontal full-body RGB video recordings, through Kinect RBG camera from both experiments.
- Depth_kinect: The frontal full-body depth video recordings, through Kinect depth sensor from both experiments.
- Frame_timestamps: Timestamps for the frames obtained through Kinect sensors.
- Data_original: The original unprocessed physiological data recordings from both experiments in Matlab .mat format.
- Data_preprocessed: The preprocessed (downsampling, EOG removal, filtering, segmenting, etc.) physiological data
  recordings from both experiment in Matlab .mat format.
- Self_Assessment: Self-Assessment of the 20 videos of both experiments.
- External_Annotations: External annotations of valence and arousal for the 20 second

### DEAP:

The DEAP dataset uses three self-assessed dimensions to describe emotional responses to music videos:

---

#### 1. Valence

- **Meaning**: Measures how **pleasant or unpleasant** an emotion feels.
- **Scale**:
    - `1` = Very unpleasant
    - `9` = Very pleasant
- **Examples**:
    - Low valence → Sadness
    - High valence → Joy

---

#### 2. Arousal

- **Meaning**: Measures the **intensity or activation** of the emotion.
- **Scale**:
    - `1` = Very calm/sleepy
    - `9` = Very excited/alert
- **Examples**:
    - Low arousal → Boredom
    - High arousal → Anger, Excitement

---

#### 3. Dominance

- **Meaning**: Measures the sense of **control or power** experienced during the emotion.
- **Scale**:
    - `1` = Very submissive/controlled
    - `9` = Very dominant/in control
- **Examples**:
    - Low dominance → Fear
    - High dominance → Confidence

---

Together, these form the **VAD model**, a 3D space for representing emotions more richly than basic categorical labels.

### DREAMER: