## Space Requirements (After preprocessing)
After processing each video origin is ~128MB.<br>
With 20 videos per person it is around 2.5GB. <br>
Experiments are on 40 people thus 100GB AMIGOS. 
> Tune that up by 20% for errors so we ask 120GB for amigos



## Download AMIGOS dataset:

Just run the ```amigos_download_script.py``` with args:

- "-p", "--path": output path of the generation

Face folder example output:

- P{0}_{1}_face.mov: Experiment n. {1} of Person {0}
- P{0}_{L}{1}_face.mov: Epxeriment n. {1} of Person {0} for

> Face_video contains the frontal face videos recorded from both experiments.
> Videos of the short videos experiment have been sorted in 40 .zip files, one for each of the recording sessions.
> File Exp1_PXX_face.zip corresponds to the trials of participant XX. In the zip file, PXX_VideoID_face.mov corresponds
> to the face video for the stimuli video VideoID of participant XX. Videos of the long videos experiment have been
> sorted in 22 .zip files, one for each of the recording sessions. File Exp2_LXX_TYY_NZZ_face.zip corresponds to
> the trials of recording session XX, type YY (Ind, Group), and Group/Individual ZZ. In the zip files of individuals'
> recordings, PXX_VideoID_face.mov corresponds to the face video for the stimuli video VideoID of participant XX.
> In the zip files of group recordings, P(XX1,XX2,XX3,XX4)_VideoID_face.mov corresponds to the face video for the
> stimuli video VideoID of participants XX1,XX2,XX3, and XX4. UserIDs are listed in the order the participants were sit,
> on a front view, during the recording session from left to right. 