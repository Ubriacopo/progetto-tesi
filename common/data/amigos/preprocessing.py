from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd

from common.data.amigos.loader import AMIGOSLoader
from common.data.sample_container import ResizeVideo, SubclipMedia
from common.data.eeg.utils import EEGNMERawTransform, EEGNMEAddAnnotationTransform
from common.data.loader import EEGDatasetDataCollection
from common.data.sampler import Segmenter, FixedIntervalsSegmenter
from common.data.transform import Compose

GENERATED_MEDIA_EXTENSIONS = [".mp4", ".wav", ".txt"]


class AMIGOSPreProcessing:
    ch_names = [
        # EEG Channels
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
        # Others
        "ECG Right", "ECG Left", "GSR"
    ]
    ch_types = ["eeg"] * 14 + ["ecg"] * 2 + ["misc"]

    # TODO Can be made generic procedure for any ds?
    @staticmethod
    def procedure(source_path: str, output_path: str):
        pipeline = AMIGOSPreProcessing(
            output_path,
            FixedIntervalsSegmenter(8),
            Compose([
                SubclipMedia(),
                EEGNMEAddAnnotationTransform(),
                ResizeVideo((260, 260))
            ])
        )
        existing_df: Optional[pd.DataFrame] = None
        if Path(output_path + "spec.csv").exists():
            existing_df = pd.read_csv(output_path + "spec.csv")

        docs: list[EEGDatasetDataCollection] = []
        for i in AMIGOSLoader(source_path).scan():
            if existing_df is not None and existing_df["entry_id"].str.contains(i.entry_id).any():
                continue  # This element was already processed.
            [docs.append(e) for e in pipeline.pre_process(i)]
            df = pd.DataFrame([d.to_dict() for d in docs])
            if existing_df is not None:
                df = pd.concat([df, existing_df], ignore_index=True)

            df.to_csv(output_path + "spec.csv", index=False)
            existing_df = df

    def __init__(self, output_path: str, segmenter: Segmenter, transform: Compose):
        self.segmenter = segmenter
        self.output_path = output_path
        self.transform = transform  # Transforms applied to EEGDatasetDataCollection

    def process_split(self, x: EEGDatasetDataCollection, idx: int, segment: tuple[int, int], out_folder: str):
        nid = x.entry_id + "_" + str(idx)
        # Take Video-Audio-Text
        y = EEGDatasetDataCollection(
            entry_id=nid,
            # No files path are associated now.
            eeg=replace(x.eeg, interval=segment),
            vid=replace(x.vid, interval=segment),
            aud=replace(x.aud, interval=segment),
            txt=replace(x.txt) if x.txt is not None else None,
        )

        y: EEGDatasetDataCollection = self.transform(y)

        # Save Video
        vid_out_path = out_folder + f'{y.entry_id}.mp4'
        y.vid.data.write_videofile(vid_out_path, audio=False, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"], )
        y.vid.file_path = vid_out_path

        # Save Audio
        aud_out_path = out_folder + f'{y.entry_id}.wav'
        y.aud.data.write_audiofile(aud_out_path)
        y.aud.file_path = aud_out_path

        # Save Text if exists
        if y.txt is not None and y.txt.data is not None:
            txt_out_path = out_folder + f'{y.entry_id}.txt'
            with open(txt_out_path, "w", encoding="utf-8") as f:
                f.write(y.txt.data)

        # EEG data is treated differently and aggregated to save space.
        # So it is not performed here.
        return y

    # Concurrency made deadlocks, will be an enhancement
    def pre_process(self, sample: EEGDatasetDataCollection, concurrent: bool = False):
        original_sample_id = sample.entry_id
        sample.eeg.data = sample.eeg.data.T  # Channels are first
        sample = EEGNMERawTransform(channel_names=self.ch_names, channel_types=self.ch_types)(sample)
        # We have already EEG data loaded we can apply any segmenter that works on those heuristics
        # Is a future enhancement to implement
        segments: list[tuple[int, int]] = self.segmenter.compute_segments(sample)

        sample_folder = self.output_path + sample.entry_id + "/"
        Path(sample_folder).mkdir(exist_ok=True)
        splits = [self.process_split(sample, idx, segment, sample_folder) for idx, segment in enumerate(segments)]

        print("Done with sample ", sample.entry_id)
        sample.eeg.data.save(self.output_path + f'{original_sample_id}.fif', overwrite=True, split_size="2GB")

        for s in splits:
            s.eeg.file_path = self.output_path + f'{original_sample_id}.fif'

        return splits


if __name__ == "__main__":
    AMIGOSPreProcessing.procedure("../../../resources/AMIGOS/", "../../../resources/AMIGOS/processed/")
