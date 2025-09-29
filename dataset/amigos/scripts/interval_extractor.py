from core_data import sampler
from core_data.extract import SegmentBasedExtractionProcessor
from core_data.media.text.extract import ExtractTextFromAudio
from core_data.media.text.transforms import WhisperExtractor
from core_data.sampler import EegFeaturesAndRandLogUIntervalsSegmenter
from dataset.amigos.loader import AmigosPointsLoader
from utils.args import work_with_config_file

cfg = work_with_config_file("./interval_sampler_default.json")
base_path = cfg["base_path"]

segmenter = EegFeaturesAndRandLogUIntervalsSegmenter(
    min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
)
if "segmenter" in cfg:
    segmenter_type = getattr(sampler, cfg["segmenter"]["type"])
    segmenter = segmenter_type(**cfg["segmenter"]["kwargs"])

output_path = cfg["base_path"] + cfg["output_path"]

SegmentBasedExtractionProcessor(
    ExtractTextFromAudio(WhisperExtractor(model_id="openai/whisper-medium", device="cuda:0")),
    # ExtractTextFromAudio(WhisperExtractor()), Much slower. GPU wins and we trade off in results
    base_path=output_path,
    segmenter=segmenter,
    loader=AmigosPointsLoader(base_path + cfg["data_path"]),
).extract_segments()
