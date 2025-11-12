from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from Levenshtein import distance
from torch.nn import Module
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


METRICS = ("PhonemeSimilarity",)
TARGET_FS = 16000


################################################################
# Definition of metrics
################################################################
class PhonemePredictor(Module):
    # espeak installation is required for this function to work
    # To install, try
    # https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#linux
    def __init__(
        self, checkpoint="facebook/wav2vec2-lv-60-espeak-cv-ft", sr=16000, device="cpu"
    ):
        # https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(checkpoint, device=device)
        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint).to(device)
        self.sr = sr
        self.device = device

    def forward(self, waveform):
        input_values = self.processor(waveform, return_tensors="pt", sampling_rate=self.sr).input_values
        if len(input_values.shape) == 3:
            input_values = input_values.squeeze(0)
        # retrieve logits
        logits = self.model(input_values.to(self.device)).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)


class LevenshteinPhonemeSimilarity:
    def __init__(self, device="cpu", reduce="mean"):
        self.phoneme_predictor = PhonemePredictor(device=device)
        self.reduce = reduce

    def __call__(self, reference: np.ndarray, sample: np.ndarray) -> float:
        sample_phonems = [r.split(" ") for r in self.phoneme_predictor.forward(sample)]
        ref_phonems = [r.split(" ") for r in self.phoneme_predictor.forward(reference)]
        assert len(sample_phonems) == len(ref_phonems), "Sample and reference must have the same number of utterances"
        lev_distance = [distance(rs, rr) for rs, rr in zip(sample_phonems, ref_phonems)]
        lps = [1 - ld / len(rr) for ld, rr in zip(lev_distance, ref_phonems)]
        if len(lps) == 1:
            return lps[0]
        elif self.reduce == "mean":
            return sum(lps) / len(lps)
        elif self.reduce == "sum":
            return sum(lps)
        elif self.reduce == None:
            return lps
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce}.")



def phoneme_similarity_metric(model, ref, inf, fs=16000):
    """Calculate the similarity between ref and inf phoneme sequences.

    Args:
        model (torch.nn.Module): phoneme recognition model
            Please use the model in
            https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
            to get comparable results.
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        similarity (float): phoneme similarity value between (-inf, 1]
    """
    if fs != TARGET_FS:
        ref = soxr.resample(ref, fs, TARGET_FS)
        inf = soxr.resample(inf, fs, TARGET_FS)
    with torch.no_grad():
        similarity = model(ref, inf)
    return similarity


################################################################
# Main entry
################################################################
def main(args):
    refs = {}
    with open(args.ref_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            refs[uid] = audio_path

    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            if uid in refs:
                data_pairs.append((uid, refs[uid], audio_path))

    size = len(data_pairs)
    assert 1 <= args.job <= args.nsplits <= size
    interval = size // args.nsplits
    start = (args.job - 1) * interval
    end = size if args.job == args.nsplits else start + interval
    data_pairs = data_pairs[start:end]
    print(
        f"[Job {args.job}/{args.nsplits}] Processing ({len(data_pairs)}/{size}) samples",
        flush=True,
    )
    suffix = "" if args.nsplits == args.job == 1 else f".{args.job}"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {
        metric: (outdir / f"{metric}{suffix}.scp").open("w") for metric in METRICS
    }

    model = LevenshteinPhonemeSimilarity(device=args.device)
    model.phoneme_predictor.eval()
    ret = []
    for uid, ref_audio, inf_audio in tqdm(data_pairs):
        _, score = process_one_pair((uid, ref_audio, inf_audio), model=model)
        ret.append((uid, score))
        for metric, value in score.items():
            writers[metric].write(f"{uid} {value}\n")

    for metric in METRICS:
        writers[metric].close()

    if args.nsplits == args.job == 1:
        with (outdir / "RESULTS.txt").open("w") as f:
            for metric in METRICS:
                mean_score = np.nanmean([score[metric] for uid, score in ret])
                f.write(f"{metric}: {mean_score:.4f}\n")
        print(
            f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True
        )


def process_one_pair(data_pair, model=None):
    uid, ref_path, inf_path = data_pair
    ref, fs = sf.read(ref_path, dtype="float32")
    inf, fs2 = sf.read(inf_path, dtype="float32")
    assert fs == fs2, (fs, fs2)
    assert ref.shape == inf.shape, (ref.shape, inf.shape)
    assert ref.ndim == 1, ref.shape

    scores = {}
    for metric in METRICS:
        if metric == "PhonemeSimilarity":
            scores[metric] = phoneme_similarity_metric(model, ref, inf, fs=fs)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_scp",
        type=str,
        required=True,
        help="Path to the scp file containing reference signals",
    )
    parser.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        help="Path to the scp file containing enhanced signals",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for writing metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for running speaker embedding extraction",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=1,
        help="Total number of computing nodes to speed up evaluation",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=1,
        help="Index of the current node (starting from 1)",
    )
    args = parser.parse_args()

    main(args)
