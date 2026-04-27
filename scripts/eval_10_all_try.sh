#!/bin/bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/urgent2026_challenge_track1}"
OUT="${OUT:-$ROOT_DIR/enhanced/bsrnn_validation_10_a100}"
DEVICE="${DEVICE:-cuda}"
NJ="${NJ:-4}"

cd "$ROOT_DIR"

LOG_DIR="$OUT/logs/eval10_all"
mkdir -p "$LOG_DIR"
mkdir -p "$OUT/score"

run_metric() {
  local name="$1"
  shift

  echo
  echo "===== Running $name ====="
  echo "Command: $*"
  echo

  "$@" > "$LOG_DIR/${name}.out" 2> "$LOG_DIR/${name}.err"
  local status=$?

  if [[ $status -eq 0 ]]; then
    echo "[OK] $name"
  else
    echo "[FAIL] $name, exit=$status"
    echo "  stdout: $LOG_DIR/${name}.out"
    echo "  stderr: $LOG_DIR/${name}.err"
  fi

  return 0
}

run_metric intrusive_se \
  python evaluation_metrics/calculate_intrusive_se_metrics.py \
    --ref_scp "$OUT/ref.scp" \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/se" \
    --nj "$NJ" \
    --chunksize 1

run_metric dnsmos \
  python evaluation_metrics/calculate_nonintrusive_dnsmos.py \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/dnsmos" \
    --device "$DEVICE"

run_metric nisqa \
  python evaluation_metrics/calculate_nonintrusive_nisqa.py \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/nisqa" \
    --device "$DEVICE"

run_metric utmos \
  python evaluation_metrics/calculate_nonintrusive_utmos.py \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/utmos" \
    --device "$DEVICE"

run_metric scoreq \
  python evaluation_metrics/calculate_nonintrusive_scoreq.py \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/scoreq" \
    --device "$DEVICE"

run_metric speechbert_score \
  python evaluation_metrics/calculate_speechbert_score.py \
    --ref_scp "$OUT/ref.scp" \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/speechbert_score" \
    --device "$DEVICE"

run_metric phoneme_similarity \
  python evaluation_metrics/calculate_phoneme_similarity.py \
    --ref_scp "$OUT/ref.scp" \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/lps" \
    --device "$DEVICE"

run_metric speaker_similarity \
  python evaluation_metrics/calculate_speaker_similarity.py \
    --ref_scp "$OUT/ref.scp" \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/spk_sim" \
    --device "$DEVICE"

run_metric emotion_similarity \
  python evaluation_metrics/calculate_emotion_similarity.py \
    --ref_scp "$OUT/ref.scp" \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/emo_sim" \
    --device "$DEVICE"

run_metric lid_accuracy \
  python evaluation_metrics/calculate_lid_accuracy.py \
    --meta_tsv "$OUT/utt2lang" \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/lid_acc" \
    --device "$DEVICE"

run_metric wer \
  python evaluation_metrics/calculate_wer.py \
    --meta_tsv "$OUT/text" \
    --utt2lang "$OUT/utt2lang" \
    --inf_scp "$OUT/inf.scp" \
    --output_dir "$OUT/score/cer" \
    --device "$DEVICE"

echo
echo "===== Summary files found ====="
find "$OUT/score" -name RESULTS.txt -print -exec cat {} \;

echo
echo "===== Failed logs quick scan ====="
grep -R "ModuleNotFoundError\|ImportError\|No module named\|doesn't exist\|not found\|HTTPError\|URLError" "$LOG_DIR" || true


