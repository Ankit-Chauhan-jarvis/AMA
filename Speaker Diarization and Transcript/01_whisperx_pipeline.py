import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import gc
import json
import time
import torch
import subprocess
from pathlib import Path


VIDEO_PATH    = "Editors_Extensions_20231207.mp4"
ACCESS_TOKEN  = "Put your key here"
OUTPUT_DIR    = Path("./output")


WHISPER_MODEL = "large-v2"
LANGUAGE      = "en"         

MIN_SPEAKERS  = 2
MAX_SPEAKERS  = 11

NOISE_REDUCE_STRENGTH = 0.75


def preprocess_audio(video_path, output_dir):
    print("--- Step 1: Converting Video to WAV ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "input_prep.wav"

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    command = (
        f'ffmpeg -i "{video_path}" '
        f'-vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{wav_path}" -loglevel error'
    )
    subprocess.run(command, shell=True, check=True)
    print(f"  Audio saved to: {wav_path}")
    return wav_path


def denoise_audio(input_wav, output_dir, strength):
    
    print("\n--- Step 2: Denoising Audio (noisereduce) ---")
    output_wav = output_dir / "input_denoised.wav"

    try:
        import noisereduce as nr
        import soundfile as sf

        audio, sr = sf.read(str(input_wav))

        noise_sample = audio[:int(sr * 0.5)]

        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_sample,
            prop_decrease=strength
        )

        sf.write(str(output_wav), reduced, sr)
        print(f"  Denoised audio saved to: {output_wav}")
        return output_wav

    except ImportError:
        print("  noisereduce not installed — skipping denoising.")
        print("  Run: pip install noisereduce soundfile")
        return input_wav
    except Exception as e:
        print(f"  Denoising failed ({e}) — using original audio.")
        return input_wav


def transcribe(audio, device):
    
    import whisperx
    print(f"\n--- Step 3: Transcribing with Whisper ({WHISPER_MODEL}) ---")
    print(f"  (If this is your first run, the model will download now — do not cancel)")

    batch_size   = 16 if device == "cuda" else 4
    compute_type = "float16" if device == "cuda" else "int8"

    model  = whisperx.load_model(WHISPER_MODEL, device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size, language=LANGUAGE)

    print(f"  Detected language : {result['language']}")
    print(f"  Segments found    : {len(result['segments'])}")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def align_timestamps(result, audio, device):
    
    import whisperx
    print("\n--- Step 4: Forced Alignment (word-level timestamps) ---")

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )
    print(f"  Alignment complete")

    del align_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def diarize(audio, device):
    
    from whisperx.diarize import DiarizationPipeline
    print(f"\n--- Step 5: Speaker Diarization "
          f"(min={MIN_SPEAKERS}, max={MAX_SPEAKERS}) ---")

    diarize_model    = DiarizationPipeline(
        token=ACCESS_TOKEN,
        device=device
    )
    diarize_segments = diarize_model(
        audio,
        min_speakers=MIN_SPEAKERS,
        max_speakers=MAX_SPEAKERS
    )
    return diarize_segments


def assign_speakers(result, diarize_segments):
    
    from whisperx.diarize import assign_word_speakers
    print("\n--- Step 6: Assigning Speakers to Words ---")
    result = assign_word_speakers(diarize_segments, result)

    speakers = set()
    for seg in result["segments"]:
        if "speaker" in seg:
            speakers.add(seg["speaker"])
    print(f"  Final speaker count : {len(speakers)} → {sorted(speakers)}")
    return result


def accurate_segment_timestamps(seg):
    
    MAX_SINGLE_WORD_SEC  = 1.0
    SHORT_SEGMENT_WORDS  = 2   

    if 'words' not in seg or not seg['words']:
        return seg['start'], seg['end']

    words = [w for w in seg['words'] if 'start' in w and 'end' in w]
    if not words:
        return seg['start'], seg['end']

    actual_start = words[0]['start']
    last_word    = words[-1]
    n_words      = len(words)
    raw_end      = last_word['end']

    if n_words <= SHORT_SEGMENT_WORDS:
        fixed_end = min(raw_end, last_word['start'] + MAX_SINGLE_WORD_SEC)
    else:
        fixed_end = raw_end

    return actual_start, fixed_end


def merge_consecutive_turns(segments):
    
    if not segments:
        return segments

    merged = [segments[0].copy()]
    for current in segments[1:]:
        prev = merged[-1]
        if current.get("speaker") == prev.get("speaker"):
            prev["end"]  = current["end"]
            prev["text"] = prev["text"].rstrip() + " " + current["text"].lstrip()
        else:
            merged.append(current.copy())

    return merged


def save_outputs(result, output_dir):
    print("\n--- Step 7: Saving Outputs ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem  = Path(VIDEO_PATH).stem
    merged_name = f"{video_stem}.json"

    segments = result["segments"]

    for seg in segments:
        corrected_start, corrected_end = accurate_segment_timestamps(seg)
        seg["start"] = corrected_start
        seg["end"]   = corrected_end
        if seg.get("words"):
            words = [w for w in seg["words"] if "start" in w and "end" in w]
            if words:
                last_w = words[-1]
                if last_w["end"] > corrected_end:
                    last_w["end"] = corrected_end

    for seg in segments:
        word_speakers = [
            w["speaker"] for w in seg.get("words", [])
            if w.get("speaker") and "start" in w and "end" in w
        ]
        if word_speakers:
            seg["speaker"] = max(set(word_speakers), key=word_speakers.count)

    seen = {}
    counter = 1
    for seg in segments:
        sp = seg.get("speaker")
        if sp and sp not in seen:
            seen[sp] = f"SPEAKER_{counter:02d}"
            counter += 1
    
    for seg in segments:
        if seg.get("speaker") in seen:
            seg["speaker"] = seen[seg["speaker"]]
        for word in seg.get("words", []):
            if word.get("speaker") in seen:
                word["speaker"] = seen[word["speaker"]]

    diarization_txt = output_dir / "diarization.txt"
    with open(diarization_txt, "w", encoding="utf-8") as f:
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            f.write(f"start={seg['start']:.3f} stop={seg['end']:.3f} speaker={speaker}\n")
    print(f"  Diarization TXT    : {diarization_txt}")

    transcript_json = output_dir / "transcript_detailed.json"
    with open(transcript_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"  Transcript JSON    : {transcript_json}")
    print(f"    → {len(segments)} segments, word-level timestamps + speaker labels")

    raw_turns = []
    for seg in segments:
        raw_turns.append({
            "speaker" : seg.get("speaker", "UNKNOWN"),
            "start"   : round(seg["start"], 3),
            "end"     : round(seg["end"],   3),
            "text"    : seg["text"].strip()
        })

    final_turns = merge_consecutive_turns(raw_turns)
    final_turns.sort(key=lambda x: x["start"])

    merged_json = output_dir / merged_name
    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump(final_turns, f, indent=4, ensure_ascii=False)
    print(f"  Final merged JSON  : {merged_json}")
    print(f"    → {len(final_turns)} merged speaker turns")

    print("\n[Preview — first 5 speaker turns]")
    for turn in final_turns[:5]:
        print(
            f"  [{turn['start']:>8.3f} → {turn['end']:>8.3f}]  "
            f"{turn['speaker']:<12}  {turn['text'][:70]}"
        )

    return final_turns


def main():
    start_time = time.time()

    try:
        wav_path = preprocess_audio(VIDEO_PATH, OUTPUT_DIR)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return

    clean_wav = denoise_audio(wav_path, OUTPUT_DIR, NOISE_REDUCE_STRENGTH)

    try:
        import whisperx
    except ImportError:
        print("\nERROR: whisperx is not installed.")
        print("Run: pip install whisperx")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    if device == "cpu":
        print("  Note: Running on CPU. large-v2 will be very slow — consider WHISPER_MODEL = 'medium'")

    audio = whisperx.load_audio(str(clean_wav))

    result = transcribe(audio, device)

    result = align_timestamps(result, audio, device)

    diarize_segments = diarize(audio, device)

    result = assign_speakers(result, diarize_segments)

    save_outputs(result, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()