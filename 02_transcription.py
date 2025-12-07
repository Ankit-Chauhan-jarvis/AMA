import whisper
import time
import json
import os
from pathlib import Path


AUDIO_FILE = "./output/input_prep.wav" 
OUTPUT_DIR = Path("./output")
MODEL_SIZE = "large"


def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: {AUDIO_FILE} not found. Please run the Diarization script first.")
        return

    # Start Timer
    start_time = time.time()

    print(f"--- Step 1: Loading Whisper Model ({MODEL_SIZE}) ---")
    model = whisper.load_model(MODEL_SIZE)

    print("\n--- Step 2: Transcribing Audio ---")
    # We enable word_timestamps to get token-level timing
    result = model.transcribe(
        AUDIO_FILE, 
        word_timestamps=True,
        fp16=False # Set to True if using GPU, False usually safer for CPU
    )

    # End Timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n--- Step 3: Saving Results ---")
    
    # 1. Save raw text
    txt_path = OUTPUT_DIR / "transcript_text.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    # 2. Save detailed segments (JSON)
    # This is crucial for the merging step later. 
    # It contains 'segments', 'start', 'end', and 'words' (token timestamps)
    json_path = OUTPUT_DIR / "transcript_detailed.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Raw text saved to: {txt_path}")
    print(f"Detailed JSON saved to: {json_path}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    # Preview of data structure for next step
    print("\n[Preview of first segment data]")
    if result['segments']:
        first_seg = result['segments'][0]
        print(f"Text: {first_seg['text']}")
        print(f"Start: {first_seg['start']}")
        print(f"End: {first_seg['end']}")
        if 'words' in first_seg:
            print(f"Token Timestamps available: Yes ({len(first_seg['words'])} words)")

if __name__ == "__main__":
    main()