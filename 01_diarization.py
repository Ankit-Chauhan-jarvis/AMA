import os
import time
import torch
import subprocess
from pathlib import Path
from pydub import AudioSegment
from pyannote.audio import Pipeline


VIDEO_PATH = "1.mp4" 

ACCESS_TOKEN = "key here" 
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_audio(input_path, output_dir):
    """Converts video to wav and adds 2 seconds silence."""
    print("--- Step 1: Preprocessing Audio ---")
    temp_wav = output_dir / "temp_input.wav"
    final_wav = output_dir / "input_prep.wav"
    
    # 1. Convert to wav, mono, 16kHz
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    command = f'ffmpeg -i "{input_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{temp_wav}" -loglevel error'
    subprocess.run(command, shell=True, check=True)

    # 2. Prepend 2-second silence (Spacer)
    # This helps Pyannote detect the first speaker more accurately
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = AudioSegment.from_wav(str(temp_wav))
    audio = spacer.append(audio, crossfade=0)
    
    audio.export(str(final_wav), format='wav')
    print(f"Processed audio saved to: {final_wav}")
    
    # Clean up temp file
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
        
    return final_wav

def main():
    # Start Timer
    start_time = time.time()

    # 1. Prepare Audio
    try:
        prepared_audio_path = preprocess_audio(VIDEO_PATH, OUTPUT_DIR)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    # 2. Load Pipeline
    print("\n--- Step 2: Loading Pyannote Pipeline ---")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=ACCESS_TOKEN
        )
    except Exception as e:
        print(f"Error loading pipeline. Check your HF Token. Details: {e}")
        return

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"Running diarization on: {device}")

    # 3. Run Diarization
    print("\n--- Step 3: Executing Diarization ---")
    diarization = pipeline(str(prepared_audio_path))

    # 4. Save Output
    output_rttm = OUTPUT_DIR / "diarization.rttm"
    output_txt = OUTPUT_DIR / "diarization.txt"
    
    # Save RTTM (Standard format)
    with open(output_rttm, "w") as rttm:
        diarization.write_rttm(rttm)

    # Save Human Readable TXT
    with open(output_txt, "w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            line = f"start={turn.start:.3f} stop={turn.end:.3f} speaker={speaker}"
            f.write(line + "\n")
            print(line) # Print to console

    # End Timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n[DONE] Diarization saved to {OUTPUT_DIR}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()