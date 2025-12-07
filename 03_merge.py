import json
import re
import os

WHISPER_FILE = "output/transcript_detailed.json"
DIARIZATION_FILE = "output/diarization.txt"
OUTPUT_FILE = "output/1.json"

def parse_diarization(file_path):
    """Parses the Pyannote text output into a list of segments."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Diarization file not found: {file_path}")
        
    segments = []
    with open(file_path, 'r') as f:
        for line in f:
            # Regex to extract values from: start=3.846 stop=10.161 speaker=SPEAKER_00
            match = re.search(r'start=(\d+\.\d+) stop=(\d+\.\d+) speaker=(\w+)', line)
            if match:
                segments.append({
                    'start': float(match.group(1)),
                    'end': float(match.group(2)),
                    'speaker': match.group(3)
                })
    return segments

def get_speaker_for_timestamp(timestamp, diarization_segments):
    """Finds the speaker active at a specific timestamp."""
    # 1. Check for exact overlap
    for seg in diarization_segments:
        if seg['start'] <= timestamp <= seg['end']:
            return seg['speaker']
    
    # 2. If no overlap (word spoken during slight silence/gap), find closest segment
    closest_speaker = "UNKNOWN"
    min_dist = float('inf')
    
    for seg in diarization_segments:
        # Distance to start or end of segment
        dist = min(abs(timestamp - seg['start']), abs(timestamp - seg['end']))
        if dist < min_dist:
            min_dist = dist
            closest_speaker = seg['speaker']
            
    return closest_speaker

def main():
    print("--- Step 1: Loading Data ---")
    
    if not os.path.exists(WHISPER_FILE):
        print(f"Error: {WHISPER_FILE} not found. Run transcription first.")
        return

    # Load Whisper Data
    with open(WHISPER_FILE, 'r', encoding='utf-8') as f:
        whisper_data = json.load(f)
    
    # Load Diarization Data
    try:
        diarization_segments = parse_diarization(DIARIZATION_FILE)
        print(f"Loaded {len(diarization_segments)} speaker turns from Pyannote.")
    except FileNotFoundError as e:
        print(e)
        return

    print("\n--- Step 2: Merging Word-Level Timestamps ---")
    
    final_turns = []
    current_speaker = None
    current_text_buffer = []
    current_start = 0
    
    # Flatten Whisper segments to iterate over words
    all_words = []
    if 'segments' in whisper_data:
        for segment in whisper_data['segments']:
            if 'words' in segment:
                all_words.extend(segment['words'])
    
    if not all_words:
        print("Error: No word-level timestamps found in Whisper JSON. Ensure 'word_timestamps=True' was used in transcription.")
        return

    for i, word in enumerate(all_words):
        word_text = word['word'].strip()
        word_start = word['start']
        word_end = word['end']
        
        # Calculate the middle of the word to determine ownership
        word_midpoint = (word_start + word_end) / 2
        
        # Identify Speaker
        speaker = get_speaker_for_timestamp(word_midpoint, diarization_segments)
        
        # Logic to group words into turns
        if speaker != current_speaker:
            # If speaker changed (and it's not the very first word), save the previous turn
            if current_speaker is not None:
                final_turns.append({
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": all_words[i-1]['end'], # End time of the previous word
                    "text": " ".join(current_text_buffer)
                })
            
            # Reset for new speaker
            current_speaker = speaker
            current_text_buffer = [word_text]
            current_start = word_start
        else:
            # Same speaker, keep building the sentence
            current_text_buffer.append(word_text)

    # Append the final remaining turn
    if current_text_buffer:
        final_turns.append({
            "speaker": current_speaker,
            "start": current_start,
            "end": all_words[-1]['end'],
            "text": " ".join(current_text_buffer)
        })

    print("\n--- Step 3: Sorting and Saving ---")
    
    # Explicitly sort by start timestamp to ensure increasing order
    print("Sorting segments by timestamp...")
    final_turns.sort(key=lambda x: x['start'])

    # Save Final JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_turns, f, indent=4, ensure_ascii=False)
    
    print(f"Success! Final sorted analysis saved to: {OUTPUT_FILE}")
    
    # Preview
    print("\n[Preview of first 3 sorted turns]")
    for turn in final_turns[:3]:
        print(f"[{turn['start']:.1f} - {turn['end']:.1f}] {turn['speaker']}: {turn['text'][:50]}...")

if __name__ == "__main__":
    main()