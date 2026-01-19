import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

client = Groq(api_key=GROQ_API_KEY)

PARTICIPATION_PROMPT = """You are an expert in meeting participation analysis.
Classify each speaker's utterance into ONE of three participation types:

## Participation Types:

### full_participation
- Active, substantive contribution to the meeting
- Sharing new information, ideas, or solutions
- Asking meaningful questions that advance discussion
- Providing detailed explanations or technical content
- Taking ownership of tasks or responsibilities

Examples:
- "I've been working on the cache implementation and found that we need to update the schema"
- "So I think we should automate the merge requests using renovate"
- "I'll reach out to you for a sync call to discuss this further"

### non_participation
- Complete silence (no utterances from speaker)
- Only filler words with no content
- Very minimal acknowledgments that don't engage with content

Examples:
- "Yeah."
- "Okay."
- "Hmm."
- (silence - no contribution)

### pretend_participation
- Commenting on what others said without adding value
- Simply agreeing without elaboration
- Repeating what others said
- Responses without specificity
- Off-topic comments
- Asking questions that were already answered
- Deferring comments ("I'll look into that later")
- Delayed/vague responses

Examples:
- "Yeah, that sounds good" (just agreeing)
- "Right, like what Aaron said"
- "I think so too"
- "Sounds good. All right."
- "Yeah. All right, there's some more activity in the doc now"

## Task:
For each utterance below, classify the participation type.
Return ONLY a JSON array with {"speaker": "...", "text": "...", "participation": "..."} for each.

Utterances:
"""


def analyze_participation(transcript_path: str, output_path: str):
    """Analyze participation balance from meeting transcript."""
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    print(f"Total utterances: {len(transcript)}")
    
    # Process in batches
    batch_size = 10
    all_results = []
    
    for i in range(0, len(transcript), batch_size):
        batch = transcript[i:i + batch_size]
        print(f"Processing utterances {i+1}-{min(i+batch_size, len(transcript))}...")
        
        # Format utterances for prompt
        utterances_text = "\n".join([
            f"{idx+1}. [{item['speaker']}]: \"{item['text'][:500]}\"" 
            for idx, item in enumerate(batch)
        ])
        
        prompt = PARTICIPATION_PROMPT + utterances_text + "\n\nJSON Output:"
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-20b",
                temperature=0.0,
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean markdown if present
            if result.startswith("```"):
                result = re.sub(r'^```(?:json)?\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
            
            batch_results = json.loads(result)
            all_results.extend(batch_results)
            
        except Exception as e:
            print(f"Error: {e}")
            for item in batch:
                all_results.append({
                    "speaker": item["speaker"],
                    "text": item["text"],
                    "participation": "unknown"
                })
    
    # Calculate statistics per speaker
    speaker_stats = {}
    for item in all_results:
        speaker = item["speaker"]
        participation = item.get("participation", "unknown")
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                "full_participation": 0,
                "non_participation": 0,
                "pretend_participation": 0,
                "total_utterances": 0,
                "utterances": []
            }
        
        speaker_stats[speaker]["total_utterances"] += 1
        if participation in speaker_stats[speaker]:
            speaker_stats[speaker][participation] += 1
        speaker_stats[speaker]["utterances"].append({
            "text": item["text"][:200],
            "participation": participation
        })
    
    # Build output
    output = {
        "summary": speaker_stats,
        "detailed_results": all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PARTICIPATION BALANCE SUMMARY")
    print("=" * 60)
    
    for speaker, stats in speaker_stats.items():
        total = stats["total_utterances"]
        print(f"\n{speaker}:")
        print(f"  Total utterances: {total}")
        print(f"  Full participation: {stats['full_participation']} ({stats['full_participation']/total*100:.1f}%)")
        print(f"  Non-participation: {stats['non_participation']} ({stats['non_participation']/total*100:.1f}%)")
        print(f"  Pretend participation: {stats['pretend_participation']} ({stats['pretend_participation']/total*100:.1f}%)")
    
    return output


if __name__ == "__main__":
    transcript_file = "Editors_Extensions_20231207.json"
    output_file = "participation_balance-1.json"
    
    analyze_participation(transcript_file, output_file)