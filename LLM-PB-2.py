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

PARTICIPATION_PROMPT = """You are an expert in meeting participation analysis based on Beck et al. (2024).
Classify each speaker's utterance into ONE of three participation types.

## Participation Types:

### full_participation
Active, substantive contribution that advances the meeting discussion:
- Sharing NEW information, ideas, technical details, or solutions
- Asking meaningful questions that advance discussion
- Providing detailed explanations or context
- Taking ownership of tasks or making commitments
- Facilitating the meeting (opening, closing, transitioning topics, asking for input)
- Summarizing discussions with added insight

Examples:
- "I've been working on the cache implementation and found we need to update the schema"
- "I'll reach out to you for a sync call to discuss this further"
- "Any other topics that people want to bring up and discuss?"
- "So I think we should automate the merge requests using renovate"

### non_participation
Minimal engagement with NO substantive content - ONLY very short fillers:
- Single word acknowledgments: "Yeah.", "Okay.", "Right.", "Hmm."
- Very short phrases: "All right.", "Sounds good.", "Cool."
- Must be 5 words or fewer with no real content

Examples:
- "Yeah."
- "Okay."
- "All right."
- "Sounds good."

### pretend_participation
Surface-level engagement that APPEARS participatory but lacks substance (Beck et al. 2024):
- Commenting on what others said WITHOUT adding new information
- Simply agreeing with others ("Yeah, that makes sense", "I agree with that")
- Repeating what others said
- Vague responses without specificity ("Yeah, there's some activity in the doc")
- Off-topic comments
- Asking questions already answered
- Deferring ("I'll look into that later")
- Combining minimal acknowledgment with vague observation

Examples:
- "Yeah, that sounds good" (agreeing without adding)
- "Right, like what Aaron said" (repeating)
- "Yeah, there's some more activity in the doc now" (vague observation)
- "I think so too" (agreeing without substance)
- "Yeah. Sounds good. All right. Awesome." (multiple fillers, no content)

## Classification Rules:
1. If utterance is 5 words or fewer AND only contains fillers → non_participation
2. If utterance shares NEW information, asks substantive questions, or facilitates meeting → full_participation
3. If utterance only agrees, repeats others, or makes vague comments → pretend_participation

## Task:
Classify each utterance. Return ONLY a JSON array with {"speaker": "...", "text": "...", "participation": "..."}.

Utterances:
"""


def analyze_participation(transcript_path: str, output_path: str):
    """Analyze participation balance from meeting transcript."""
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    print(f"Total utterances: {len(transcript)}")
    
    batch_size = 8
    all_results = []
    
    for i in range(0, len(transcript), batch_size):
        batch = transcript[i:i + batch_size]
        print(f"Processing utterances {i+1}-{min(i+batch_size, len(transcript))}...")
        
        utterances_text = "\n".join([
            f"{idx+1}. [{item['speaker']}]: \"{item['text']}\"" 
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
        if participation in ["full_participation", "non_participation", "pretend_participation"]:
            speaker_stats[speaker][participation] += 1
        speaker_stats[speaker]["utterances"].append({
            "text": item["text"][:200],
            "participation": participation
        })
    
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
    
    for speaker, stats in sorted(speaker_stats.items()):
        total = stats["total_utterances"]
        full = stats["full_participation"]
        non = stats["non_participation"]
        pretend = stats["pretend_participation"]
        
        print(f"\n{speaker}:")
        print(f"  Total utterances: {total}")
        print(f"  Full participation: {full} ({full/total*100:.1f}%)")
        print(f"  Non-participation: {non} ({non/total*100:.1f}%)")
        print(f"  Pretend participation: {pretend} ({pretend/total*100:.1f}%)")
    
    return output


if __name__ == "__main__":
    transcript_file = "Editors_Extensions_20231207.json"
    output_file = "participation_balance-2.json"
    
    analyze_participation(transcript_file, output_file)