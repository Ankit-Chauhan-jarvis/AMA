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

ACT4TEAMS_PROMPT = """You are an expert in meeting analysis using the act4teams framework.
Label each sentence with ONE of these five categories:

## Labels:

### task (Problem-Oriented)
- Technical/content-related statements about meeting topic
- Sharing information/knowledge, identifying problems and solutions
- Past-oriented, informing about things that happened
- Abstract goals or suggestions (can, maybe, hopefully)
Examples:
- "So you can make multiple ones there, so that should not be a problem"
- "There's a lot of structure. Actually, I put a few links here"

### procedural (Procedural +)
- Managing/structuring the meeting
- Time management, clarification, summarization
- Handing over floor, asking someone to continue
- Present-oriented
Examples:
- "Aaron, do you want to verbalize?"
- "Okay, we're live. Hello, everyone. Welcome to the meeting."
- "But let me just share my screen here"

### social (Socio-Emotional +)
- Creating positive atmosphere
- Praise, recognition, support
- Humor, expressing feelings
- Acknowledgments: Yeah, ok, great, alright
Examples:
- "Nice. Awesome."
- "I love that."
- "Very happy about this."

### action (Action-Oriented +)
- Advancing work after/outside meeting
- Concrete action plans, task assignments
- Taking responsibility, commitments
- Future-oriented
Examples:
- "Yeah, I'll do that."
- "I'll reach out to you for a sync call"
- "So make sure to read it."

### counter (Counterproductive)
- Harms productivity or relationships
- Interruptions, devaluing others
- Getting lost in unnecessary details
Examples:
- "Sorry Amy, Before you go, I just want to..." (Interruption)

## Task:
Label each sentence below. Return ONLY a JSON array with {"text": "...", "label": "..."} for each.

Sentences:
"""


def split_into_sentences(text: str) -> list[str]:
    """Split transcript into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]


def label_batch(sentences: list[str]) -> list[dict]:
    """Label a batch of sentences."""
    numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
    prompt = ACT4TEAMS_PROMPT + numbered + "\n\nJSON Output:"

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
        
        return json.loads(result)
        
    except Exception as e:
        print(f"Error: {e}")
        return [{"text": s, "label": "unknown"} for s in sentences]


def label_transcript(transcript_path: str, output_path: str, batch_size: int = 15):
    """Main function to label transcript."""
    with open(transcript_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sentences = split_into_sentences(text)
    print(f"Total sentences: {len(sentences)}")
    
    all_labeled = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        print(f"Processing {i+1}-{min(i+batch_size, len(sentences))}...")
        labeled = label_batch(batch)
        all_labeled.extend(labeled)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_labeled, f, indent=2, ensure_ascii=False)
    
    print(f"Output saved to: {output_path}")
    return all_labeled


if __name__ == "__main__":
    transcript_file = "Transcript_text.txt"
    output_file = "labeled_output.json"
    
    label_transcript(transcript_file, output_file)