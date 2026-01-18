import os
import json
import re
from typing import List, Dict

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

client = Groq(api_key=GROQ_API_KEY)

def split_into_sentences(text: str) -> List[str]:
    text = text.replace("\r", " ").replace("\n", " ").strip()
    parts = re.split(r'([.!?])\s+', text)
    sentences = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip() if i < len(parts) else ""
        punct = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if chunk:
            s = (chunk + punct).strip()
            if s:
                sentences.append(s)
    return [s for s in sentences if len(s) > 1]

def build_labeling_prompt(sentences: List[str]) -> str:
    schema_description = """
You are an expert meeting analyst. Classify each sentence into exactly ONE of:

1) task  (Problem-Oriented Statements)
   - Technical/content-related statements about the meeting topic.
   - Questions, sharing information/knowledge, identifying problems and solutions.
   - Often past-oriented, informing about things/tasks that have happened.
   - Aims/objectives or abstract goals/ideals.
   - Suggestions with no concrete action yet (can, maybe, hopefully).
   - Exclude meeting-structuring statements and concrete post-meeting actions.

2) procedural  (Procedural Statements (+))
   - Managing/structuring the meeting, goal orientation, time management.
   - Clarifying, concretizing, summarizing contributions.
   - Questions/suggestions about how to proceed in the meeting.
   - Handing over the floor, asking someone to continue, sharing screen.
   - Present-oriented.
   - Exclude post-meeting tasks.

3) social  (Socio-Emotional Statements (+))
   - Creating a positive atmosphere, social competence.
   - Praise, recognition, support.
   - Humor, jokes, expressing feelings.
   - Encouraging participation, reacting to others.
   - Exclude neutral fillers unrelated to previous comments.

4) action  (Action-Oriented Statements (+))
   - Advancing work after/outside the meeting.
   - Concrete future plans, steps, strategies.
   - Direct task assignments and commitments.
   - Showing interest in change and taking responsibility.
   - Future-oriented.
   - Exclude statements only about behavior inside the meeting and pure information/optional offers.

5) counter  (Counterproductive/Destructive Statements)
   - Negative procedural, socio-emotional, or action-oriented statements.
   - Harming productivity or working relationships.
   - Unnecessary detail without purpose.
   - Devaluing others, hostile remarks.
   - No interest in change/action/responsibility, unconstructive complaining.
   - Interruptions (not letting someone finish).

Output format:
Return ONLY a JSON list where each element is:
{"text": "<original sentence>", "label": "<one_of: task|procedural|social|action|counter>"}
"""

    few_shot_examples = """
Examples:

Input sentences:
1) "Next one is, remember, you can sign up to be a host on the conference section at the next GitLab Summit."
2) "But let me just share my screen here for a second."
3) "Yeah, ok, great, that's really helpful."
4) "But by the end of next week, I think we'll have a working version behind the feature flag, after that I'll work on the rest."
5) "Sorry Amy, before you go, I just want to say that your idea makes no sense at all."

Expected JSON:
[
  {"text": "Next one is, remember, you can sign up to be a host on the conference section at the next GitLab Summit.", "label": "task"},
  {"text": "But let me just share my screen here for a second.", "label": "procedural"},
  {"text": "Yeah, ok, great, that's really helpful.", "label": "social"},
  {"text": "But by the end of next week, I think we'll have a working version behind the feature flag, after that I'll work on the rest.", "label": "action"},
  {"text": "Sorry Amy, before you go, I just want to say that your idea makes no sense at all.", "label": "counter"}
]

Now classify the following sentences from a meeting transcript.
"""

    sentences_block = "\n".join(f'{i}) "{s}"' for i, s in enumerate(sentences, start=1))
    return schema_description + "\n" + few_shot_examples + "\nInput sentences:\n" + sentences_block

def extract_json_block(text: str) -> str:
    """
    Extract the first top-level JSON array from the model output.
    Works when the model returns extra text around the JSON.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in model output.")
    return text[start:end + 1]

def label_sentences(sentences: List[str]) -> List[Dict[str, str]]:
    if not sentences:
        return []

    prompt = build_labeling_prompt(sentences)

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
        temperature=0.0,
    )

    raw = chat_completion.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
    
        try:
            json_block = extract_json_block(raw)
            parsed = json.loads(json_block)
        except Exception:
            print("Could not extract JSON from model output. Raw output:")
            print(raw)
            return []

    result: List[Dict[str, str]] = []
    for item in parsed:
        text_val = item.get("text", "").strip()
        label_val = item.get("label", "").strip().lower()
        if label_val not in {"task", "procedural", "social", "action", "counter"}:
            label_val = "task"
        result.append({"text": text_val, "label": label_val})
    return result


def main():
    transcript_path = "Transcript_text.txt"
    output_path = "labeled_transcript.json"

    with open(transcript_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sentences = split_into_sentences(raw_text)
    labeled = label_sentences(sentences)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    print(f"Saved labeled data to {output_path}")

if __name__ == "__main__":
    main()
