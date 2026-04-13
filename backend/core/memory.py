import json
import os
import requests
from datetime import datetime
from typing import Optional

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations")
LLAMA_URL = "http://127.0.0.1:8001/v1/completions"  # ✅ FIXED
MODEL_NAME = "your-model-name"  # ⚠️ set this properly

EXTRACTION_PROMPT = """
You are a memory extraction assistant. Your only job is to extract personal facts about the user from their message.

Rules:
- Extract ONLY facts explicitly stated or clearly implied about the USER (not about documents).
- Facts can include: name, age, location, preferences, travel plans, language, profession, or any personal context.
- Return a flat JSON object with short snake_case keys and concise string values.
- If no personal facts are found, return an empty object: {}
- Return ONLY valid JSON. No explanation, no markdown, no extra text.

User message: "{message}"

JSON:"""


def _call_llm(prompt: str, timeout: int = 30) -> str:
    try:
        response = requests.post(
            LLAMA_URL,
            json={
                "model": MODEL_NAME,          # ✅ REQUIRED
                "prompt": prompt,
                "max_tokens": 200,            # ✅ FIXED (was n_predict)
                "temperature": 0.0,
                "stop": ["\n\n", "```", "User message:"],
            },
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["text"].strip()   # ✅ FIXED

    except Exception as e:
        print(f"[memory] LLM call failed: {e}")
        return "{}"


def _memory_path(session_id: str) -> str:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return os.path.join(MEMORY_DIR, f"memory_{session_id}.json")


def load_memory(session_id: str) -> dict:
    path = _memory_path(session_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("facts", {})
    except Exception:
        return {}


def save_memory(session_id: str, facts: dict):
    path = _memory_path(session_id)
    payload = {
        "session_id": session_id,
        "updated_at": datetime.now().isoformat(),
        "facts": facts,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def extract_user_facts(message: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(message=message.replace('"', "'"))
    raw = _call_llm(prompt)

    # Clean possible markdown wrapping
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        facts = json.loads(raw)
        if not isinstance(facts, dict):
            return {}
        return {k: v for k, v in facts.items() if v and str(v).strip()}
    except json.JSONDecodeError:
        print(f"[memory] Could not parse LLM output as JSON: {repr(raw)}")
        return {}


def update_memory(session_id: str, message: str) -> dict:
    existing = load_memory(session_id)
    new_facts = extract_user_facts(message)

    if new_facts:
        existing.update(new_facts)
        save_memory(session_id, existing)
        print(f"[memory] Updated memory for session {session_id[:8]}: {new_facts}")

    return existing


def format_memory_for_prompt(session_id: str) -> str:
    facts = load_memory(session_id)
    if not facts:
        return ""

    lines = ["Known facts about the user:"]
    for key, value in facts.items():
        label = key.replace("_", " ").capitalize()
        lines.append(f"  - {label}: {value}")

    return "\n".join(lines)


def clear_memory(session_id: str):
    path = _memory_path(session_id)
    if os.path.exists(path):
        os.remove(path)
    print(f"[memory] Cleared memory for session {session_id[:8]}")