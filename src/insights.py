"""Optional: a LOCAL LLM (Ollama) drafts short insights from verified quotes only.

Every claim must cite evidence numbers and pass hard checks, or it is rejected.
Nothing here is final: a researcher reviews it in the dashboard.
Setup: install Ollama, then  ollama pull llama3.2:3b"""
import json
import re
import urllib.request

LLM_MODEL = "llama3.2:3b"
URL = "http://localhost:11434/api/generate"
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "seventeen", "eighteen", "twenty", "hundred"}

PROMPT = """You are helping a researcher. Below is numbered evidence from interviews with care-experienced young adults (synthetic data).
Write 2 or 3 short claims about this theme.

Rules:
- Use ONLY the evidence below. Do not add causes, advice, numbers or facts that are not in it.
- Every claim must cite the evidence numbers that support it, for example ["E1","E3"].
- If the evidence contains contrasting experiences, say so in one claim.
- Do not mention participant IDs or counts. The system adds those.
- If the evidence is too thin, return an empty list.

Evidence:
{evidence}

Return ONLY JSON in this form:
{{"claims":[{{"text":"...","evidence":["E1"]}}]}}"""


def ask_llm(prompt, model=LLM_MODEL):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(json.loads(r.read())["response"])


def _words(t):
    return set(re.findall(r"[a-z]+", t.lower()))


def validate(claims, ev_map):
    """Return (accepted, error). Any broken rule rejects the whole answer."""
    if not isinstance(claims, list) or not claims:
        return None, "model returned no claims"
    accepted = []
    for c in claims:
        text = c.get("text", "") if isinstance(c, dict) else ""
        ids = c.get("evidence", []) if isinstance(c, dict) else []
        if not text or len(text) > 300:
            return None, "claim missing or too long"
        if not ids or any(i not in ev_map for i in ids):
            return None, f"claim cites unknown evidence: {ids}"
        if re.search(r"\bP\d{2,3}\b", text):
            return None, "claim mentions a participant ID"
        cited = " ".join(ev_map[i]["text"] for i in ids).lower()
        if set(re.findall(r"\d+", text)) - set(re.findall(r"\d+", cited)):
            return None, "claim contains a number that is not in its cited evidence"
        if (_words(text) & NUMBER_WORDS) - _words(cited):
            return None, "claim contains a number word that is not in its cited evidence"
        accepted.append({"text": text, "evidence": [{"participant": ev_map[i]["participant"], "line": ev_map[i]["line"],
                                                      "text": ev_map[i]["text"]} for i in ids]})
    return accepted, None


def insight_for(quotes, model=LLM_MODEL):
    """quotes: list of dicts with participant, line, text (already verified). Two attempts, then give up."""
    quotes = quotes[:8]
    ev_map = {f"E{i + 1}": q for i, q in enumerate(quotes)}
    evidence = "\n".join(f'E{i + 1}: "{q["text"]}"' for i, q in enumerate(quotes))
    err = "not attempted"
    for _ in range(2):
        try:
            out = ask_llm(PROMPT.format(evidence=evidence), model)
        except Exception as e:
            return {"claims": [], "error": f"Could not reach the local model ({type(e).__name__}). Is Ollama running?"}
        claims, err = validate(out.get("claims"), ev_map)
        if claims:
            return {"claims": claims, "error": None, "model": model}
    return {"claims": [], "error": f"Rejected by checks: {err}"}


if __name__ == "__main__":   # offline self-test of the checks, no model needed
    ev = {"E1": {"participant": "P001", "line": 3, "text": "I waited almost a year for counselling."}}
    print("good  :", validate([{"text": "Some people waited a long time for counselling.", "evidence": ["E1"]}], ev)[1])
    print("number:", validate([{"text": "People waited 12 months.", "evidence": ["E1"]}], ev)[1])
    print("badref:", validate([{"text": "Waiting was long.", "evidence": ["E9"]}], ev)[1])
    print("ID    :", validate([{"text": "P001 waited a long time.", "evidence": ["E1"]}], ev)[1])
