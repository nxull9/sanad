"""Language model calls, through OpenRouter, with every reply kept on disk.

A reply is stored under a hash of exactly what was asked (model, messages,
settings). Asking the same thing again reads the stored reply and costs
nothing, so every result in this repository can be rebuilt for free: set
SANAD_OFFLINE=1 and a missing reply is an error rather than a paid call.

Spending is capped in code. A `Budget` is shared by every call in a run, and
a call that would start after the cap is reached raises `BudgetExceeded`
instead. The agent treats that like any other limit: it stops and hands the
incident to a person.
"""

import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAYS = ROOT / "replays"
URL = "https://openrouter.ai/api/v1/chat/completions"

# Price per million tokens (input, output), from OpenRouter on 30 October 2026.
# Used only when a reply does not report its own cost.
PRICES = {
    "qwen/qwen3.7-flash": (0.03, 0.13),
    "deepseek/deepseek-v4-flash": (0.091, 0.182),
}


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, max_usd):
        self.max_usd = max_usd
        self.spent = 0.0      # paid calls only
        self.calls = 0        # every call, stored or paid
        self.paid_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost = 0.0       # what the calls cost when first made, stored or not

    def check(self):
        if self.spent >= self.max_usd:
            raise BudgetExceeded(f"spent ${self.spent:.4f} of ${self.max_usd:.4f}")

    def record(self, usage, cost, paid):
        self.calls += 1
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)
        self.cost += cost
        if paid:
            self.paid_calls += 1
            self.spent += cost


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
    return key


def _cost(model, usage):
    if "cost" in usage:
        return float(usage["cost"])
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out) / 1e6


def _tls():
    """Python from python.org on macOS ships without root certificates;
    the system's own bundle serves instead."""
    context = ssl.create_default_context()
    if not context.get_ca_certs() and Path("/etc/ssl/cert.pem").exists():
        context.load_verify_locations("/etc/ssl/cert.pem")
    return context


def _post(body, key, attempts=6):
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "X-Title": "Sanad"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120, context=_tls()) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError) as error:
            status = getattr(error, "code", None)
            if attempt == attempts - 1 or (status is not None and status < 500 and status != 429):
                raise
            time.sleep(min(30, 2 ** (attempt + 1)))     # rides out a short network drop


def chat(model, messages, budget, max_tokens=1500, reasoning=None):
    """The model's reply, parsed as JSON, and whether it came from disk.
    Hidden reasoning is off unless asked for: it is billed as output and,
    on a small model, can use up the whole reply before any answer comes."""
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "reasoning": {"effort": reasoning, "exclude": True} if reasoning else {"enabled": False},
            "usage": {"include": True}}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = REPLAYS / model.replace("/", "__") / digest[:2] / f"{digest}.json"
    if path.exists():
        stored = json.loads(path.read_text())
        budget.record(stored["usage"], stored["cost"], paid=False)
        return stored["reply"], True
    if os.environ.get("SANAD_OFFLINE"):
        raise RuntimeError(f"no stored reply for {path.name} and SANAD_OFFLINE is set")
    budget.check()
    key = api_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env)")
    response = _post(body, key)
    if "choices" not in response:
        raise RuntimeError(f"OpenRouter: {response.get('error', response)}")
    text = response["choices"][0]["message"].get("content") or ""
    usage = response.get("usage", {})
    cost = _cost(model, usage)
    budget.record(usage, cost, paid=True)
    reply = parse(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "request": body, "text": text, "reply": reply,
                                "usage": usage, "cost": cost}, ensure_ascii=False, indent=1))
    return reply, False


def parse(text):
    """JSON from a reply, forgiving a code fence or a sentence around it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                pass
    return {"_unparsed": text[:500]}
