"""What the platform runs, and which builds of it are known to be broken.

The simulated platform never decides that a fault "is fixed". Each replica
runs some artifacts (a model checkpoint, a chat template, a quantization, a
tokenizer, a server build) and each artifact either has a defect or does not.
What a defect does to answers is set here, from what is known about real
serving faults:

- a wrong or regressed checkpoint loses domain knowledge first;
- a chat template mismatch breaks structured output almost entirely;
- aggressive quantization breaks arithmetic, and hurts non-Latin scripts such
  as Arabic more than English (Cohere, "How does quantization affect
  multilingual LLMs?", arXiv 2407.03211);
- a tokenizer that does not match the checkpoint garbles Arabic and leaves
  English mostly intact;
- silent data corruption on a GPU corrupts answers of every kind.
"""

# The release each replica starts on: the known-good configuration.
GOOD = {
    "model_version": "allam-34b-v2.3",
    "template": "chat-tpl-12",
    "quantization": "fp8",
    "tokenizer": "tok-2.1",
    "server_build": "serve-1.7.2",
    "memory_limit_gb": 80,
}

# Earlier releases, so the change log has a believable history to roll back
# through. All of them were good in their day.
HISTORY = {
    "model_version": ["allam-34b-v2.1", "allam-34b-v2.2"],
    "template": ["chat-tpl-10", "chat-tpl-11"],
    "quantization": ["bf16"],
    "tokenizer": ["tok-2.0"],
    "server_build": ["serve-1.7.0", "serve-1.7.1"],
    "memory_limit_gb": [80],
}

# Bad builds a fault can deploy, and the defect each one carries.
BAD = {
    "model_version": ("allam-34b-v2.4-rc", "domain_regression"),
    "template": ("chat-tpl-13", "format_break"),
    "quantization": ("int4-awq-unvalidated", "quantization_damage"),
    "tokenizer": ("tok-2.2-beta", "tokenizer_mismatch"),
    "server_build": ("serve-1.8.0", "memory_leak"),
    "memory_limit_gb": (40, "memory_limit_too_low"),
}

DEFECT_OF = {(key, value): defect for key, (value, defect) in BAD.items()}

# Chance an answer goes wrong, by defect, language and question category.
# "*" applies to any language or category not named.
WRONG_RATE = {
    "domain_regression": {"*": {"fact": 0.7, "retrieval": 0.7, "math": 0.15, "*": 0.05}},
    "format_break": {"*": {"format": 1.0, "*": 0.25}},
    "quantization_damage": {"en": {"math": 0.6, "*": 0.05},
                            "ar": {"math": 0.85, "*": 0.35}},
    "tokenizer_mismatch": {"ar": {"*": 0.9}, "en": {"*": 0.03}},
    "silent_data_corruption": {"*": {"*": 0.6}},
    "index_corruption": {"*": {"retrieval": 0.9, "*": 0.0}},
}

# A healthy model is not perfect. Canaries are chosen as questions it gets
# right, but now and then it still slips; thresholds must allow for this.
BASELINE_WRONG = 0.01

# The other side of the platform: gateway and workers.
GATEWAY_GOOD = {"timeout_ms": 30000, "log_level": "info"}
GATEWAY_BAD = {"timeout_ms": 800}
WORKERS_GOOD = {"max_concurrency": 64}
WORKERS_BAD = {"max_concurrency": 4}


def rate(defect, lang, category):
    """Chance this defect spoils an answer. Defects that crash or slow the
    server, rather than change what it says, spoil none."""
    table = WRONG_RATE.get(defect, {})
    by_lang = table.get(lang, table.get("*", {}))
    return by_lang.get(category, by_lang.get("*", 0.0))
