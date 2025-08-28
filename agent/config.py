import os

# Prefer env; you can hard-code, but avoid committing secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Controller guardrails
MAX_STEPS_DEFAULT = 1