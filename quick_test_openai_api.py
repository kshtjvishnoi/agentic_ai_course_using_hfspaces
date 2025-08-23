# tests/quick_test_openai_api.py
from __future__ import annotations
import os
import sys
import argparse
import traceback

# ---- Adjust this import to match where your code lives ----
# Your earlier messages showed: C:\...\agent\tools\media_tools.py
# So:
from agent.tools.media_tools import _llm_answer, asr_tool

# If your project root isn't already on sys.path, uncomment and fix relative path:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")

def test_llm(question: str, context: str, expect: str | None) -> bool:
    banner("[1/2] LLM smoke test via _llm_answer")
    print(f"Model       : {os.getenv('OPENAI_MODEL')}")
    print(f"Question    : {question!r}")
    print(f"Context     : {context!r}")
    try:
        out = _llm_answer(question, context)
    except Exception as e:
        print("EXCEPTION while calling _llm_answer():")
        traceback.print_exc()
        return False

    print(f"LLM Output  : {out!r}")
    if out.startswith("ERROR:"):
        print("❌ LLM returned an error. Check your API key/model.")
        return False

    if expect is not None:
        ok = out.strip() == expect.strip()
        print("Expected    :", expect)
        print("Match       :", "✅ YES" if ok else "❌ NO")
        return ok

    print("No 'expect' provided; treating any non-error output as success.")
    return True

def test_asr(audio_path: str, question: str | None, expect: str | None) -> bool:
    banner("[2/2] ASR test via asr_tool (Whisper)")
    print(f"Audio path  : {audio_path}")
    if not os.path.exists(audio_path):
        print("❌ File not found. Provide a valid audio file path.")
        return False

    # Minimal State: your tool accepts a dict-like State
    state = {}
    if question:
        state["question"] = question

    try:
        out = asr_tool(state, file_path=audio_path)
    except Exception as e:
        print("EXCEPTION while calling asr_tool():")
        traceback.print_exc()
        return False

    print(f"ASR Output  : {out!r}")
    if out.startswith("ERROR:"):
        print("❌ ASR returned an error. See message above.")
        return False

    if expect is not None:
        ok = out.strip() == expect.strip()
        print("Expected    :", expect)
        print("Match       :", "✅ YES" if ok else "❌ NO")
        return ok

    print("No 'expect' provided; treating any non-error output as success.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Quick test for your OpenAI LLM + ASR tools")
    parser.add_argument("--llm-question", default="What is the codename?", help="LLM test question")
    parser.add_argument("--llm-context",  default="The codename is AURORA.", help="LLM test context")
    parser.add_argument("--llm-expect",   default="AURORA", help="Exact expected LLM answer (set empty to skip check)")
    parser.add_argument("--audio",         default=None, help="Path to an audio file for ASR test (wav/mp3/m4a...)")
    parser.add_argument("--asr-question",  default="Which fruit is mentioned?", help="Question for ASR post-processing")
    parser.add_argument("--asr-expect",    default=None, help="Exact expected ASR+LLM answer (set empty to skip check)")
    args = parser.parse_args()

    # Env sanity
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is not set.")
        sys.exit(1)
    if not os.getenv("OPENAI_MODEL"):
        print("⚠️  OPENAI_MODEL not set; your code will still try, but please set one explicitly.")

    # LLM test
    llm_expect = args.llm_expect if args.llm_expect != "" else None
    ok1 = test_llm(args.llm_question, args.llm_context, llm_expect)

    # ASR test (optional)
    ok2 = True
    if args.audio:
        asr_expect = args.asr_expect if args.asr_expect != "" else None
        ok2 = test_asr(args.audio, args.asr_question, asr_expect)
    else:
        banner("[2/2] ASR test skipped (no --audio given)")

    print("\nSummary:", "✅ ALL TESTS PASSED" if (ok1 and ok2) else "❌ SOME TESTS FAILED")
    sys.exit(0 if (ok1 and ok2) else 2)

if __name__ == "__main__":
    main()
