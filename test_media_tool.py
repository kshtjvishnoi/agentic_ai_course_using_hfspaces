# test_media_tools.py
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

# --- Make sure we can import your package (expects repo root /agent/...) ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import your tools
try:
    from agent.tools.media_tools import yt_transcript_tool, asr_tool
except Exception as e:
    print(f"ERROR: Could not import tools from agent.tools.media_tools: {e}")
    sys.exit(1)

# Optionally override config from environment, if present
try:
    from agent import config as agent_config
    if os.getenv("OPENAI_API_KEY"):
        agent_config.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    if os.getenv("OPENAI_MODEL"):
        agent_config.OPENAI_MODEL = os.environ["OPENAI_MODEL"]
except Exception:
    # If your project doesn't expose agent.config, ignore
    pass


def run_yt(question: str, url: str | None, vid: str | None, lang: str):
    state = {"question": question}
    res = yt_transcript_tool(state, url=url, video_id=vid, language=lang)
    print("\n=== yt_transcript_tool RESULT ===\n")
    print(res)


def run_asr(question: str | None, file_path: str):
    state = {"question": question} if question else {}
    res = asr_tool(state, file_path=file_path)
    print("\n=== asr_tool RESULT ===\n")
    print(res)


def main():
    p = argparse.ArgumentParser(
        description="Quick tester for yt_transcript_tool and asr_tool."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # YouTube transcript tester
    p_yt = sub.add_parser("yt", help="Test yt_transcript_tool")
    p_yt.add_argument("-q", "--question", required=True, help="Question to answer from the transcript")
    m = p_yt.add_mutually_exclusive_group(required=True)
    m.add_argument("--url", help="YouTube URL (e.g., https://youtu.be/XXXXXXXXXXX)")
    m.add_argument("--id", dest="video_id", help="YouTube video ID (11 chars)")
    p_yt.add_argument("--lang", default="en", help="Preferred transcript language (default: en)")

    # ASR tester
    p_asr = sub.add_parser("asr", help="Test asr_tool (Whisper)")
    p_asr.add_argument("-q", "--question", help="Question with formatting rules (optional; returns transcript if omitted)")
    p_asr.add_argument("--file", required=True, dest="file_path", help="Path to audio file (mp3/wav/m4a/mp4/…)")

    args = p.parse_args()

    if args.cmd == "yt":
        run_yt(args.question, args.url, getattr(args, "video_id", None), args.lang)
    elif args.cmd == "asr":
        run_asr(args.question, args.file_path)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
