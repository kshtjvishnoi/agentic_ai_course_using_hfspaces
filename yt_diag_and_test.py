# yt_transcript_probe.py
from __future__ import annotations
import argparse, re, sys, inspect
from pathlib import Path

def extract_vid(s: str) -> str:
    s = (s or "").strip()
    if re.fullmatch(r"[-_a-zA-Z0-9]{11}", s):
        return s
    m = re.search(r"(?:[?&]v=|/shorts/|youtu\.be/)([-_a-zA-Z0-9]{11})", s)
    return m.group(1) if m else ""

def main():
    ap = argparse.ArgumentParser(description="Probe YouTube transcripts via youtube-transcript-api and your tool.")
    ap.add_argument("--url", help="YouTube URL")
    ap.add_argument("--id", dest="video_id", help="YouTube video ID (11 chars)")
    ap.add_argument("--lang", default="en", help="Preferred language (default: en)")
    ap.add_argument("--cookies", help="Path to cookies.txt exported for youtube.com (optional)")
    ap.add_argument("--use-list", action="store_true", help="Also try list_transcripts fallback")
    ap.add_argument("-q", "--question", default="Summarize in one sentence.")
    args = ap.parse_args()

    try:
        import youtube_transcript_api as yta
        from youtube_transcript_api import YouTubeTranscriptApi
        print("== youtube_transcript_api ==")
        print("file     :", getattr(yta, "__file__", "<no __file__>"))
        print("version  :", getattr(yta, "__version__", "<no __version__>"))
        print("has get  :", hasattr(YouTubeTranscriptApi, "get_transcript"))
        print("has list :", hasattr(YouTubeTranscriptApi, "list_transcripts"))
    except Exception as e:
        print("FAILED import youtube_transcript_api:", e); sys.exit(1)

    vid = extract_vid(args.url or args.video_id or "")
    if not vid:
        print("ERROR: Provide --url or --id"); sys.exit(2)
    print("\n== Direct get_transcript ==")
    try:
        segs = YouTubeTranscriptApi.get_transcript(vid, languages=[args.lang, "en"], cookies=args.cookies)
        print(f"OK: {len(segs)} segments via get_transcript")
        print("Sample:", " ".join(s.get("text","") for s in segs[:3])[:200].replace("\n"," "))
    except Exception as e:
        print("get_transcript FAILED:", e)

    if args.use_list and hasattr(YouTubeTranscriptApi, "list_transcripts"):
        print("\n== list_transcripts fallback ==")
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(vid, cookies=args.cookies)
            chosen = None
            # prefer requested lang
            for t in transcripts:
                if getattr(t, "language_code", "")[:2] == args.lang[:2]:
                    chosen = t; break
            # then English
            if chosen is None:
                for t in transcripts:
                    if getattr(t, "language_code", "")[:2] == "en":
                        chosen = t; break
            # else try translating first available to English
            if chosen is None:
                try:
                    first = next(iter(transcripts))
                    chosen = first.translate("en")
                except Exception:
                    pass
            if chosen is None:
                print("No suitable transcript found via list_transcripts")
            else:
                segs2 = chosen.fetch()
                print(f"OK: {len(segs2)} segments via list_transcripts ({getattr(chosen,'language_code', 'unknown')})")
                print("Sample:", " ".join(s.get("text","") for s in segs2[:3])[:200].replace("\n"," "))
        except Exception as e:
            print("list_transcripts FAILED:", e)

    # Now invoke your tool
    print("\n== Your yt_transcript_tool ==")
    try:
        # Ensure we import your package
        ROOT = Path(__file__).resolve().parent
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from agent.tools.media_tools import yt_transcript_tool
        print("media_tools.py path:", inspect.getsourcefile(yt_transcript_tool))
        out = yt_transcript_tool({"question": args.question}, url=args.url, video_id=args.video_id, language=args.lang)
        print("\n--- TOOL RESULT ---\n", out)
    except Exception as e:
        print("yt_transcript_tool FAILED:", e)

if __name__ == "__main__":
    main()
