from __future__ import annotations
import base64, os, re, shutil
from typing import Optional
from openai import OpenAI

import chess
import chess.engine

from ..registry import tool
from ..state import State
from ..config import OPENAI_API_KEY, OPENAI_MODEL


# ----------------- helpers -----------------

def _b64_image_data_uri(path: str) -> str:
    ext = (os.path.splitext(path)[1] or "").lower().lstrip(".") or "png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"

def _normalize_fen(fen: str) -> str:
    """Ensure FEN has 6 fields; fill unknowns if missing."""
    fen = fen.strip().replace("\n", " ").replace("\t", " ")
    fen = re.sub(r"\s+", " ", fen)
    parts = fen.split()
    if len(parts) < 2:
        # maybe only placement given; assume 'w'
        parts += ["w"]
    if len(parts) < 3:
        parts += ["-"]        # castling
    if len(parts) < 4:
        parts += ["-"]        # en passant
    if len(parts) < 5:
        parts += ["0"]        # halfmove clock
    if len(parts) < 6:
        parts += ["1"]        # fullmove number
    return " ".join(parts[:6])

def _validate_fen(fen: str) -> Optional[str]:
    try:
        chess.Board(fen)  # will raise on invalid
        return None
    except Exception as e:
        return str(e)

def _resolve_stockfish_path() -> Optional[str]:
    """Best-effort discovery of a Stockfish UCI engine binary."""
    # 1) Explicit env var
    p = os.getenv("STOCKFISH_PATH")
    if p and os.path.exists(p):
        return p
    # 2) PATH lookup
    for name in ("stockfish", "stockfish.exe"):
        found = shutil.which(name)
        if found:
            return found
    # 3) Nothing found
    return None

def _uci_to_san(fen: str, uci: str) -> str:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    return board.san(move)


# ----------------- tools -----------------

@tool("chess_from_image")
def chess_from_image_tool(state: State, file_path: str | None = None, **kwargs) -> str:
    """
    Read a chessboard position from an image and return a FULL FEN (6 fields).
    Uses OpenAI Vision. If the question specifies who's to move, use that side.
    Returns FEN only on success, or "ERROR: ..." on failure.
    """
    if not OPENAI_API_KEY:
        return "ERROR: OPENAI_API_KEY missing."

    path = file_path or state.get("file_name") or ""
    if not path or not os.path.exists(path):
        return "ERROR: image file not found."

    # Encode image for the API
    data_uri = _b64_image_data_uri(path)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Hint side-to-move from the question if present
    qtext = (state.get("question") or "").lower()
    side_hint = None
    if "black" in qtext and "turn" in qtext:
        side_hint = "b"
    elif "white" in qtext and "turn" in qtext:
        side_hint = "w"

    sys = (
        "You are a chessboard OCR expert. You will be shown an image of a chess position.\n"
        "Output the exact FEN **with all 6 fields** in a single line:\n"
        "  <pieces> <side> <castling> <en-passant> <halfmove> <fullmove>\n"
        "Rules:\n"
        "- Use 'w' or 'b' for side to move. If the user specifies the side in the question, obey it.\n"
        "- If castling rights are unclear, output '-'. If no en passant target, output '-'.\n"
        "- If halfmove/fullmove are unknown, use '0' and '1' respectively.\n"
        "- Return ONLY the FEN string, no extra text.\n"
    )
    usr = [
        {"type": "text", "text": "Extract the FEN from this image. If the prompt mentions whose move, use that for the side."},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    if side_hint:
        usr.insert(0, {"type": "text", "text": f"Side to move (from prompt): {side_hint}."})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
        )
    except Exception as e:
        return f"ERROR: vision request failed: {e}"

    fen_raw = (resp.choices[0].message.content or "").strip()
    fen_raw = fen_raw.splitlines()[0].strip().strip("`").strip()
    fen = _normalize_fen(fen_raw)

    err = _validate_fen(fen)
    if err:
        return f"ERROR: invalid FEN: {fen} ({err})"

    return fen


@tool("chess_engine")
def chess_engine_tool(state: State, fen: str | None = None, depth: int = 16, **kwargs) -> str:
    """
    Given a FEN, compute the best move (for side to move) using Stockfish and return it in SAN.
    Tries a local UCI engine first; if not found, falls back to the `stockfish` Python wrapper if installed.
    Returns SAN on success, or "ERROR: ..." on failure.
    """
    fen = fen or (state.get("question") or "").strip()
    if not fen:
        return "ERROR: no FEN provided."

    # Validate FEN before launching engine
    try:
        board = chess.Board(fen)
    except Exception as e:
        return f"ERROR: invalid FEN: {e}"

    # 1) Try external Stockfish via python-chess UCI
    engine_path = _resolve_stockfish_path()
    if engine_path:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            try:
                # Either play() or analyse(); play is simpler and returns a best move
                result = engine.play(board, chess.engine.Limit(depth=int(depth)))
                san = board.san(result.move)
                return san
            finally:
                engine.quit()
        except Exception as e:
            # fall through to wrapper
            pass

    # 2) Try the `stockfish` pip wrapper (if available)
    try:
        from stockfish import Stockfish  # pip install stockfish
        sf = Stockfish(parameters={"Threads": 2, "Minimum Thinking Time": 30})
        # Try to set a sensible skill/strength (optional; ignores if not supported)
        try:
            sf.update_engine_parameters({"Skill Level": 20})
        except Exception:
            pass
        sf.set_fen_position(fen)
        uci = sf.get_best_move()
        if not uci:
            return "ERROR: could not obtain best move from Stockfish wrapper."
        return _uci_to_san(fen, uci)
    except Exception as e:
        return "ERROR: Stockfish engine not available. Set STOCKFISH_PATH or install `stockfish`. "
