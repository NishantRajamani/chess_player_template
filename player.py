import re
import random
import chess
import torch
from typing import Optional, List

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from chess_tournament.players import Player

# values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


class TransformerPlayer(Player):

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "TransformerPlayer",
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        candidate_count: int = 6,
        max_new_tokens: int = 32,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.candidate_count = candidate_count
        self.max_new_tokens = max_new_tokens

        # lazy components
        self.tokenizer = None
        self.model = None

        # move history to prevent loops/bouncing (stores UCI strings)
        self.move_history: List[str] = []


    def _load_model(self):
        if self.model is not None:
            return

        print(f"[{self.name}] Loading model {self.model_id} (attempting 4-bit bnb) ...")

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Trying to use 4-bit quantization with bitsandbytes
        use_bnb = True
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            bnb_config = None
            use_bnb = False

        try:
            if use_bnb and bnb_config is not None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
        except Exception as e:
            try:
                print(f"[{self.name}] bnb/device_map load failed ({e}), falling back to plain load...")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            except Exception as e2:
                print(f"[{self.name}] model load failed: {e2}")
                raise

        self.model.eval()

    # Heuristic scoring
    def _heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0.0

        # capture
        if board.is_capture(move):
            cap = board.piece_at(move.to_square)
            if cap:
                score += PIECE_VALUES.get(cap.piece_type, 0)

        if move.promotion:
            score += PIECE_VALUES.get(chess.QUEEN, 900) + 300

        board.push(move)
        try:
            mobility = len(list(board.legal_moves))
            score += mobility * 1.0

            to_sq = move.to_square
            file = chess.square_file(to_sq)
            rank = chess.square_rank(to_sq)
            center_dist = abs(3.5 - file) + abs(3.5 - rank)
            score += (8 - center_dist) * 4.0

            moved_piece = board.piece_at(to_sq)
            if moved_piece:
                if board.is_attacked_by(board.turn, to_sq):
                    score -= PIECE_VALUES.get(moved_piece.piece_type, 0) * 0.8

                attackers = len(board.attackers(board.turn, to_sq))
                defenders = len(board.attackers(not board.turn, to_sq))
                if attackers > defenders:
                    score -= (attackers - defenders) * PIECE_VALUES.get(moved_piece.piece_type, 0) * 0.25
        finally:
            board.pop()

        score += (random.random() - 0.5) * 1e-6
        return score

    # Prompt builder (choose_uci) with move history included
    def _build_choose_uci_prompt(self, fen: str, candidates: list[str], recent_moves: list[str]) -> str:
        history_str = " ".join(recent_moves[-10:]) if recent_moves else "none"
        moves_block = "\n".join(f"- {m}" for m in candidates)
    
        prompt = f"""You are a grandmaster-level chess engine. Analyze the position carefully.
    FEN: {fen}
    Recent moves played: {history_str}
    
    CANDIDATE MOVES (you must pick from these only):
    {moves_block}
    
    CHESS PRINCIPLES TO FOLLOW:
    1. In the opening: develop knights and bishops before moving the queen
    2. Castle early to protect your king - never leave king in center if castling is available
    3. NEVER move to a square where your piece can be captured for free
    4. If you can capture a higher-value piece safely, do it
    5. AVOID repeating the same moves over and over again - recent moves were: {history_str}
    6. Control the center with pawns and other pieces cause controlling the center wins the game
    7. Connect your rooks after castling
    8. Do not move the same piece twice in the opening
    
    Think step by step:
    1. Is my king safe? Can I castle?
    2. Am I hanging any pieces?
    3. Can I win material?
    4. Which move improves my position most?
    
    Reply with ONLY one UCI move from the candidate list. Nothing else."""
        
        return prompt

    # fallback
    def _deterministic_fallback(self, candidates_uci: List[str], board: chess.Board) -> str:
        for u in candidates_uci:
            mv = chess.Move.from_uci(u)
            if not self._candidate_allows_opponent_mate_in_one(board, mv):
                return u
        return candidates_uci[0] if candidates_uci else None

    def _candidate_allows_opponent_mate_in_one(self, board: chess.Board, move: chess.Move) -> bool:
        board.push(move)
        try:
            for reply in board.legal_moves:
                board.push(reply)
                is_mate = board.is_checkmate()
                board.pop()
                if is_mate:
                    return True
            return False
        finally:
            board.pop()

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        for mv in legal_moves:
            board.push(mv)
            if board.is_checkmate():
                board.pop()
                uci = mv.uci()
                self.move_history.append(uci)
                return uci
            board.pop()

        scored = [(mv, self._heuristic_score(board, mv)) for mv in legal_moves]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_moves = [mv for mv, _ in scored[: min(self.candidate_count, len(scored))]]
        candidates_uci = [m.uci() for m in top_moves]

        if len(self.move_history) >= 1:
            last_move = self.move_history[-1]
            if len(last_move) >= 4:
                reversed_last = last_move[2:4] + last_move[0:2]
                filtered = [m for m in candidates_uci if m != reversed_last]
                if filtered:
                    candidates_uci = filtered

        non_bad = []
        for u in list(candidates_uci):
            mv = chess.Move.from_uci(u)
            if not self._candidate_allows_opponent_mate_in_one(board, mv):
                non_bad.append(u)
        if non_bad:
            candidates_uci = non_bad

        recent = self.move_history[-10:]

        prompt = self._build_choose_uci_prompt(fen, candidates_uci, recent)

        try:
            self._load_model()
            model_device = next(self.model.parameters()).device

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(model_device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            raw = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            match = self.UCI_REGEX.search(raw)
            chosen = None
            if match:
                cand = match.group(1).lower().replace("=", "")
                if cand in candidates_uci:
                    chosen = cand

            if not chosen:
                s = raw.strip()
                if s in candidates_uci:
                    chosen = s

            if not chosen:
                chosen = self._deterministic_fallback(candidates_uci, board)

        except Exception:
            chosen = self._deterministic_fallback(candidates_uci, board) or (random.choice(legal_moves).uci())

        if chosen:
            self.move_history.append(chosen)
            if len(self.move_history) > 200:
                self.move_history = self.move_history[-200:]

        return chosen
