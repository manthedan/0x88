from __future__ import annotations

FILES = "abcdefgh"
PIECES = ".PNBRQKpnbrqk"
PIECE_INDEX = {ch: i for i, ch in enumerate(PIECES)}
PIECE_SYMBOLS = PIECES[1:]
PIECE_SYMBOL_INDEX = {ch: i for i, ch in enumerate(PIECE_SYMBOLS)}
POLICY_MAP = "uci_queen_knight_promo_v1"
PROMOTIONS = "qrbn"
CHESSBENCH_PROMOTIONS = {"n": 0, "b": 1, "r": 2, "q": 3}
ACTION_SPACE = 64 * 64 * 5
CHESSBENCH_AV_POLICY_SIZE = 4096 + 4096 * 4
_ACTION_PROMOTIONS = {"n": 1, "b": 2, "r": 3, "q": 4}


def fixed_policy_moves() -> list[str]:
    out: set[str] = set()
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    knights = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]

    def on(f: int, r: int) -> bool:
        return 0 <= f < 8 and 0 <= r < 8

    def sq(f: int, r: int) -> str:
        return FILES[f] + str(r + 1)

    for r in range(8):
        for f in range(8):
            fr = sq(f, r)
            for df, dr in dirs:
                for n in range(1, 8):
                    tf, tr = f + df * n, r + dr * n
                    if not on(tf, tr):
                        break
                    out.add(fr + sq(tf, tr))
            for df, dr in knights:
                if on(f + df, r + dr):
                    out.add(fr + sq(f + df, r + dr))
    for r in [1, 6]:
        tr = 7 if r == 6 else 0
        for f in range(8):
            for df in [-1, 0, 1]:
                if on(f + df, tr):
                    for promo in PROMOTIONS:
                        out.add(sq(f, r) + sq(f + df, tr) + promo)
    return sorted(out)


_POLICY_MOVES: list[str] | None = None
_POLICY_INDEX: dict[str, int] | None = None


def policy_moves() -> list[str]:
    global _POLICY_MOVES
    if _POLICY_MOVES is None:
        _POLICY_MOVES = fixed_policy_moves()
    return _POLICY_MOVES


def policy_index(move: str) -> int:
    global _POLICY_INDEX
    if _POLICY_INDEX is None:
        _POLICY_INDEX = {move: i for i, move in enumerate(policy_moves())}
    return _POLICY_INDEX[move]


def square_index(square: str) -> int:
    if len(square) != 2 or square[0] not in FILES or square[1] not in "12345678":
        raise ValueError(f"invalid square: {square}")
    return FILES.index(square[0]) + (int(square[1]) - 1) * 8


def move_to_action_id(move: str) -> int:
    if len(move) not in (4, 5):
        raise ValueError(f"invalid UCI move: {move}")
    from_sq = square_index(move[:2])
    to_sq = square_index(move[2:4])
    promo = _ACTION_PROMOTIONS.get(move[4].lower(), 0) if len(move) == 5 else 0
    return (from_sq * 64 + to_sq) * 5 + promo


def move_to_chessbench_av_class(move: str) -> int:
    if len(move) not in (4, 5):
        raise ValueError(f"invalid UCI move: {move}")
    from_sq = square_index(move[:2])
    to_sq = square_index(move[2:4])
    ft = from_sq * 64 + to_sq
    if len(move) >= 5:
        promo = move[4].lower()
        if promo in CHESSBENCH_PROMOTIONS:
            return 4096 + ft * 4 + CHESSBENCH_PROMOTIONS[promo]
    return ft


# SquareFormer policy and ChessBench direct AV caches intentionally share the
# compact 20,480-class map: 4,096 from-to classes followed by 4 promotion classes
# for every from-to pair in n/b/r/q order.
move_to_squareformer_policy_index = move_to_chessbench_av_class
