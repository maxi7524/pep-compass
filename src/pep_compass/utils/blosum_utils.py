import blosum as bl

_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

_CACHE: dict[int, bl.BLOSUM] = {}


def load_blosum(n: int) -> bl.BLOSUM:
    if n not in _CACHE:
        _CACHE[n] = bl.BLOSUM(n)
    return _CACHE[n]


def blosum_score(seq1: str, seq2: str, matrix: bl.BLOSUM) -> float:
    """Sum of per-position BLOSUM scores for two sequences.

    Characters not in the standard amino acid alphabet (e.g. space padding)
    contribute 0. For sequences of different lengths only the overlapping
    prefix is scored.
    """
    score = 0.0
    for a, b in zip(seq1, seq2):
        if a in _AMINO_ACIDS and b in _AMINO_ACIDS:
            score += matrix[a][b]
    return score


def is_blosum_neighbour(seq1: str, seq2: str, min_score: float, matrix: bl.BLOSUM) -> bool:
    return blosum_score(seq1, seq2, matrix) >= min_score
