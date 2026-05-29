"""Standalone motif-based filtering function for peptide sequences."""

import re
from typing import Literal, Optional, Tuple, Set


# Amino acid classifications
HYDROPHOBIC = set("AILMFWYV")  # A, I, L, M, F, W, Y, V
HYDROPHILIC = set("STNQDERK")  # S, T, N, Q, D, E, R, K
BULKY_HINDERED = set("WFY")  # Tryptophan, Tyrosine, Phenylalanine
CATIONIC = set("KRH")  # Lysine, Arginine, Histidine
ANIONIC = set("DE")  # Aspartic acid, Glutamic acid
AROMATIC = set("FWY")  # Phenylalanine, Tryptophan, Tyrosine
SMALL_HYDROPHOBIC = set("AGV")  # Small hydrophobic residues

# Aggregation-prone motifs
AGGREGATION_MOTIFS = [
    "VLVL",  # β-sheet formation and hydrophobic packing
    "WWYF",  # Aromatic stacking
    "KLLL",  # Hydrophobic collapse
    "STVIIE", "VQIVYK", "GNNQQNY", "QYNNQ",  # Zipper Motifs
    "QQQQQQQQQ", "NNNNNNNN",  # PolyQ and PolyN
    "GSVIIE", "VQIVYK", "SSTY", "SVQIVY",  # Steric Zipper Motifs
    "KLVFFA", "LVFFAEDVGSNK",  # Aβ Aggregation
    "VTGVTAVAQK",  # α-Synuclein Aggregation
    "GSSGSS",  # Prion-Like Domains
    "VLIVLG", "YVIVFV",  # Hydrophobic β-strand rich sequences
    "LIVVV", "WLIVI",  # Aggregation-prone hotspots
    "VVVIV", "LLVVV",  # β-Sheet Forming motifs
    "PXXPXX", "PQPQPQ",  # Polyproline-II-like Aggregation Motifs
    "LVFFA", "IYFV",  # Amphipathic β-Strand Motifs
    "STVIIE", "TTVIE", "NFGAIL",  # Known hexapeptide aggregation motifs
    "DFNKF",  # Amyloid-forming motifs
]

# Structural motifs - α-helix
ALPHA_HELIX_MOTIFS = [
    "EAAAK",  # Alanine-rich "HEL" core
    "KΦKΦKΦ",  # Amphipathic helix pattern (Φ = hydrophobic)
]

# Structural motifs - β-sheet
BETA_SHEET_MOTIFS = [
    "DPG",  # β-hairpin with type II' turn (D-Pro-Gly)
    "LPG",  # β-hairpin with type II turn (L-Pro-Gly)
]

# Structural motifs - β-turn
BETA_TURN_MOTIFS = [
    "XDNX", "XNNX", "XDSX", "XNSX",  # Type I turn (X-(D/N)-(G/A/S)-X)
    "XPGX",  # Type II turn (X-Pro-Gly-X)
    "XDDX", "XNDX",  # Type I' / II' turn
]

# Structural motifs - PPII (Polyproline-II)
PPII_MOTIFS = [
    "PXXP",  # P-X-X-P (SH3-like)
    "GPG", "GAP",  # Collagen repeat patterns
]

# Structural motifs - Transmembrane
TM_MOTIFS = [
    "GXXXG",  # G-X-X-X-G (gly zipper)
    "AXXXA", "SXXXS",  # Small-xxx-small in TM
]

# Gram-positive selective peptides (validated)
GRAM_POSITIVE_SELECTIVE = [
    "LFFIIRILRILKLL",
    "IIGLSLIAKILAKLL",
    "IFLHLAILKIIRLL",
    "ILLSIWSGIKGLL",
    "ILFPIVSVLKGLL",
    "IILRIIAKFSKLLL",
    "PYPYPFRPLPPIPFPRFPWFRRNFPIPIP",
]

# Gram-negative selective peptides (validated)
GRAM_NEGATIVE_SELECTIVE = [
    "NIWFKLVKQRQLRRFKRYPFLRGKYVDGTRGG",
    "QISKLLARLRRLKWAKIRTQIAGVRKYFTRSYWRELIAK",
    "IGWISNRVILKLLRFAKNMK",
    "TWFIKVSLIRKKLLENNEKINW",
    "KRFWTLIKMLRF",
    "LGKRLMGWLYKLAIKIWDNKLRKATGGL",
    "KRRWKWKKIKKLLKLL",
    "MLKLLSIKLKSKKKILRMTLTKLNKWLKTKKKWLRNLKTM",
    "MIKVSKLVVAKKLNLRKKLKKYLKKPKLGKILLL",
    "MLNTRLLAQIKKWVAAKRWR",
    "KIKIKIKKLKKLKLKLKLKLKLKN",
    "MKKRIRKKILKKILKWLKKGKHRAYL",
    "LKINRALRARLKLRTKLSLYFKRRANA",
    "NKIRWINKYVKKLQLKRLLVKS",
    "KLAKLLKKLAKLLK",
    "KWKLFKKIEKVGRNIRDGIIKAGPAVAVVGQAASLAK",
    "GLKALKKVSKGIHKAIKLINNHVH",
    "GWLKKIGKKIERVGQHTRDATIQGLGIAQQAANVAATAR",
    "GWLKKIGKKIERVGQHTRDATIQGIGVAQQAINVAATAK",
    "GWLKKIGKKIERVGQHTRDATITGLGIAQQAANVAATAR",
    "GWLKKIGKKIERVGQHTRDATIQTIGIAQQAANVALTAR",
    "GWLKKIGKKIERVGQHTRDATIQGIGVAQQAINVAATAK",
    "GWLKKIGKKIERVGQHTRDATIQGIGVAQQAANVAATVK",
    "KIGKKLSRWLFRLNIRGLPKVKFPK",
    "FIAKKLSLRWFRLNIRTLPKVKFPK",
    "LRIKNHWIHFRVLAKALW",
    "RNPLRKVRYSLRLPAKAAQRLKRIANFQPGSLRRIGLSALK",
]

# Immunomodulatory motifs
IMMUNOMODULATORY_MOTIFS = [
    "BBXB", "XBBXBX", "XBBBXXBX",  # Heparan-sulfate binding (B = K/R)
    "FSSE",  # Blocks HMGB1–MD-2/TLR4
    "RRWQWR",  # LPS binding/neutralization
    "KRIVKLIKKWLR",  # KR-12–type cores
    "KWL", "KRL",  # LL-37 C-term clusters
    "KW", "KWK",  # LPS/LTA binding
    "GKYGFY",  # Thrombin TCP seed
    "PR", "PRP",  # Pro-Arg repeats
    "GXCX",  # γ-core motif (simplified)
    "WKYMV",  # FPR2 agonism (note: WKYMVm has lowercase m, we'll check case-insensitive)
    "WRW4",  # FPR2 antagonism
    "TKPR",  # Tuftsin
    "KPV",  # α-MSH-derived
    "GHK",  # Anti-inflammatory
    "RGD", "LDV",  # Integrin binding
    "SLIGKV", "SLIGRL",  # PAR-2 agonism
    "VGVAPG",  # Elastokine
    "PGP",  # CXCR2 agonism
]


def calculate_hydrophobic_abundance(sequence: str) -> float:
    """Calculate the percentage of hydrophobic residues in a sequence."""
    if not sequence:
        return 0.0
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    if not seq_clean:
        return 0.0
    hydrophobic_count = sum(1 for aa in seq_clean if aa in HYDROPHOBIC)
    return (hydrophobic_count / len(seq_clean)) * 100


def calculate_hydrophilic_abundance(sequence: str) -> float:
    """Calculate the percentage of hydrophilic residues in a sequence."""
    if not sequence:
        return 0.0
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    if not seq_clean:
        return 0.0
    hydrophilic_count = sum(1 for aa in seq_clean if aa in HYDROPHILIC)
    return (hydrophilic_count / len(seq_clean)) * 100


def has_glycine_repeats(sequence: str, min_repeats: int = 3) -> bool:
    """Check if sequence contains ≥N consecutive glycines."""
    if not sequence:
        return False
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    pattern = "G" * min_repeats
    return pattern in seq_clean


def has_bulky_residue_repeats(sequence: str, min_repeats: int = 2) -> bool:
    """Check if sequence contains consecutive bulky hindered residues."""
    if not sequence:
        return False
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    pattern = "[" + "".join(BULKY_HINDERED) + "]" + "{" + str(min_repeats) + ",}"
    return bool(re.search(pattern, seq_clean))


def has_aggregation_motif(sequence: str, motifs: Optional[list[str]] = None) -> bool:
    """Check if sequence contains any aggregation-prone motifs."""
    if not sequence:
        return False
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    
    if motifs is None:
        motifs = AGGREGATION_MOTIFS
    
    for motif in motifs:
        # Handle patterns with X (any residue)
        if "X" in motif:
            # Convert X to regex wildcard
            pattern = motif.replace("X", ".")
            if re.search(pattern, seq_clean):
                return True
        else:
            if motif in seq_clean:
                return True
    return False


def has_structural_motif(sequence: str, motif_type: str) -> bool:
    """
    Check if sequence contains structural motifs.
    
    Args:
        sequence: Peptide sequence
        motif_type: Type of structural motif to check:
            - "alpha_helix": α-helix motifs
            - "beta_sheet": β-sheet motifs
            - "beta_turn": β-turn motifs
            - "ppii": Polyproline-II motifs
            - "tm": Transmembrane motifs
    
    Returns:
        True if sequence contains the specified structural motif type
    """
    if not sequence:
        return False
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    
    motif_map = {
        "alpha_helix": ALPHA_HELIX_MOTIFS,
        "beta_sheet": BETA_SHEET_MOTIFS,
        "beta_turn": BETA_TURN_MOTIFS,
        "ppii": PPII_MOTIFS,
        "tm": TM_MOTIFS,
    }
    
    if motif_type not in motif_map:
        raise ValueError(f"Unknown motif type: {motif_type}. Must be one of {list(motif_map.keys())}")
    
    motifs = motif_map[motif_type]
    
    for motif in motifs:
        # Handle patterns with X (any residue) or Φ (hydrophobic)
        if "X" in motif:
            pattern = motif.replace("X", ".")
            if re.search(pattern, seq_clean):
                return True
        elif "Φ" in motif:
            # Replace Φ with hydrophobic residues
            pattern = motif.replace("Φ", f"[{''.join(HYDROPHOBIC)}]")
            if re.search(pattern, seq_clean):
                return True
        else:
            if motif in seq_clean:
                return True
    return False


def has_gram_positive_motif(sequence: str) -> bool:
    """Check if sequence contains Gram-positive selective peptide motifs."""
    if not sequence:
        return False
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    
    for motif in GRAM_POSITIVE_SELECTIVE:
        if motif in seq_clean:
            return True
    return False


def has_gram_negative_motif(sequence: str) -> bool:
    """Check if sequence contains Gram-negative selective peptide motifs."""
    if not sequence:
        return False
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    
    for motif in GRAM_NEGATIVE_SELECTIVE:
        if motif in seq_clean:
            return True
    return False


def has_immunomodulatory_motif(sequence: str) -> bool:
    """Check if sequence contains immunomodulatory motifs."""
    if not sequence:
        return False
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    
    for motif in IMMUNOMODULATORY_MOTIFS:
        # Handle patterns with X (any residue) or B (basic = K/R)
        if "X" in motif:
            pattern = motif.replace("X", ".")
            if re.search(pattern, seq_clean):
                return True
        elif "B" in motif:
            # Replace B with basic residues (K/R)
            pattern = motif.replace("B", "[KR]")
            if re.search(pattern, seq_clean):
                return True
        else:
            # Case-insensitive check for motifs with lowercase letters
            if motif.upper() in seq_clean or motif.lower() in seq_clean:
                return True
    return False


def calculate_proline_percentage(sequence: str) -> float:
    """Calculate the percentage of proline residues."""
    if not sequence:
        return 0.0
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    if not seq_clean:
        return 0.0
    proline_count = seq_clean.count("P")
    return (proline_count / len(seq_clean)) * 100


def calculate_net_charge(sequence: str) -> int:
    """Calculate net charge (positive - negative)."""
    if not sequence:
        return 0
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    if not seq_clean:
        return 0
    positive = sum(1 for aa in seq_clean if aa in CATIONIC)
    negative = sum(1 for aa in seq_clean if aa in ANIONIC)
    return positive - negative


def has_sequential_repeats(sequence: str, min_repeats: int = 3, motif_length: int = 3) -> bool:
    """Check if sequence contains ≥N repeats of any tripeptide (or specified length) motif."""
    if not sequence or len(sequence) < motif_length * min_repeats:
        return False
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    if len(seq_clean) < motif_length * min_repeats:
        return False
    
    for i in range(len(seq_clean) - motif_length * min_repeats + 1):
        motif = seq_clean[i:i + motif_length]
        # Check if this motif repeats at least min_repeats times
        pattern = motif * min_repeats
        if pattern in seq_clean:
            return True
    return False


def count_cysteine_residues(sequence: str) -> int:
    """Count the number of cysteine residues in a sequence."""
    if not sequence:
        return 0
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    return seq_clean.count("C")


def has_hydrophobic_runs(sequence: str, min_length: int = 3) -> bool:
    """Check if sequence contains runs of hydrophobic residues (≥N consecutive)."""
    if not sequence:
        return False
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    pattern = "[" + "".join(HYDROPHOBIC) + "]" + "{" + str(min_length) + ",}"
    return bool(re.search(pattern, seq_clean))


def has_qn_rich_tracts(sequence: str, min_length: int = 5) -> bool:
    """Check if sequence contains Q/N-rich tracts (≥N consecutive Q or N)."""
    if not sequence:
        return False
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    pattern = "[QN]" + "{" + str(min_length) + ",}"
    return bool(re.search(pattern, seq_clean))


def filter_peptide_by_motifs(
    sequence: str,
    *,
    # Hydrophobic/hydrophilic filters
    filter_hydrophobic_abundance: bool = True,
    hydrophobic_threshold: float = 70.0,
    filter_hydrophilic_abundance: bool = True,
    hydrophilic_threshold: float = 70.0,
    # Glycine repeats
    filter_glycine_repeats: bool = True,
    glycine_repeat_threshold: int = 3,
    # Bulky residues
    filter_bulky_repeats: bool = True,
    bulky_repeat_threshold: int = 2,
    # Aggregation motifs
    filter_aggregation_motifs: bool = True,
    custom_aggregation_motifs: Optional[list[str]] = None,
    # Proline
    filter_proline_percentage: bool = True,
    proline_threshold: float = 20.0,
    # Charge filters
    filter_excessive_negative_charge: bool = True,
    min_net_charge: int = 2,
    filter_excessive_positive_charge: bool = True,
    max_net_charge: int = 10,
    # Sequential repeats
    filter_sequential_repeats: bool = True,
    sequential_repeat_threshold: int = 3,
    sequential_repeat_motif_length: int = 3,
    # Cysteine
    filter_cysteine_count: bool = True,
    max_cysteine: int = 1,
    # Extended filters
    filter_hydrophobic_runs: bool = False,
    hydrophobic_run_min_length: int = 3,
    filter_qn_rich_tracts: bool = False,
    qn_rich_min_length: int = 5,
    filter_structural_motifs: Optional[list[str]] = None,
    filter_gram_positive_motifs: bool = False,
    filter_gram_negative_motifs: bool = False,
    filter_immunomodulatory_motifs: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Filter a peptide sequence based on motif-based rules.
    
    Args:
        sequence: Peptide sequence (string of amino acids)
        filter_hydrophobic_abundance: Filter if hydrophobic abundance > threshold
        hydrophobic_threshold: Maximum allowed hydrophobic abundance (%)
        filter_hydrophilic_abundance: Filter if hydrophilic abundance > threshold
        hydrophilic_threshold: Maximum allowed hydrophilic abundance (%)
        filter_glycine_repeats: Filter if contains ≥N consecutive glycines
        glycine_repeat_threshold: Minimum number of consecutive glycines to filter
        filter_bulky_repeats: Filter if contains consecutive bulky residues
        bulky_repeat_threshold: Minimum number of consecutive bulky residues to filter
        filter_aggregation_motifs: Filter if contains aggregation-prone motifs
        custom_aggregation_motifs: Custom list of aggregation motifs (if None, uses default)
        filter_proline_percentage: Filter if proline percentage > threshold
        proline_threshold: Maximum allowed proline percentage
        filter_excessive_negative_charge: Filter if net charge < min_net_charge
        min_net_charge: Minimum allowed net charge
        filter_excessive_positive_charge: Filter if net charge > max_net_charge
        max_net_charge: Maximum allowed net charge
        filter_sequential_repeats: Filter if contains ≥N repeats of tripeptide motifs
        sequential_repeat_threshold: Minimum number of repeats to filter
        sequential_repeat_motif_length: Length of motif to check for repeats
        filter_cysteine_count: Filter if cysteine count > max_cysteine
        max_cysteine: Maximum allowed cysteine residues
        filter_hydrophobic_runs: Filter if contains runs of hydrophobic residues
        hydrophobic_run_min_length: Minimum length of hydrophobic run to filter
        filter_qn_rich_tracts: Filter if contains Q/N-rich tracts
        qn_rich_min_length: Minimum length of Q/N tract to filter
        filter_structural_motifs: List of structural motif types to filter (e.g., ["alpha_helix", "beta_sheet"])
        filter_gram_positive_motifs: Filter if contains Gram-positive selective motifs
        filter_gram_negative_motifs: Filter if contains Gram-negative selective motifs
        filter_immunomodulatory_motifs: Filter if contains immunomodulatory motifs
    
    Returns:
        Tuple of (passes_filter: bool, reason: Optional[str])
        - If passes_filter is True, reason is None
        - If passes_filter is False, reason contains the violation description
    """
    if not sequence:
        return False, "Empty sequence"
    
    seq_upper = sequence.upper().strip()
    
    # Remove any non-amino acid characters (spaces, etc.)
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    
    if not seq_clean:
        return False, "No valid amino acids found"
    
    # 1. Check hydrophobic abundance
    if filter_hydrophobic_abundance:
        hydrophobic_pct = calculate_hydrophobic_abundance(seq_clean)
        if hydrophobic_pct > hydrophobic_threshold:
            return False, f"Hydrophobic abundance too high: {hydrophobic_pct:.1f}% > {hydrophobic_threshold}%"
    
    # 2. Check hydrophilic abundance
    if filter_hydrophilic_abundance:
        hydrophilic_pct = calculate_hydrophilic_abundance(seq_clean)
        if hydrophilic_pct > hydrophilic_threshold:
            return False, f"Hydrophilic abundance too high: {hydrophilic_pct:.1f}% > {hydrophilic_threshold}%"
    
    # 3. Check glycine repeats (≥3 consecutive Gly)
    if filter_glycine_repeats:
        if has_glycine_repeats(seq_clean, min_repeats=glycine_repeat_threshold):
            return False, f"Contains {glycine_repeat_threshold} or more consecutive glycines"
    
    # 4. Check bulky residue repeats
    if filter_bulky_repeats:
        if has_bulky_residue_repeats(seq_clean, min_repeats=bulky_repeat_threshold):
            return False, f"Contains {bulky_repeat_threshold} or more consecutive bulky residues (W/F/Y)"
    
    # 5. Check aggregation motifs
    if filter_aggregation_motifs:
        if has_aggregation_motif(seq_clean, motifs=custom_aggregation_motifs):
            return False, "Contains aggregation-prone motif"
    
    # 6. Check proline percentage
    if filter_proline_percentage:
        proline_pct = calculate_proline_percentage(seq_clean)
        if proline_pct > proline_threshold:
            return False, f"Proline percentage too high: {proline_pct:.1f}% > {proline_threshold}%"
    
    # 7. Check net charge (excessive negative)
    if filter_excessive_negative_charge:
        net_charge = calculate_net_charge(seq_clean)
        if net_charge < min_net_charge:
            return False, f"Net charge too negative: {net_charge} < {min_net_charge}"
    
    # 8. Check net charge (excessive positive)
    if filter_excessive_positive_charge:
        net_charge = calculate_net_charge(seq_clean)
        if net_charge > max_net_charge:
            return False, f"Net charge too positive: {net_charge} > {max_net_charge}"
    
    # 9. Check sequential repeats (≥3 repeats of any tripeptide motif)
    if filter_sequential_repeats:
        if has_sequential_repeats(seq_clean, min_repeats=sequential_repeat_threshold, motif_length=sequential_repeat_motif_length):
            return False, f"Contains {sequential_repeat_threshold} or more repeats of a {sequential_repeat_motif_length}-mer motif"
    
    # 10. Check cysteine count
    if filter_cysteine_count:
        cysteine_count = count_cysteine_residues(seq_clean)
        if cysteine_count > max_cysteine:
            return False, f"Too many cysteine residues: {cysteine_count} > {max_cysteine}"
    
    # 11. Check hydrophobic runs
    if filter_hydrophobic_runs:
        if has_hydrophobic_runs(seq_clean, min_length=hydrophobic_run_min_length):
            return False, f"Contains hydrophobic run of length >= {hydrophobic_run_min_length}"
    
    # 12. Check Q/N-rich tracts
    if filter_qn_rich_tracts:
        if has_qn_rich_tracts(seq_clean, min_length=qn_rich_min_length):
            return False, f"Contains Q/N-rich tract of length >= {qn_rich_min_length}"
    
    # 13. Check structural motifs
    if filter_structural_motifs:
        for motif_type in filter_structural_motifs:
            if has_structural_motif(seq_clean, motif_type):
                return False, f"Contains {motif_type} structural motif"
    
    # 14. Check Gram-positive selective motifs
    if filter_gram_positive_motifs:
        if has_gram_positive_motif(seq_clean):
            return False, "Contains Gram-positive selective peptide motif"
    
    # 15. Check Gram-negative selective motifs
    if filter_gram_negative_motifs:
        if has_gram_negative_motif(seq_clean):
            return False, "Contains Gram-negative selective peptide motif"
    
    # 16. Check immunomodulatory motifs
    if filter_immunomodulatory_motifs:
        if has_immunomodulatory_motif(seq_clean):
            return False, "Contains immunomodulatory motif"
    
    # All checks passed
    return True, None


# Convenience function with default settings
def is_valid_peptide(sequence: str) -> bool:
    """
    Quick check if a peptide sequence passes all default motif filters.
    
    Args:
        sequence: Peptide sequence
    
    Returns:
        True if sequence passes all filters, False otherwise
    """
    passes, _ = filter_peptide_by_motifs(sequence)
    return passes


# Function to get detailed validation report
def validate_peptide_detailed(sequence: str) -> dict:
    """
    Get a detailed validation report for a peptide sequence.
    
    Args:
        sequence: Peptide sequence
    
    Returns:
        Dictionary with validation results and metrics
    """
    if not sequence:
        return {
            "valid": False,
            "reason": "Empty sequence",
            "metrics": {}
        }
    
    seq_upper = sequence.upper().strip()
    seq_clean = "".join(c for c in seq_upper if c.isalpha())
    
    if not seq_clean:
        return {
            "valid": False,
            "reason": "No valid amino acids found",
            "metrics": {}
        }
    
    # Calculate all metrics
    hydrophobic_count = sum(1 for aa in seq_clean if aa in HYDROPHOBIC)
    hydrophilic_count = sum(1 for aa in seq_clean if aa in HYDROPHILIC)
    positive = sum(1 for aa in seq_clean if aa in CATIONIC)
    negative = sum(1 for aa in seq_clean if aa in ANIONIC)
    proline_count = seq_clean.count("P")
    cysteine_count = seq_clean.count("C")
    aromatic_count = sum(1 for aa in seq_clean if aa in AROMATIC)
    
    metrics = {
        "length": len(seq_clean),
        "hydrophobic_abundance": (hydrophobic_count / len(seq_clean)) * 100,
        "hydrophilic_abundance": (hydrophilic_count / len(seq_clean)) * 100,
        "net_charge": positive - negative,
        "positive_charge": positive,
        "negative_charge": negative,
        "proline_percentage": (proline_count / len(seq_clean)) * 100,
        "cysteine_count": cysteine_count,
        "aromatic_count": aromatic_count,
        "has_glycine_repeats": has_glycine_repeats(seq_clean),
        "has_bulky_repeats": has_bulky_residue_repeats(seq_clean),
        "has_aggregation_motif": has_aggregation_motif(seq_clean),
        "has_sequential_repeats": has_sequential_repeats(seq_clean),
        "has_hydrophobic_runs": has_hydrophobic_runs(seq_clean),
        "has_qn_rich_tracts": has_qn_rich_tracts(seq_clean),
        "has_gram_positive_motif": has_gram_positive_motif(seq_clean),
        "has_gram_negative_motif": has_gram_negative_motif(seq_clean),
        "has_immunomodulatory_motif": has_immunomodulatory_motif(seq_clean),
    }
    
    # Run validation
    passes, reason = filter_peptide_by_motifs(seq_clean)
    
    return {
        "valid": passes,
        "reason": reason,
        "metrics": metrics,
        "sequence": seq_clean
    }




