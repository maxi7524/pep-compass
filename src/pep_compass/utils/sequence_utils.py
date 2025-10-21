def to_one_hot(x, pad=26):
    alphabet = list(" ACDEFGHIKLMNPQRSTVWY")
    classes = range(0, 21)
    aa_encoding = dict(zip(alphabet, classes))
    if len(x) > 25:
        x = x[:25]
    else:
        x = x + "".join([" " * (25 - len(x))])
    return [aa_encoding[aa] for aa in x]


def translate_generated_peptide(decoded_peptide):
    # Expects a tensor of shape (1, 25, 21)
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    return "".join(
        [
            alphabet[el - 1] if el != 0 else ""
            for el in decoded_peptide[0].argmax(axis=1)
        ]
    )