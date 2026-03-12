"""Diagnostic: understand why projected direction fails and test fixes."""
import sys, os, torch, time
sys.path.insert(0, os.path.dirname(__file__))

from geodesic_mutation_check import (
    build_encoder_decoder, SEED_PEPTIDES, ALPHABET, MAX_PEPTIDE_LEN,
    compute_svd_and_mutations, mutation_direction_projected,
    euler_reprojection, _eval_endpoint, _quick_tangent_space,
    scale_direction, geodesic_live_gamma, geodesic_reprojection,
    build_frozen_gamma, check_single_mutation,
)
from pep_compass.local_enumeration.mutation.mutation_potentials import DecoderLogProbPotential
import numpy as np

enc_dec = build_encoder_decoder("cpu")
pot = DecoderLogProbPotential(enc_dec)

pname, peptide = list(SEED_PEPTIDES.items())[0]  # middle-1
padded = peptide.ljust(MAX_PEPTIDE_LEN)

with torch.no_grad():
    z_parent = enc_dec.encode_peptides([peptide])[0]

jac, U, S, V, mutations, ts = compute_svd_and_mutations(enc_dec, peptide, z_parent)
potentials = pot.compute(peptide, mutations)
flat = []
for pos, aa_dict in potentials.items():
    pidx = ALPHABET.index(padded[pos])
    for ai, lp in aa_dict.items():
        if ai != pidx:
            flat.append((pos, ai, lp))

lps = np.array([m[2] for m in flat])
sh = lps - lps.max()
probs = np.exp(sh) / np.exp(sh).sum()
above = [(m, p) for m, p in zip(flat, probs) if p > 1.0 / len(flat)]

print(f"Peptide: {pname} = {peptide}")
print(f"Testing {len(above)} mutations\n")

AMBIENT_DIM = 25 * 21  # 525

# Pick one representative mutation (first one from middle-1)
pos, ai, _ = above[0][0]
mutant_list = list(peptide)
mutant_list[pos] = ALPHABET[ai]
mutant_peptide = "".join(mutant_list)
with torch.no_grad():
    z_mut = enc_dec.encode_peptides([mutant_peptide])[0]

v_enc = z_mut - z_parent
td = torch.norm(v_enc).item()
print(f"=== Mutation: {padded[pos]}→{ALPHABET[ai]} pos={pos}  dist={td:.4f} ===\n")

# ── Diagnostic 1: How does the projected direction evolve along the ENCODING path? ──
print("Direction evolution along z_parent → z_mutant (encoding path):")
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    z_interp = z_parent + alpha * v_enc
    ts_interp, _ = _quick_tangent_space(enc_dec, z_interp)
    v_proj_interp = mutation_direction_projected(ts_interp, pos, ai)
    cos_with_enc = (torch.dot(v_proj_interp, v_enc) / (torch.norm(v_proj_interp) * torch.norm(v_enc) + 1e-12)).item()
    norm_proj = torch.norm(v_proj_interp).item()
    # Also decode this interpolated point
    with torch.no_grad():
        dec = enc_dec.decode_peptides(z_interp.unsqueeze(0))[0].strip()
    aa_at_pos = dec[pos] if pos < len(dec) else "?"
    print(f"  α={alpha:.1f}  cos(proj,enc)={cos_with_enc:+.4f}  ||v_proj||={norm_proj:.1f}  decoded[{pos}]={aa_at_pos}  decoded={dec}")

# ── Diagnostic 2: What does the one-hot e_mut vs the actual decoder diff look like? ──
print("\nAmbient directions comparison:")
with torch.no_grad():
    p_parent = enc_dec.decoder_forward(z_parent.unsqueeze(0), softmax=True, flatten=True)[0]  # (525,)
    p_mutant = enc_dec.decoder_forward(z_mut.unsqueeze(0), softmax=True, flatten=True)[0]
delta_p = p_mutant - p_parent
e_mut_onehot = torch.zeros(AMBIENT_DIM)
e_mut_onehot[pos * 21 + ai] = 1.0

# Also try e_mut as the difference at the mutation position only
e_mut_diff = torch.zeros(AMBIENT_DIM)
parent_aa_idx = ALPHABET.index(padded[pos])
e_mut_diff[pos * 21 + ai] = 1.0
e_mut_diff[pos * 21 + parent_aa_idx] = -1.0

cos_onehot_delta = (torch.dot(e_mut_onehot, delta_p) / (torch.norm(e_mut_onehot) * torch.norm(delta_p) + 1e-12)).item()
cos_diff_delta = (torch.dot(e_mut_diff, delta_p) / (torch.norm(e_mut_diff) * torch.norm(delta_p) + 1e-12)).item()
print(f"  cos(e_onehot, Δp) = {cos_onehot_delta:.4f}")
print(f"  cos(e_diff, Δp)   = {cos_diff_delta:.4f}")
print(f"  ||Δp|| = {torch.norm(delta_p).item():.4f}")
print(f"  ||e_onehot|| = {torch.norm(e_mut_onehot).item():.4f}")
print(f"  ||e_diff|| = {torch.norm(e_mut_diff).item():.4f}")

# ── Diagnostic 3: Project the actual decoder difference Δp instead of one-hot ──
print("\nProjecting Δp (actual decoder difference) instead of one-hot:")
ts_parent, _ = _quick_tangent_space(enc_dec, z_parent)
v_from_delta = ts_parent.project_ambient_vector_to_horizontal_space(delta_p)
cos_delta_enc = (torch.dot(v_from_delta, v_enc) / (torch.norm(v_from_delta) * torch.norm(v_enc) + 1e-12)).item()
v_from_delta_scaled = scale_direction(v_from_delta, td)
z_end_delta = z_parent + v_from_delta_scaled
with torch.no_grad():
    dec_delta = enc_dec.decode_peptides(z_end_delta.unsqueeze(0))[0].strip()
print(f"  cos(J^+@Δp, v_enc) = {cos_delta_enc:+.4f}")
print(f"  decoded = {dec_delta}  (target: {mutant_peptide})")

# ── Diagnostic 4: Project e_diff (signed one-hot) ──
print("\nProjecting e_diff (+1 at mutant, -1 at parent):")
v_from_diff = ts_parent.project_ambient_vector_to_horizontal_space(e_mut_diff)
cos_diff_enc = (torch.dot(v_from_diff, v_enc) / (torch.norm(v_from_diff) * torch.norm(v_enc) + 1e-12)).item()
v_from_diff_scaled = scale_direction(v_from_diff, td)
z_end_diff = z_parent + v_from_diff_scaled
with torch.no_grad():
    dec_diff = enc_dec.decode_peptides(z_end_diff.unsqueeze(0))[0].strip()
print(f"  cos(J^+@e_diff, v_enc) = {cos_diff_enc:+.4f}")
print(f"  decoded = {dec_diff}  (target: {mutant_peptide})")

# ── Diagnostic 5: Decoder gradient ascent ──
print("\n=== Decoder gradient ascent: ∂ log p(aa_mut | z, pos) / ∂z ===")

def decoder_logprob_gradient(enc_dec, z, pos, ai):
    """Compute gradient of log p(aa=ai | z, pos) w.r.t. z."""
    z_var = z.clone().detach().requires_grad_(True)
    # Forward through decoder to get logits
    logits = enc_dec.decoder_forward(z_var.unsqueeze(0), softmax=False, flatten=False)  # (1, 25, 21)
    log_probs = torch.nn.functional.log_softmax(logits[0, pos, :], dim=-1)
    loss = log_probs[ai]
    loss.backward()
    return z_var.grad.detach()

# Gradient at z_parent
grad_at_parent = decoder_logprob_gradient(enc_dec, z_parent, pos, ai)
cos_grad_enc = (torch.dot(grad_at_parent, v_enc) / (torch.norm(grad_at_parent) * torch.norm(v_enc) + 1e-12)).item()
print(f"  cos(∇ log p, v_enc) = {cos_grad_enc:+.4f}  ||∇||={torch.norm(grad_at_parent).item():.4f}")

# Compare gradient with projected direction
v_proj_at_parent = mutation_direction_projected(ts_parent, pos, ai)
cos_grad_proj = (torch.dot(grad_at_parent, v_proj_at_parent) / (torch.norm(grad_at_parent) * torch.norm(v_proj_at_parent) + 1e-12)).item()
print(f"  cos(∇ log p, v_proj) = {cos_grad_proj:+.4f}")

# Single step: Euclidean line in gradient direction, scaled to td
v_grad_scaled = grad_at_parent / torch.norm(grad_at_parent) * td
z_grad_1step = z_parent + v_grad_scaled
with torch.no_grad():
    dec_grad = enc_dec.decode_peptides(z_grad_1step.unsqueeze(0))[0].strip()
    hmm = sum(1 for a, b in zip(dec_grad, mutant_peptide) if a != b)
print(f"  1-step (Euclidean, ||td||): decoded={dec_grad}  h={hmm}  (target: {mutant_peptide})")

# Gradient ascent: iteratively follow ∂ log p / ∂z
print("\nGradient ascent (fixed step size):")
for lr, n_steps in [(0.01, 50), (0.01, 200), (0.05, 50), (0.05, 200), (0.1, 50), (0.1, 200)]:
    z_ga = z_parent.clone()
    for step_i in range(n_steps):
        g = decoder_logprob_gradient(enc_dec, z_ga, pos, ai)
        z_ga = z_ga + lr * g
    with torch.no_grad():
        dec_ga = enc_dec.decode_peptides(z_ga.unsqueeze(0))[0].strip()
    hmm = sum(1 for a, b in zip(dec_ga, mutant_peptide) if a != b)
    d_to_mut = torch.norm(z_ga - z_mut).item()
    d_from_parent = torch.norm(z_ga - z_parent).item()
    # Check decoded at position
    pos_match = (dec_ga[pos] == ALPHABET[ai]) if pos < len(dec_ga) else False
    print(f"  lr={lr} N={n_steps:3d}: h={hmm}  pos_match={pos_match}  d_mut={d_to_mut:.3f}  d_par={d_from_parent:.3f}  decoded={dec_ga}")

# ── Diagnostic 6: Multi-objective gradient (maximize mutation AA, keep rest fixed) ──
print("\n=== Multi-objective gradient: maximize mut_aa AND preserve parent ===")

def multi_obj_gradient(enc_dec, z, pos, ai, peptide, alpha_preserve=1.0):
    """Gradient that maximizes p(aa_mut | z, pos) while preserving other positions."""
    z_var = z.clone().detach().requires_grad_(True)
    logits = enc_dec.decoder_forward(z_var.unsqueeze(0), softmax=False, flatten=False)  # (1, 25, 21)
    # Mutation objective: maximize log p(ai | z, pos)
    log_probs_mut = torch.nn.functional.log_softmax(logits[0, pos, :], dim=-1)
    obj_mut = log_probs_mut[ai]
    # Preservation objective: maximize log p(parent_aa | z, pos') for all other positions
    obj_preserve = torch.tensor(0.0)
    for p in range(len(peptide)):
        if p == pos:
            continue
        parent_aa = ALPHABET.index(peptide[p]) if p < len(peptide) else 0
        log_probs_p = torch.nn.functional.log_softmax(logits[0, p, :], dim=-1)
        obj_preserve = obj_preserve + log_probs_p[parent_aa]
    total = obj_mut + alpha_preserve * obj_preserve
    total.backward()
    return z_var.grad.detach()

print("Multi-objective gradient ascent:")
for lr, n_steps, alpha in [(0.01, 100, 0.1), (0.01, 200, 0.1), (0.05, 100, 0.1), (0.01, 100, 1.0), (0.01, 200, 1.0)]:
    z_ga = z_parent.clone()
    for step_i in range(n_steps):
        g = multi_obj_gradient(enc_dec, z_ga, pos, ai, padded, alpha_preserve=alpha)
        z_ga = z_ga + lr * g
    with torch.no_grad():
        dec_ga = enc_dec.decode_peptides(z_ga.unsqueeze(0))[0].strip()
    hmm = sum(1 for a, b in zip(dec_ga, mutant_peptide) if a != b)
    d_to_mut = torch.norm(z_ga - z_mut).item()
    pos_match = (dec_ga[pos] == ALPHABET[ai]) if pos < len(dec_ga) else False
    print(f"  lr={lr} N={n_steps:3d} α={alpha}: h={hmm}  pos_match={pos_match}  d_mut={d_to_mut:.3f}  decoded={dec_ga}")

# ── Diagnostic 7: Test lr=0.01 with α=0.1 on ALL mutations ──
print("\n=== lr=0.01 α=0.1 on all mutations ===")
for n_steps in [100, 200]:
    print(f"\n  N={n_steps}:")
    for (mpos, mai, _), sp in above:
        ml = list(peptide)
        ml[mpos] = ALPHABET[mai]
        mp = "".join(ml)
        with torch.no_grad():
            zm = enc_dec.encode_peptides([mp])[0]
        z_ga = z_parent.clone()
        for step_i in range(n_steps):
            g = multi_obj_gradient(enc_dec, z_ga, mpos, mai, padded, alpha_preserve=0.1)
            z_ga = z_ga + 0.01 * g
        with torch.no_grad():
            dec_ga = enc_dec.decode_peptides(z_ga.unsqueeze(0))[0].strip()
        hmm = sum(1 for a, b in zip(dec_ga, mp) if a != b)
        d_to_mut = torch.norm(z_ga - zm).item()
        d_from_par = torch.norm(z_ga - z_parent).item()
        pos_match = (dec_ga[mpos] == ALPHABET[mai]) if mpos < len(dec_ga) else False
        print(f"    {padded[mpos]}→{ALPHABET[mai]} pos={mpos}: h={hmm}  pos_match={pos_match}  d_mut={d_to_mut:.3f}  d_par={d_from_par:.3f}  decoded={dec_ga}  (target={mp})")

# ── Diagnostic 8: Vary α and lr systematically ──
print("\n=== Hyperparameter sweep (N=150) ===")
first_mut = above[0][0]
mpos, mai, _ = first_mut
ml = list(peptide); ml[mpos] = ALPHABET[mai]; mp = "".join(ml)
with torch.no_grad():
    zm = enc_dec.encode_peptides([mp])[0]

for lr in [0.005, 0.01, 0.02]:
    for alpha in [0.05, 0.1, 0.2, 0.5]:
        z_ga = z_parent.clone()
        for step_i in range(150):
            g = multi_obj_gradient(enc_dec, z_ga, mpos, mai, padded, alpha_preserve=alpha)
            z_ga = z_ga + lr * g
        with torch.no_grad():
            dec_ga = enc_dec.decode_peptides(z_ga.unsqueeze(0))[0].strip()
        hmm = sum(1 for a, b in zip(dec_ga, mp) if a != b)
        d_from_par = torch.norm(z_ga - z_parent).item()
        print(f"  lr={lr} α={alpha}: h={hmm}  d_par={d_from_par:.3f}  decoded={dec_ga}")
