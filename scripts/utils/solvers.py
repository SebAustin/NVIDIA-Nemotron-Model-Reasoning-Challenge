"""Deterministic per-family solvers for the Nemotron Reasoning puzzles.

Each puzzle family is a few-shot *rule induction* task: the prompt shows several
input->output examples of a hidden rule, then asks for the output on a new input.
A solver parses the examples, infers the rule, and returns the predicted answer
(as the exact string the competition expects), or ``None`` if it cannot.

Families (see 01_eda.py / nemotron-ground-truth):
    gravity, unit_conversion, numeral      -> solved 100% (verified on train)
    bit_manipulation, encryption, equation -> hard (rule-induction); best effort

Run ``python scripts/utils/solvers.py --data data/train.csv`` to print per-family
solve/accuracy against the real metric.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------
# Family detection (canonical category keys used everywhere downstream)
# ----------------------------------------------------------------------------
FAMILIES = [
    "bit_manipulation",
    "gravity",
    "unit_conversion",
    "encryption",
    "numeral",
    "equation",
]


def classify_family(prompt: str) -> str:
    t = str(prompt).lower()
    if "bit manipulation" in t:
        return "bit_manipulation"
    if "gravitational constant" in t:
        return "gravity"
    if "unit conversion" in t:
        return "unit_conversion"
    if "encryption rules" in t:
        return "encryption"
    if "numeral system" in t:
        return "numeral"
    if "applied to equations" in t:
        return "equation"
    return "other"


# ----------------------------------------------------------------------------
# Shared parsing helpers
# ----------------------------------------------------------------------------
def _arrow_pairs(prompt: str, sep: str = "->") -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in str(prompt).splitlines():
        if sep in line:
            lhs, rhs = line.split(sep, 1)
            pairs.append((lhs.strip(), rhs.strip()))
    return pairs


# ----------------------------------------------------------------------------
# gravity:  d = 0.5 * g * t^2, with g secretly changed (constant within a row)
# ----------------------------------------------------------------------------
def solve_gravity(prompt: str) -> Optional[str]:
    obs = re.findall(r"t\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)", str(prompt))
    q = re.search(r"determine the falling distance for t\s*=\s*([\d.]+)", str(prompt))
    if not obs or not q:
        return None
    # least squares through the origin: d = g * (0.5 t^2)
    xs = [0.5 * float(t) ** 2 for t, _ in obs]
    ys = [float(d) for _, d in obs]
    denom = sum(x * x for x in xs)
    if denom == 0:
        return None
    g = sum(x * y for x, y in zip(xs, ys)) / denom
    tt = float(q.group(1))
    return f"{g * 0.5 * tt * tt:.2f}"


# ----------------------------------------------------------------------------
# unit_conversion:  y = k*x (+ b), linear scale/offset
# ----------------------------------------------------------------------------
def solve_unit_conversion(prompt: str) -> Optional[str]:
    pairs = re.findall(r"([\d.]+)\s*m\s*becomes\s*([\d.]+)", str(prompt))
    q = re.search(r"convert the following measurement:\s*([\d.]+)", str(prompt))
    if not pairs or not q:
        return None
    xs = [float(a) for a, _ in pairs]
    ys = [float(b) for _, b in pairs]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) > 1e-12:
        k = (n * sxy - sx * sy) / den
        b = (sy - k * sx) / n
    else:
        k, b = (sy / sx if sx else 0.0), 0.0
    xt = float(q.group(1))
    return f"{k * xt + b:.2f}"


# ----------------------------------------------------------------------------
# numeral:  standard Roman numerals (verified across train)
# ----------------------------------------------------------------------------
_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def to_roman(n: int) -> Optional[str]:
    if n <= 0 or n > 3999:
        return None
    out = []
    for v, sym in _ROMAN:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def solve_numeral(prompt: str) -> Optional[str]:
    pairs = re.findall(r"(\d+)\s*->\s*([A-Za-z]+)", str(prompt))
    q = re.search(r"write the number (\d+)", str(prompt))
    if not pairs or not q:
        return None
    if all(to_roman(int(n)) == r for n, r in pairs):
        return to_roman(int(q.group(1)))
    return None


# ----------------------------------------------------------------------------
# encryption: monoalphabetic letter substitution, aligned from example sentences.
# Optional Wonderland vocabulary lets us recover letters unseen in a row's
# examples by pattern-matching test words against known plaintext words.
# ----------------------------------------------------------------------------
def _enc_letter_map(pairs: Sequence[Tuple[str, str]]) -> Optional[Dict[str, str]]:
    cmap: Dict[str, str] = {}
    for cipher, plain in pairs:
        if len(cipher) != len(plain):
            return None
        for a, b in zip(cipher, plain):
            if a == " " or b == " ":
                if a != b:
                    return None
                continue
            if a in cmap and cmap[a] != b:
                return None
            cmap[a] = b
    return cmap


def solve_encryption(prompt: str, vocab: Optional[set] = None) -> Optional[str]:
    pairs = _arrow_pairs(prompt)
    q = re.search(r"decrypt the following text:\s*(.+)$", str(prompt), re.M)
    if not pairs or not q:
        return None
    cmap = _enc_letter_map(pairs)
    if cmap is None:
        return None
    test = q.group(1).strip()
    # direct decode where possible
    words_out: List[str] = []
    fully_ok = True
    for word in test.split(" "):
        chars = []
        unknown = False
        for ch in word:
            if ch in cmap:
                chars.append(cmap[ch])
            else:
                chars.append(None)
                unknown = True
        if not unknown:
            words_out.append("".join(chars))
            continue
        # try to fill from vocabulary by pattern (known letters fixed)
        if vocab:
            cand = _match_vocab(word, cmap, vocab)
            if cand is not None:
                # learn the new letters so later words benefit
                for ch, pl in zip(word, cand):
                    cmap.setdefault(ch, pl)
                words_out.append(cand)
                continue
        fully_ok = False
        words_out.append("".join(c if c else "?" for c in chars))
    if not fully_ok:
        return None
    return " ".join(words_out)


def _match_vocab(cipher_word: str, cmap: Dict[str, str], vocab: set) -> Optional[str]:
    """Find the unique vocab word consistent with the known cipher->plain letters
    and an injective extension for unknown cipher letters."""
    known_plain = set(cmap.values())
    candidates = []
    for w in vocab:
        if len(w) != len(cipher_word):
            continue
        ok = True
        local: Dict[str, str] = {}
        used = set()
        for ch, pl in zip(cipher_word, w):
            if ch in cmap:
                if cmap[ch] != pl:
                    ok = False
                    break
            else:
                # new cipher letter must map to an unused plaintext letter
                if pl in known_plain or pl in used:
                    # could still be valid if it maps consistently within the word
                    if local.get(ch, pl) != pl:
                        ok = False
                        break
                if ch in local and local[ch] != pl:
                    ok = False
                    break
                local[ch] = pl
                used.add(pl)
        if ok:
            candidates.append(w)
    if len(candidates) == 1:
        return candidates[0]
    return None


def build_encryption_vocab(prompts: Sequence[str]) -> set:
    """Collect the plaintext vocabulary from all encryption example RHS words."""
    vocab: set = set()
    for p in prompts:
        for _, plain in _arrow_pairs(p):
            for w in plain.split():
                if w.isalpha():
                    vocab.add(w)
    return vocab


# ----------------------------------------------------------------------------
# bit_manipulation: 8-bit -> 8-bit.  (1) affine over GF(2) per output bit with a
# determinacy + leave-one-out check; (2) a library of named ops composed with a
# derived XOR/ADD constant; accept only hypotheses that survive leave-one-out.
# ----------------------------------------------------------------------------
def _bits(s: str) -> List[int]:
    return [int(c) for c in s]


def _gf2_consistent_and_predict(
    ins: List[List[int]], outs_bit: List[int], test: List[int]
) -> Tuple[bool, Optional[int]]:
    """Solve A w = b over GF(2) for one output bit (A rows = augmented inputs).
    Returns (consistent, predicted_bit_or_None_if_undetermined)."""
    rows = [r[:] for r in ins]
    b = outs_bit[:]
    nrows = len(rows)
    ncols = len(rows[0])
    pivots: List[int] = []
    r = 0
    for col in range(ncols):
        piv = next((rr for rr in range(r, nrows) if rows[rr][col]), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        b[r], b[piv] = b[piv], b[r]
        for rr in range(nrows):
            if rr != r and rows[rr][col]:
                rows[rr] = [x ^ y for x, y in zip(rows[rr], rows[r])]
                b[rr] ^= b[r]
        pivots.append(col)
        r += 1
        if r == nrows:
            break
    for rr in range(nrows):
        if not any(rows[rr]) and b[rr]:
            return False, None  # inconsistent: not affine
    w = [0] * ncols
    for i, col in enumerate(pivots):
        w[col] = b[i]
    return True, sum(wi * ti for wi, ti in zip(w, test)) % 2


def _affine_predict(ins, outs, test) -> Optional[str]:
    # determinacy: test (input side) must lie in row space of example inputs
    at = [[ins[r][c] for r in range(len(ins))] for c in range(len(ins[0]))]
    det_ok, _ = _gf2_consistent_and_predict(at, test, [0] * len(at[0]))
    # The above only checks consistency of A^T x = test; reuse solver for membership:
    if not _in_rowspace(ins, test):
        return None
    bits = []
    for j in range(8):
        ok, pred = _gf2_consistent_and_predict(ins, [o[j] for o in outs], test)
        if not ok or pred is None:
            return None
        bits.append(pred)
    return "".join(str(x) for x in bits)


def _in_rowspace(ins: List[List[int]], test: List[int]) -> bool:
    # is `test` a GF(2) linear combination of rows of `ins`? solve ins^T x = test
    at = [[ins[r][c] for r in range(len(ins))] for c in range(len(ins[0]))]
    b = test[:]
    nrows = len(at)
    ncols = len(at[0])
    r = 0
    for col in range(ncols):
        piv = next((rr for rr in range(r, nrows) if at[rr][col]), None)
        if piv is None:
            continue
        at[r], at[piv] = at[piv], at[r]
        b[r], b[piv] = b[piv], b[r]
        for rr in range(nrows):
            if rr != r and at[rr][col]:
                at[rr] = [x ^ y for x, y in zip(at[rr], at[r])]
                b[rr] ^= b[r]
        r += 1
        if r == nrows:
            break
    for rr in range(nrows):
        if not any(at[rr]) and b[rr]:
            return False
    return True


def _bit_op_library() -> Dict[str, Callable[[List[int]], List[int]]]:
    ops: Dict[str, Callable[[List[int]], List[int]]] = {
        "id": lambda b: b[:],
        "not": lambda b: [1 - x for x in b],
        "reverse": lambda b: b[::-1],
        "swap_nibbles": lambda b: b[4:] + b[:4],
    }
    for k in range(1, 8):
        ops[f"rol{k}"] = (lambda k: lambda b: b[k:] + b[:k])(k)
        ops[f"ror{k}"] = (lambda k: lambda b: b[-k:] + b[:-k])(k)
        ops[f"shl{k}"] = (lambda k: lambda b: b[k:] + [0] * k)(k)
        ops[f"shr{k}"] = (lambda k: lambda b: [0] * k + b[:-k])(k)
    return ops


_BIT_OPS = _bit_op_library()


def _op_hypotheses(ins, outs, test) -> Optional[str]:
    def as_int(bits):
        return int("".join(str(x) for x in bits), 2)

    for op in _BIT_OPS.values():
        # op then XOR constant
        res = None
        consistent = True
        for i, o in zip(ins, outs):
            xr = [oi ^ p for oi, p in zip(o, op(i))]
            if res is None:
                res = xr
            elif res != xr:
                consistent = False
                break
        if consistent and res is not None:
            return "".join(str(t ^ r) for t, r in zip(op(test), res))
        # op then ADD constant mod 256
        c = None
        consistent = True
        for i, o in zip(ins, outs):
            d = (as_int(o) - as_int(op(i))) % 256
            if c is None:
                c = d
            elif c != d:
                consistent = False
                break
        if consistent and c is not None:
            return format((as_int(op(test)) + c) % 256, "08b")
    return None


def solve_bit_manipulation(prompt: str) -> Optional[str]:
    pairs = [
        (a, b)
        for a, b in _arrow_pairs(prompt)
        if re.fullmatch(r"[01]{8}", a) and re.fullmatch(r"[01]{8}", b)
    ]
    q = re.search(r"determine the output for:\s*([01]{8})", str(prompt))
    if not pairs or not q:
        return None
    ins_aug = [_bits(a) + [1] for a, _ in pairs]
    outs = [_bits(b) for _, b in pairs]
    test_aug = _bits(q.group(1)) + [1]

    # candidate from explicit op library (preferred: most interpretable)
    op_ins = [_bits(a) for a, _ in pairs]
    op_pred = _op_hypotheses(op_ins, outs, _bits(q.group(1)))
    aff_pred = _affine_predict(ins_aug, outs, test_aug)

    # leave-one-out agreement filter: a hypothesis must also reconstruct a held
    # out example to be trusted (guards against nonlinear rules masquerading).
    def loo_ok(predictor) -> bool:
        if len(pairs) < 3:
            return True
        for hold in range(len(pairs)):
            sub_in = [op_ins[i] for i in range(len(pairs)) if i != hold]
            sub_out = [outs[i] for i in range(len(pairs)) if i != hold]
            got = predictor(sub_in, sub_out, op_ins[hold])
            if got is None or got != "".join(str(x) for x in outs[hold]):
                return False
        return True

    if op_pred is not None and loo_ok(_op_hypotheses):
        return op_pred

    def aff_predictor(sub_in, sub_out, tst):
        si = [r + [1] for r in sub_in]
        return _affine_predict(si, sub_out, tst + [1])

    if aff_pred is not None and loo_ok(aff_predictor):
        return aff_pred
    return None


# ----------------------------------------------------------------------------
# equation: per-row secret operator semantics over a symbol alphabet. We try a
# small library of operand-level operations keyed by the middle operator char.
# This only covers the simplest rows (e.g. concatenation); the family is flagged.
# ----------------------------------------------------------------------------
def _eq_examples(prompt: str) -> List[Tuple[str, str]]:
    pairs = []
    for line in str(prompt).splitlines():
        if " = " in line and "->" not in line and "becomes" not in line:
            lhs, rhs = line.split(" = ", 1)
            pairs.append((lhs.strip(), rhs.rstrip()))
    return pairs


def solve_equation(prompt: str) -> Optional[str]:
    pairs = _eq_examples(prompt)
    q = re.search(r"determine the result for:\s*(.+)$", str(prompt), re.M)
    if not pairs or not q:
        return None
    query = q.group(1).strip()
    if len(query) < 3:
        return None
    op_char = query[len(query) // 2] if len(query) % 2 == 1 else None

    # operand-level binary ops keyed by the assumed middle operator
    def split_mid(s: str):
        if len(s) % 2 == 0:
            return None
        m = len(s) // 2
        return s[:m], s[m], s[m + 1:]

    candidate_ops = {
        "concat": lambda a, b: a + b,
        "concat_rev": lambda a, b: b + a,
        "interleave": lambda a, b: "".join(x + y for x, y in zip(a, b)) + a[len(b):] + b[len(a):],
    }
    # infer which named op is consistent for the query's operator char
    for name, fn in candidate_ops.items():
        consistent = True
        used = False
        for lhs, rhs in pairs:
            parts = split_mid(lhs)
            if not parts:
                consistent = False
                break
            a, oc, b = parts
            if op_char is not None and oc != op_char:
                continue  # only learn from same-operator examples
            used = True
            try:
                if fn(a, b) != rhs:
                    consistent = False
                    break
            except Exception:
                consistent = False
                break
        if consistent and used:
            parts = split_mid(query)
            if parts:
                a, _, b = parts
                try:
                    return fn(a, b)
                except Exception:
                    return None
    return None


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
SOLVERS: Dict[str, Callable[[str], Optional[str]]] = {
    "gravity": solve_gravity,
    "unit_conversion": solve_unit_conversion,
    "numeral": solve_numeral,
    "encryption": solve_encryption,
    "bit_manipulation": solve_bit_manipulation,
    "equation": solve_equation,
}


def solve(prompt: str, family: Optional[str] = None, **kwargs) -> Optional[str]:
    fam = family or classify_family(prompt)
    fn = SOLVERS.get(fam)
    if fn is None:
        return None
    try:
        if fam == "encryption":
            return fn(prompt, vocab=kwargs.get("vocab"))
        return fn(prompt)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Self-validation CLI
# ----------------------------------------------------------------------------
def _main() -> None:
    import argparse
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import pandas as pd
    from scripts.utils.competition_metric import scores

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    df["family"] = df["prompt"].map(classify_family)
    enc_vocab = build_encryption_vocab(
        df[df.family == "encryption"]["prompt"].tolist()
    )

    print(
        f"{'family':18s} {'n':>5s} {'solved':>7s} {'rate':>7s} "
        f"{'correct':>8s} {'acc':>7s}"
    )
    overall_correct = 0
    for fam in FAMILIES:
        sub = df[df.family == fam]
        n = len(sub)
        solved = correct = 0
        for _, row in sub.iterrows():
            pred = solve(row["prompt"], fam, vocab=enc_vocab)
            if pred is not None:
                solved += 1
                if scores(str(row["answer"]), str(pred)):
                    correct += 1
        overall_correct += correct
        flag = "" if correct == n else "  <-- FLAG (<100%)"
        print(
            f"{fam:18s} {n:5d} {solved:7d} {solved / n:7.1%} "
            f"{correct:8d} {correct / n:7.1%}{flag}"
        )
    print(f"\nOverall solver accuracy on train: {overall_correct}/{len(df)} "
          f"= {overall_correct / len(df):.1%}")


if __name__ == "__main__":
    _main()
