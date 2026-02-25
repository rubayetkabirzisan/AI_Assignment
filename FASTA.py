# FASTA.py
# AI CT-2 Replacement Assignment
# BioPython + Sequence Analysis + Pattern Analysis + ML (Train/Test/Validation + Loss)
#
# ✅ Updates added (as you requested):
# 1) Read FASTA file (already)
# 2) Mutation analysis:
#    - "WHERE" mutation occurs inside flanking sequence (left/center/right)
#    - Mutation pattern (SNP/INS/DEL + transition/transversion)
#    - Mutation & targeted disease correlation (CSV outputs)
# 3) Simple grammar rules (1–2 rules; optional 3rd tuned threshold)
# 4) Train/Test those rules (accuracy ~50% ok)
# 5) Validation with loss function (log loss)

import os
import re
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from itertools import product

import numpy as np
import pandas as pd

from Bio import SeqIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss
)

BASES = "ACGT"
TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}


# -----------------------------
# NEW: helpers for "WHERE"
# -----------------------------
def get_variant_pos(row: pd.Series) -> Optional[int]:
    """
    Tries to find the genomic position column in the TSV.
    Common columns: start, pos, position (case-insensitive).
    Returns an int position (typically 1-based in VCF-like files).
    """
    # Try exact common names first
    for col in ["start", "pos", "position"]:
        if col in row and pd.notna(row[col]):
            try:
                return int(row[col])
            except Exception:
                pass

    # Try case-insensitive match
    lower_map = {c.lower(): c for c in row.index}
    for key in ["start", "pos", "position"]:
        if key in lower_map:
            col = lower_map[key]
            if pd.notna(row[col]):
                try:
                    return int(row[col])
                except Exception:
                    pass
    return None


def mutation_where_category(rel_index: Optional[int], seq_len: int) -> Optional[str]:
    """
    Categorize where mutation lies in the flanking sequence.
    rel_index is 0-based within the flanking sequence.
    """
    if rel_index is None or seq_len <= 0:
        return None
    # If rel_index is outside, mark unknown (prevents misleading "where")
    if rel_index < 0 or rel_index >= seq_len:
        return "unknown"

    frac = rel_index / seq_len
    if frac < 1 / 3:
        return "left"
    elif frac < 2 / 3:
        return "center"
    else:
        return "right"


# -----------------------------
# NEW: Rule-based "grammar"
# -----------------------------
def grammar_predict(ref: str, alt: str, tt: Optional[str], gc: float, gc_thresh: Optional[float] = None) -> int:
    """
    Rule-based classifier ("grammar").
    Returns 1 (Pathogenic) or 0 (Benign).

    Base Grammar (2 rules):
    1) If INS or DEL => Pathogenic
    2) Else if SNP and transversion => Pathogenic
       Else => Benign

    Optional tuned add-on rule (only if gc_thresh is not None):
    3) If GC >= gc_thresh => Pathogenic
    """
    ref = str(ref).upper()
    alt = str(alt).upper()

    # Rule 1: Indels tend to be more disruptive
    if len(ref) != len(alt):
        return 1

    # Rule 2: transversion SNP
    if tt == "transversion":
        return 1

    # Optional: GC threshold rule (tiny "training" via threshold search)
    if gc_thresh is not None and gc >= gc_thresh:
        return 1

    return 0


def prob_from_rule(pred: int) -> float:
    """
    Convert a hard rule decision into a probability for log-loss.
    Avoids 0/1 probabilities which can explode log_loss.
    """
    return 0.7 if pred == 1 else 0.3


@dataclass
class BedRow:
    chrom: str
    start0: int  # BED start (0-based)
    end: int
    name: str    # contains variant_id|gene|sigShort
    score: str
    strand: str


def parse_bed(path: str) -> List[BedRow]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            # BED here has 6 columns
            if len(parts) < 6:
                continue
            chrom, start0, end, name, score, strand = parts[:6]
            rows.append(BedRow(chrom=chrom, start0=int(start0), end=int(end),
                               name=name, score=score, strand=strand))
    return rows


def parse_fasta_regions(path: str) -> Dict[Tuple[str, int, int], str]:
    """
    FASTA headers look like: chr1:6425049-6425349
    We'll store region key as (chrom, start1, end).
    """
    region_to_seq = {}
    for record in SeqIO.parse(path, "fasta"):
        header = str(record.id)  # e.g., chr1:6425049-6425349
        m = re.match(r"(chr[\w]+):(\d+)-(\d+)", header)
        if not m:
            continue
        chrom = m.group(1)
        start1 = int(m.group(2))
        end = int(m.group(3))
        seq = str(record.seq).upper()
        region_to_seq[(chrom, start1, end)] = seq
    return region_to_seq


def bed_variant_to_region(bed_rows: List[BedRow]) -> Dict[int, Tuple[str, int, int]]:
    """
    Convert BED row to mapping: variant_id -> (chrom, start1, end)
    Note: BED start is 0-based, FASTA start is 1-based => start1 = start0 + 1
    """
    mapping = {}
    for r in bed_rows:
        variant_id = int(r.name.split("|")[0])
        mapping[variant_id] = (r.chrom, r.start0 + 1, r.end)
    return mapping


def seq_base_features(seq: str) -> np.ndarray:
    seq = seq.upper()
    L = len(seq)
    counts = {b: seq.count(b) for b in BASES}
    gc = (counts["G"] + counts["C"]) / L if L else 0.0
    return np.array([
        L,
        counts["A"] / L,
        counts["C"] / L,
        counts["G"] / L,
        counts["T"] / L,
        gc
    ], dtype=np.float32)


def kmer_3mer_features(seq: str, kmer_index: Dict[str, int]) -> np.ndarray:
    seq = seq.upper()
    L = len(seq)
    vec = np.zeros(len(kmer_index), dtype=np.float32)
    denom = max(L - 2, 1)
    for i in range(L - 2):
        k = seq[i:i + 3]
        if set(k) <= set(BASES):
            vec[kmer_index[k]] += 1.0
    vec /= denom
    return vec


def mutation_features(ref: str, alt: str) -> np.ndarray:
    ref = ref.upper()
    alt = alt.upper()
    is_snp = int(len(ref) == 1 and len(alt) == 1)
    is_ins = int(len(ref) < len(alt))
    is_del = int(len(ref) > len(alt))
    length_change = len(alt) - len(ref)

    is_transition = int(is_snp and (ref, alt) in TRANSITIONS)
    is_transversion = int(is_snp and not is_transition)

    return np.array([
        is_snp,
        is_ins,
        is_del,
        length_change,
        is_transition,
        is_transversion
    ], dtype=np.float32)


def label_pathogenicity(sig: str) -> Optional[int]:
    patho = {"Pathogenic", "Likely Pathogenic", "Pathogenic/Likely Pathogenic"}
    benign = {"Benign", "Likely Benign", "Benign/Likely Benign"}
    if sig in patho:
        return 1
    if sig in benign:
        return 0
    return None


def ensure_outputs_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def main():
    # ====== INPUT FILES (edit if your filenames differ) ======
    FASTA_FILE = "flanking_sequences.fasta"
    TSV_FILE = "deafness_vcf.tsv"
    BED_FILE = "deafness_flanking.bed"

    OUT_DIR = "outputs"
    ensure_outputs_dir(OUT_DIR)

    # ====== LOAD DATA ======
    print("Loading BED...")
    bed_rows = parse_bed(BED_FILE)
    vid_to_region = bed_variant_to_region(bed_rows)

    print("Loading FASTA (BioPython)...")
    region_to_seq = parse_fasta_regions(FASTA_FILE)

    print("Loading TSV...")
    df = pd.read_csv(TSV_FILE, sep="\t")
    df["variant_id"] = df["variant_id"].astype(int)

    # ====== JOIN VARIANTS -> SEQUENCE ======
    def get_seq_for_variant(vid: int) -> Optional[str]:
        region = vid_to_region.get(vid)
        if region is None:
            return None
        return region_to_seq.get(region)

    df["sequence"] = df["variant_id"].apply(get_seq_for_variant)
    df_joined = df.dropna(subset=["sequence"]).copy()

    print(f"Variants total: {len(df)}")
    print(f"Variants with sequence found: {len(df_joined)}")

    # ============================================================
    # NEW: MUTATION "WHERE" INSIDE FLANKING SEQUENCE
    # ============================================================
    # We infer genomic variant position from TSV (start/pos/position)
    # region start is from FASTA header key (chrom, start1, end)
    # rel_index = variant_pos - region_start1 (0-based within sequence)
    print("Computing mutation 'where' (left/center/right) within flanking sequence...")

    df_joined["variant_pos"] = df_joined.apply(get_variant_pos, axis=1)

    def compute_rel_index(row: pd.Series) -> Optional[int]:
        vid = int(row["variant_id"])
        region = vid_to_region.get(vid)
        if region is None:
            return None
        chrom, region_start1, region_end = region
        pos = row["variant_pos"]
        if pos is None or pd.isna(pos):
            return None
        # assumes pos is 1-based; if your TSV is 0-based, change to: int(pos + 1) - region_start1
        return int(pos) - int(region_start1)

    df_joined["rel_index"] = df_joined.apply(compute_rel_index, axis=1)
    df_joined["where"] = df_joined.apply(lambda r: mutation_where_category(r["rel_index"], len(r["sequence"])), axis=1)

    # ====== SEQUENCE + MUTATION SUMMARY ======
    # Mutation type counts
    def mut_type(ref: str, alt: str) -> str:
        if len(ref) == 1 and len(alt) == 1:
            return "SNP"
        if len(ref) < len(alt):
            return "INS"
        if len(ref) > len(alt):
            return "DEL"
        return "OTHER"

    df_joined["mut_type"] = [mut_type(r, a) for r, a in zip(df_joined["ref"], df_joined["alt"])]

    # Transition / transversion for SNP
    def snp_tt(ref: str, alt: str) -> Optional[str]:
        if len(ref) == 1 and len(alt) == 1:
            return "transition" if (ref.upper(), alt.upper()) in TRANSITIONS else "transversion"
        return None

    df_joined["tt"] = [snp_tt(r, a) for r, a in zip(df_joined["ref"], df_joined["alt"])]

    # Sequence composition
    def gc_content(seq: str) -> float:
        seq = seq.upper()
        return (seq.count("G") + seq.count("C")) / len(seq)

    df_joined["gc"] = df_joined["sequence"].apply(gc_content)

    # ----------------------------------------
    # NEW: mutation "where" distribution summary
    # ----------------------------------------
    where_counts = df_joined["where"].value_counts(dropna=False).to_dict()

    mutation_summary = {
        "total_variants_with_seq": int(len(df_joined)),
        "mutation_type_counts": df_joined["mut_type"].value_counts().to_dict(),
        "transition_transversion_counts (SNP only)": df_joined["tt"].value_counts(dropna=True).to_dict(),
        "mutation_where_counts (left/center/right/unknown)": where_counts,
        "gc_mean": float(df_joined["gc"].mean()),
        "gc_min": float(df_joined["gc"].min()),
        "gc_max": float(df_joined["gc"].max()),
    }

    # Save mutation summary
    mutation_summary_path = os.path.join(OUT_DIR, "mutation_summary.csv")
    with open(mutation_summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in mutation_summary.items():
            w.writerow([k, str(v)])

    # Save basic sequence summary (first 200 rows preview for readability)
    seq_summary_path = os.path.join(OUT_DIR, "sequence_summary.csv")
    preview_cols = [
        "variant_id", "chromosome", "start", "end", "ref", "alt",
        "clinical_significance", "molecular_consequence", "genes",
        "mut_type", "tt", "gc", "variant_pos", "rel_index", "where"
    ]
    # Keep only columns that exist in TSV (prevents crash if column name differs)
    preview_cols = [c for c in preview_cols if c in df_joined.columns]
    df_joined[preview_cols].head(200).to_csv(seq_summary_path, index=False)

    print("Saved:")
    print(" -", mutation_summary_path)
    print(" -", seq_summary_path)

    # ============================================================
    # NEW: MUTATION & DISEASE CORRELATION TABLES
    # ============================================================
    if "disease_names" in df_joined.columns:
        print("Generating disease correlation tables...")

        disease_mut = (
            df_joined.dropna(subset=["disease_names"])
            .groupby(["disease_names", "mut_type"])
            .size()
            .reset_index(name="count")
            .sort_values(["disease_names", "count"], ascending=[True, False])
        )
        disease_mut_path = os.path.join(OUT_DIR, "disease_mutation_type_counts.csv")
        disease_mut.to_csv(disease_mut_path, index=False)

        disease_tt = (
            df_joined.dropna(subset=["disease_names", "tt"])
            .groupby(["disease_names", "tt"])
            .size()
            .reset_index(name="count")
            .sort_values(["disease_names", "count"], ascending=[True, False])
        )
        disease_tt_path = os.path.join(OUT_DIR, "disease_transition_transversion_counts.csv")
        disease_tt.to_csv(disease_tt_path, index=False)

        disease_where = (
            df_joined.dropna(subset=["disease_names", "where"])
            .groupby(["disease_names", "where"])
            .size()
            .reset_index(name="count")
            .sort_values(["disease_names", "count"], ascending=[True, False])
        )
        disease_where_path = os.path.join(OUT_DIR, "disease_where_counts.csv")
        disease_where.to_csv(disease_where_path, index=False)

        print("Saved:")
        print(" -", disease_mut_path)
        print(" -", disease_tt_path)
        print(" -", disease_where_path)
    else:
        print("NOTE: 'disease_names' column not found in TSV, so disease correlation tables were skipped.")

    # ====== ML DATASET ======
    df_joined["y"] = df_joined["clinical_significance"].apply(label_pathogenicity)
    df_ml = df_joined.dropna(subset=["y"]).copy()
    df_ml["y"] = df_ml["y"].astype(int)

    print("\nML dataset size (after filtering uncertain/conflicting):", len(df_ml))
    print("Label counts:", df_ml["y"].value_counts().to_dict())

    # ============================================================
    # NEW: RULE-BASED GRAMMAR (Train/Val/Test + Loss)
    # ============================================================
    print("\nRunning rule-based grammar (1–2 rules) with train/val/test + loss...")

    df_ml = df_ml.reset_index(drop=True)

    train_idx, temp_idx = train_test_split(
        df_ml.index, test_size=0.30, random_state=42, stratify=df_ml["y"]
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=df_ml.loc[temp_idx, "y"]
    )

    df_train = df_ml.loc[train_idx].copy()
    df_val = df_ml.loc[val_idx].copy()
    df_test = df_ml.loc[test_idx].copy()

    # Choose whether to tune ONE threshold (GC threshold) for tiny "training"
    use_gc_tuning = True  # set False if teacher insists ONLY 2 rules, no tuning

    gc_thresh_best = None
    best_val_acc = -1.0
    best_val_loss = 1e9

    if use_gc_tuning:
        candidate_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        for thr in candidate_thresholds:
            preds = []
            probs = []
            y_true = df_val["y"].values

            for ref, alt, tt, gc in zip(df_val["ref"], df_val["alt"], df_val["tt"], df_val["gc"]):
                pred = grammar_predict(ref, alt, tt, gc, gc_thresh=thr)
                preds.append(pred)
                probs.append(prob_from_rule(pred))

            acc = accuracy_score(y_true, preds)
            loss = log_loss(y_true, probs)

            if (acc > best_val_acc) or (acc == best_val_acc and loss < best_val_loss):
                best_val_acc = acc
                best_val_loss = loss
                gc_thresh_best = thr

    def eval_grammar(df_split: pd.DataFrame, gc_thr: Optional[float]):
        y_true = df_split["y"].values
        preds = []
        probs = []
        for ref, alt, tt, gc in zip(df_split["ref"], df_split["alt"], df_split["tt"], df_split["gc"]):
            pred = grammar_predict(ref, alt, tt, gc, gc_thresh=gc_thr)
            preds.append(pred)
            probs.append(prob_from_rule(pred))
        acc = accuracy_score(y_true, preds)
        loss = log_loss(y_true, probs)
        cm_rule = confusion_matrix(y_true, preds)
        return acc, loss, cm_rule

    train_acc_r, train_loss_r, train_cm_r = eval_grammar(df_train, gc_thresh_best if use_gc_tuning else None)
    val_acc_r, val_loss_r, val_cm_r = eval_grammar(df_val, gc_thresh_best if use_gc_tuning else None)
    test_acc_r, test_loss_r, test_cm_r = eval_grammar(df_test, gc_thresh_best if use_gc_tuning else None)

    rule_report_lines = []
    rule_report_lines.append("=== RULE-BASED GRAMMAR RESULTS ===")
    rule_report_lines.append("Grammar rules:")
    rule_report_lines.append("  Rule 1: If INS or DEL => Pathogenic")
    rule_report_lines.append("  Rule 2: Else if SNP is transversion => Pathogenic")
    if use_gc_tuning:
        rule_report_lines.append(f"  Tuned extra rule: If GC >= {gc_thresh_best} => Pathogenic")
    else:
        rule_report_lines.append("  (No extra tuned rule; strict 2-rule grammar only)")
    rule_report_lines.append("")
    rule_report_lines.append(f"Train Accuracy: {train_acc_r:.4f} | Train Log Loss: {train_loss_r:.4f}")
    rule_report_lines.append(f"Val   Accuracy: {val_acc_r:.4f} | Val   Log Loss: {val_loss_r:.4f}")
    rule_report_lines.append(f"Test  Accuracy: {test_acc_r:.4f} | Test  Log Loss: {test_loss_r:.4f}")
    rule_report_lines.append("")
    rule_report_lines.append("Confusion Matrix (Test):")
    rule_report_lines.append(str(test_cm_r))

    rule_metrics_path = os.path.join(OUT_DIR, "rule_metrics.txt")
    with open(rule_metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rule_report_lines))

    print("Saved:", rule_metrics_path)

    # ============================================================
    # ORIGINAL ML: Logistic Regression (Train/Val/Test + Loss)
    # ============================================================

    # Build k-mer index
    kmers = ["".join(p) for p in product(BASES, repeat=3)]
    kmer_index = {k: i for i, k in enumerate(kmers)}

    # Feature matrix
    X_list = []
    for seq, ref, alt in zip(df_ml["sequence"], df_ml["ref"], df_ml["alt"]):
        base_feat = seq_base_features(seq)
        mut_feat = mutation_features(ref, alt)
        kmer_feat = kmer_3mer_features(seq, kmer_index)
        X_list.append(np.concatenate([base_feat, mut_feat, kmer_feat], axis=0))
    X = np.vstack(X_list)
    y = df_ml["y"].values

    # Train/Val/Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Scale + Model
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    model.fit(X_train_s, y_train)

    # Predictions
    val_proba = model.predict_proba(X_val_s)[:, 1]
    test_proba = model.predict_proba(X_test_s)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    # Metrics
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    val_auc = roc_auc_score(y_val, val_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    # Loss function (log loss / cross-entropy)
    val_loss = log_loss(y_val, val_proba)
    test_loss = log_loss(y_test, test_proba)

    cm = confusion_matrix(y_test, test_pred)

    metrics_text = []
    metrics_text.append("=== ML RESULTS (Pathogenic vs Benign) ===")
    metrics_text.append(f"Train size: {len(y_train)} | Val size: {len(y_val)} | Test size: {len(y_test)}")
    metrics_text.append("")
    metrics_text.append(f"Validation Accuracy: {val_acc:.4f}")
    metrics_text.append(f"Validation ROC-AUC : {val_auc:.4f}")
    metrics_text.append(f"Validation Log Loss (Cross-Entropy): {val_loss:.4f}")
    metrics_text.append("")
    metrics_text.append(f"Test Accuracy: {test_acc:.4f}")
    metrics_text.append(f"Test ROC-AUC : {test_auc:.4f}")
    metrics_text.append(f"Test Log Loss (Cross-Entropy): {test_loss:.4f}")
    metrics_text.append("")
    metrics_text.append("Confusion Matrix (Test):")
    metrics_text.append(str(cm))
    metrics_text.append("")
    metrics_text.append("Classification Report (Test):")
    metrics_text.append(classification_report(y_test, test_pred))

    ml_metrics_path = os.path.join(OUT_DIR, "ml_metrics.txt")
    with open(ml_metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metrics_text))

    print("\nSaved:", ml_metrics_path)

    # Save test predictions (for submission evidence)
    test_pred_path = os.path.join(OUT_DIR, "test_predictions.csv")
    pd.DataFrame({
        "y_true": y_test,
        "y_pred": test_pred,
        "y_prob_pathogenic": test_proba
    }).to_csv(test_pred_path, index=False)

    print("Saved:", test_pred_path)

    print("\nDONE ✅ Run complete. Check the outputs/ folder.")
    print("\nOutputs now include (at least):")
    print(" - mutation_summary.csv")
    print(" - sequence_summary.csv")
    print(" - rule_metrics.txt  ✅ (your grammar rules + train/val/test + loss)")
    print(" - ml_metrics.txt")
    print(" - test_predictions.csv")
    print("Plus disease correlation CSVs if disease_names exists.")


if __name__ == "__main__":
    main()