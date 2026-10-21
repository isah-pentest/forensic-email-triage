#!/usr/bin/env python3
"""Rebuild the 10k/50k/100k Enron subsets, weak labels and metadata from the public
CMU Enron maildir, using the released split identifiers.

The original sampling (random.sample over os.walk order) is not portable, but every
message id is SHA-1(first 50,000 chars of the first 20,000 chars of the file, read as
UTF-8 with errors ignored), so the exact messages can be recovered by hashing every
file. Code below is ported verbatim from the notebook cells "Cell 5" (labelling)
and "Step 3A" (metadata).

Usage:
  python code/rebuild_subsets_from_maildir.py --maildir /path/to/enron/maildir --out /path/to/project
Writes <out>/data/{scale}_full.csv, <out>/data/{scale}_metadata_v1.csv and copies
<out>/frozen_release_v1/{splits,rules}, i.e. the layout code/limitations_upgrade.py expects
(run it with FET_ROOT=<out>). Self-checks compare labels with the released split metadata
and the released 100k rule-trigger file; any mismatch stops the script.
"""
import argparse, hashlib, os, re, shutil, sys
from email.utils import parsedate_to_datetime
from pathlib import Path
import pandas as pd, yaml

REPO = Path(__file__).resolve().parents[1]
SCALES = ["Enron_10000", "Enron_50000", "Enron_100000"]
RULES = yaml.safe_load(open(REPO / "rules" / "phishing_rules_v2.yaml"))
MAX_CHARS = 20000

# ---------------- notebook "Cell 5" -------------------------------------------------
def stable_id_from_message(msg):
    if not isinstance(msg, str): msg = ""
    return hashlib.sha1(msg[:50000].encode("utf-8", errors="ignore")).hexdigest()
def extract_subject(text):
    m = re.search(r"(?im)^subject:\s*(.*)$", text or ""); return m.group(1).strip() if m else ""
def extract_body(text):
    t = text or ""; parts = t.split("\n\n", 1); return parts[1] if len(parts) > 1 else t
def get_header_value(message, header):
    if not isinstance(message, str): return ""
    m = re.search(rf"(?im)^{header}:\s*(.*)$", message); return m.group(1).strip() if m else ""
def extract_addresses(v):
    if not isinstance(v, str): return []
    return re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", v, flags=re.I)
def has_external_party(message):
    addrs = []
    for h in ["From", "To", "Cc", "Bcc"]: addrs.extend(extract_addresses(get_header_value(message, h)))
    if not addrs: return "@enron.com" not in (message or "").lower()
    return any(not a.lower().endswith("@enron.com") for a in addrs)
def score_phishing(text, rules, raw=False):
    if not isinstance(text, str) or not text.strip(): return 0
    t = text.lower(); w = rules["weights"]
    core = any(k in t for k in rules["core_indicators"]); urg = any(k in t for k in rules["urgency_indicators"])
    act = any(k in t for k in rules["action_indicators"])
    url = re.search(rules["regex_indicators"]["url"], t) is not None
    sh = re.search(rules["regex_indicators"]["url_shorteners"], t) is not None
    s = w["core"]*core + w["urgency"]*urg + w["action"]*act + w["url"]*url + w["url_shortener"]*sh
    if raw: return s
    if (not core) and any(k in t for k in rules["benign_context_indicators"]) and s <= 2: return 0
    return s
def label_dataset(df):
    df = df.copy()
    df["subject"] = df["message"].apply(extract_subject); df["body"] = df["message"].apply(extract_body)
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    thr = RULES["thresholds"]["phish_score_threshold"]
    df["label3_v3"] = [2 if score_phishing(t, RULES) >= thr else (1 if has_external_party(m) else 0)
                       for t, m in zip(df.text, df.message)]
    df["y_bin"] = (df["label3_v3"] == 2).astype(int)
    return df

# ---------------- notebook "Step 3A" ------------------------------------------------
def is_internal_address(a): return isinstance(a, str) and a.lower().endswith("@enron.com")
def tokenize_len(t): return 0 if not isinstance(t, str) or not t.strip() else len(t.split())
def has_attachment_mention(t):
    if not isinstance(t, str): return 0
    t = t.lower(); return int(any(p in t for p in ["attached", "attachment", "attached file", "see attached", "enclosed"]))
def extract_metadata_features(df):
    rows = []
    for _, r in df.iterrows():
        message, subject, body, text = r["message"], r["subject"], r["body"], r["text"]
        fr, to, cc, bcc = (extract_addresses(get_header_value(message, h)) for h in ["From", "To", "Cc", "Bcc"])
        recips = to + cc + bcc; parties = fr + recips
        try: dt = parsedate_to_datetime(get_header_value(message, "Date"))
        except Exception: dt = None
        if dt is not None:
            hour, dow = int(dt.hour), int(dt.weekday()); wkend = int(dow >= 5); bh = int(8 <= hour <= 18 and dow < 5)
        else:
            hour, dow, wkend, bh = -1, -1, 0, 0
        sl = len(subject) if isinstance(subject, str) else 0; bl = len(body) if isinstance(body, str) else 0
        btl = tokenize_len(body); subj_lower = subject.lower() if isinstance(subject, str) else ""
        rows.append({"id": r["id"], "label3_v3": r["label3_v3"], "y_bin": r["y_bin"],
            "is_internal_sender": int(any(is_internal_address(a) for a in fr)),
            "is_external_sender": int(len(fr) > 0 and not all(is_internal_address(a) for a in fr)),
            "num_to_recipients": len(to), "num_cc_recipients": len(cc), "num_bcc_recipients": len(bcc),
            "has_cc": int(len(cc) > 0), "has_bcc": int(len(bcc) > 0),
            "is_reply_subject": int(subj_lower.startswith("re:")),
            "is_forward_subject": int(subj_lower.startswith("fw:") or subj_lower.startswith("fwd:")),
            "num_recipient_domains": len({a.split("@")[-1].lower() for a in recips if "@" in a}),
            "is_internal_only_thread": int(len(parties) > 0 and all(is_internal_address(a) for a in parties)),
            "is_mixed_domain_thread": int(any(is_internal_address(a) for a in parties) and any(not is_internal_address(a) for a in parties)),
            "subject_char_len": sl, "body_char_len": bl, "subject_token_len": tokenize_len(subject), "body_token_len": btl,
            "body_to_subject_len_ratio": float(bl) / float(max(sl, 1)), "has_empty_subject": int(sl == 0),
            "has_short_body": int(btl < 20), "hour_of_day": hour, "is_business_hours": bh, "day_of_week": dow,
            "is_weekend": wkend, "has_attachment_mention": has_attachment_mention(text),
            "has_multiple_recipients": int(len(recips) > 1)})
    return pd.DataFrame(rows)

# ---------------- rebuild -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--maildir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out)
    splits = {s: {sp: pd.read_csv(REPO / "splits" / s / f"{sp}_ids.csv").id.tolist() for sp in ["train", "val", "test"]} for s in SCALES}
    wanted = set().union(*[set(i) for s in splits.values() for i in s.values()])
    found, nfiles = {}, 0
    for root, _, files in os.walk(a.maildir):
        for fn in files:
            nfiles += 1
            try:
                with open(os.path.join(root, fn), "r", encoding="utf-8", errors="ignore") as f: msg = f.read()[:MAX_CHARS]
            except Exception:
                msg = ""
            h = stable_id_from_message(msg)
            if h in wanted and h not in found: found[h] = msg
    print(f"hashed {nfiles} files (notebook reported 517401); recovered {len(found)}/{len(wanted)} released ids")
    if nfiles != 517401: print("WARNING: file count differs from the original maildir")
    missing = wanted - set(found)
    if missing:
        sys.exit(f"{len(missing)} released ids not found: is this the same Enron maildir release (CMU, May 2015)?")
    (out / "data").mkdir(parents=True, exist_ok=True)
    for s in SCALES:
        order = splits[s]["train"] + splits[s]["val"] + splits[s]["test"]
        df = label_dataset(pd.DataFrame({"id": order, "message": [found[i] for i in order]}))
        meta = yaml.safe_load(open(REPO / "splits" / s / "split_meta.yaml"))
        for sp in ["train", "val", "test"]:
            got = df[df.id.isin(set(splits[s][sp]))].label3_v3.value_counts().to_dict()
            if {int(k): int(v) for k, v in got.items()} != {int(k): int(v) for k, v in meta[f"label_distribution_{sp}"].items()}:
                sys.exit(f"{s}/{sp}: label distribution {got} != released {meta[f'label_distribution_{sp}']}")
        if s == "Enron_100000":
            rt = pd.read_csv(REPO / "results" / "rule_triggers_100k.csv")[["id", "rule_score", "y_bin"]].set_index("id")
            raw = pd.Series([score_phishing(t, RULES, raw=True) for t in df.text], index=df.id)
            if not (raw.loc[rt.index] == rt.rule_score).all() or not (df.set_index("id").y_bin.loc[rt.index] == rt.y_bin).all():
                sys.exit("100k rule scores/labels differ from results/rule_triggers_100k.csv")
        df[["id", "message", "subject", "body", "text", "label3_v3", "y_bin"]].to_csv(out / "data" / f"{s}_full.csv", index=False)
        extract_metadata_features(df).to_csv(out / "data" / f"{s}_metadata_v1.csv", index=False)
        print(f"{s}: rebuilt {len(df)} messages; label distributions match the released split metadata")
    for sub in ["splits", "rules"]:
        shutil.copytree(REPO / sub, out / "frozen_release_v1" / sub, dirs_exist_ok=True)
    print(f"done. Next: FET_ROOT={out} python code/limitations_upgrade.py")

if __name__ == "__main__":
    main()
