"""How many Yelp-polarity TEST reviews are short enough for the auto_LiRPA pipeline (<= 8/10/12 word-piece tokens incl. CLS/SEP)
and correctly classified by yelp_bert_small_3 / _6?  CPU only.  Labels: csv 1 -> 0 (negative), 2 -> 1 (positive) (DeepT: label-1)."""
import sys, os, csv, torch; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deept_gauge import load_deept, DeepTNet, embed, positions, DT
rows = list(csv.reader(open(os.path.join(DT, "..", "data", "yelp", "test.csv"))))
import re
def words(t): return re.findall(r"\w+|[^\w\s]", t.replace("\\n", " ").replace('\\"', '"'))   # nltk is not in the venv; BERT's basic tokenizer re-splits punctuation anyway
data = [{"label": int(l) - 1, "sent_a": words(t)} for l, t in rows]
short = [ex for ex in data if len(ex["sent_a"]) <= 14]
print(f"{len(data)} test reviews; {len(short)} with <= 14 words")
for name in sys.argv[1:] or ["yelp_bert_small_3"]:
    m, tok = load_deept(name); net = DeepTNet(m).eval(); cnt = {8: [0, 0], 10: [0, 0], 12: [0, 0]}
    for ex in short:
        e, toks = embed(m, tok, ex)
        if not positions(toks): continue
        with torch.no_grad(): ok = net(e).argmax(1).item() == ex["label"]
        for L in cnt:
            if e.shape[1] <= L: cnt[L][0] += 1; cnt[L][1] += ok
    print(name, {L: f"{c[0]} short, {c[1]} correct" for L, c in cnt.items()})
