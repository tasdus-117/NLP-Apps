"""Option A: raw English news to PERSON/ORG/LOCATION; shared token inference."""
import os
import sys
sys.dont_write_bytecode = True
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
import json
import re
import unicodedata
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from baseline_ner import ROOT, RUNTIME, ART, CACHE, LABELS, align_labels, chunk_words, load_data, normalize_bio, spans, write_json
CHECKPOINT = ART/'checkpoint'
MAX_LENGTH = 256
SCHEMA = {'PER':'PERSON','ORG':'ORG','LOC':'LOCATION'}

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_rows(rows, tokenizer, max_length=MAX_LENGTH):
    records, mapping = [], []
    for sid, row in enumerate(rows):
        for start, end in chunk_words(row['words'], tokenizer, max_length):
            enc = tokenizer(row['words'][start:end], is_split_into_words=True, truncation=False)
            wids = enc.word_ids()
            assert {w for w in wids if w is not None} == set(range(end - start))
            enc['labels'] = align_labels(wids, [LABELS.index(t) for t in row['tags'][start:end]])
            records.append(dict(enc))
            mapping.append((sid, start, end, wids))
    return records, mapping


@torch.inference_mode()
def predict_rows(model, tokenizer, rows, batch_size=32, max_length=MAX_LENGTH):
    records, mapping = encode_rows(rows, tokenizer, max_length)
    result = [[None] * len(r['words']) for r in rows]
    collator = DataCollatorForTokenClassification(tokenizer)
    model.eval()
    device = next(model.parameters()).device
    for offset in range(0, len(records), batch_size):
        batch = collator(records[offset:offset + batch_size])
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        logits = model(**inputs).logits
        ids = logits.argmax(-1).cpu().tolist()
        for j, values in enumerate(ids):
            sid, start, end, wids = mapping[offset + j]
            previous = None
            for wid, tag in zip(wids, values):
                if wid is not None and wid != previous:
                    result[sid][start + wid] = LABELS[tag]
                previous = wid
    assert all(all(t is not None for t in row) for row in result)
    return result


def core_demo(model, tokenizer, rows):
    records, mapping = encode_rows(rows[:2], tokenizer)
    batch = DataCollatorForTokenClassification(tokenizer)(records[:2])
    model.eval()
    with torch.inference_mode():
        out = model(**{k: v.to(next(model.parameters()).device) for k, v in batch.items()})
    return dict(tokens=tokenizer.convert_ids_to_tokens(batch['input_ids'][0].tolist()),
                input_ids=batch['input_ids'].tolist(), attention_mask=batch['attention_mask'].tolist(),
                word_ids=mapping[0][3], labels=batch['labels'].tolist(),
                logits=out.logits[0].cpu().tolist(), predictions=out.logits.argmax(-1).cpu().tolist(),
                shapes={k: list(v.shape) for k, v in batch.items()},
                logits_shape=list(out.logits.shape), loss=float(out.loss))

def raw_words(text):
    # A transparent tokenizer, independent of gold; abbreviations/punctuation can differ from CoNLL.
    matches = [m for m in re.finditer(r"\w+(?:[-'’]\w+)*|[^\w\s]", text, flags=re.UNICODE)
               if not all(unicodedata.category(c).startswith('C') or c == '\ufffd' for c in m.group())]
    return [m.group() for m in matches], [(m.start(), m.end()) for m in matches]


def read_raw_file(path):
    with open(path, encoding='utf-8', newline='') as f:
        return f.read()


def to_json(text, words, offsets, tags):
    if not len(words) == len(offsets) == len(tags):
        raise ValueError('words/offsets/tags must align')
    out = dict(text=text, PERSON=[], ORG=[], LOCATION=[])
    for start, end, typ in sorted(spans(tags)):
        if typ in SCHEMA:
            a, b = offsets[start][0], offsets[end - 1][1]
            out[SCHEMA[typ]].append(dict(text=text[a:b], start=a, end=b))
    out['word_predictions'] = [dict(word=w, start=a, end=b, tag=t)
                               for w, (a, b), t in zip(words, offsets, normalize_bio(tags))]
    return out


class Extractor:
    def __init__(self, checkpoint=ART / 'checkpoint'):
        seed_all()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint, local_files_only=True)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        if [self.model.config.id2label[i] for i in range(len(LABELS))] != LABELS:
            raise ValueError('Checkpoint label mapping differs from LAB 09')

    def extract(self, text):
        words, offsets = raw_words(text)
        tags = predict_rows(self.model, self.tokenizer, [dict(words=words, tags=['O'] * len(words))])[0] if words else []
        return to_json(text, words, offsets, tags)


def char_spans(output):
    return {(e['start'], e['end'], typ) for typ, key in SCHEMA.items() for e in output[key]}


def evaluate(extractor, count=30):
    """Gold from first 30 official test rows fixed before inference; same-domain reuse."""
    rows = load_data()['test'][:count]
    examples, tp, fp, fn = [], 0, 0, 0
    for row in rows:
        text = ' '.join(row['words'])
        offsets, cursor = [], 0
        for word in row['words']:
            offsets.append((cursor, cursor + len(word)))
            cursor += len(word) + 1
        gold = {(offsets[a][0], offsets[b - 1][1], typ) for a, b, typ in spans(row['tags']) if typ in SCHEMA}
        output = extractor.extract(text)
        predicted = char_spans(output)
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        examples.append(dict(split='test', id=row['id'], gold=[list(x) for x in sorted(gold)], output=output))
    p, r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    result = dict(sentences=len(rows), tp=tp, fp=fp, fn=fn, support=tp + fn,
                  precision=p, recall=r, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                  source='CoNLL-2003 official test first 30 rows; same-domain, overlaps main test; no tuning',
                  gold='Official corpus annotation projected onto space-joined words; independent of predictions',
                  metric='exact character span end-exclusive + type; MISC excluded in gold and prediction',
                  limitations='Reconstructed whitespace, not original article spacing; raw regex wordization differs; no independent out-of-domain claim')
    write_json(ART / 'ie_evaluation.json', result)
    write_json(ART / 'ie_examples.json', examples)
    print(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text', default='Barack Obama visited Paris.')
    p.add_argument('--input-file', type=str)
    p.add_argument('--checkpoint', default=str(CHECKPOINT))
    p.add_argument('--output', default=str(ART / 'extraction.json'))
    p.add_argument('--evaluate', action='store_true')
    args = p.parse_args()
    extractor = Extractor(args.checkpoint)
    if args.evaluate:
        evaluate(extractor)
    text = read_raw_file(args.input_file) if args.input_file else args.text
    result = extractor.extract(text)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
