"""CoNLL baseline, shared BIO/scoring, pinned data preparation and active self-checks."""
import os
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get('LAB09_RUNTIME', str(ROOT.parent.parent / 'NLP_runtime/lab09'))).expanduser().resolve()
if RUNTIME == ROOT or ROOT in RUNTIME.parents:
    raise ValueError('LAB09_RUNTIME must be outside the submission directory lab09')
ART = RUNTIME / 'artifacts'
CACHE = RUNTIME / '.cache'
os.environ.setdefault('MPLCONFIGDIR', str(CACHE / 'matplotlib'))
os.environ.setdefault('HF_HOME', str(CACHE / 'huggingface'))
LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
TYPES = ['PER', 'ORG', 'LOC', 'MISC']

SOURCE_MANIFEST = {'dataset': {'repo': 'eriktks/conll2003', 'revision': 'ce85b39f9dd99f552d0739d456814e95fb6a39b0', 'files': [{'name': 'conll2003/test/0000.parquet', 'url': 'https://huggingface.co/datasets/eriktks/conll2003/resolve/ce85b39f9dd99f552d0739d456814e95fb6a39b0/conll2003/test/0000.parquet', 'bytes': 283307, 'sha256': '14f894263b4ee3563395a57b8ba4cfaabec7db4ea956ba34e2f84a19781cdc43'}, {'name': 'conll2003/train/0000.parquet', 'url': 'https://huggingface.co/datasets/eriktks/conll2003/resolve/ce85b39f9dd99f552d0739d456814e95fb6a39b0/conll2003/train/0000.parquet', 'bytes': 1227788, 'sha256': '546cbb4200311629804a65cbe19c8846f16ab1529c4da48bcc7be8c40daf29c6'}, {'name': 'conll2003/validation/0000.parquet', 'url': 'https://huggingface.co/datasets/eriktks/conll2003/resolve/ce85b39f9dd99f552d0739d456814e95fb6a39b0/conll2003/validation/0000.parquet', 'bytes': 311883, 'sha256': '114dec371f357e2fa397df14311891440d0ee1030b3a2fec4025ff84b875a59d'}]}, 'model': {'repo': 'distilbert/distilbert-base-cased', 'revision': '6ea81172465e8b0ad3fddeed32b986cdcdcffcf0', 'files': [{'name': 'config.json', 'url': 'https://huggingface.co/distilbert/distilbert-base-cased/resolve/6ea81172465e8b0ad3fddeed32b986cdcdcffcf0/config.json', 'bytes': 465, 'sha256': '165376e2c837d836c75ae30fde8c5569beb3ed7325b7473c18a76e4608cdec22'}, {'name': 'model.safetensors', 'url': 'https://huggingface.co/distilbert/distilbert-base-cased/resolve/6ea81172465e8b0ad3fddeed32b986cdcdcffcf0/model.safetensors', 'bytes': 263260784, 'sha256': '64080c2b1998c3bd1214311ab9f1e97b40052fa81d508b8b534a936c4ba1626b'}, {'name': 'tokenizer.json', 'url': 'https://huggingface.co/distilbert/distilbert-base-cased/resolve/6ea81172465e8b0ad3fddeed32b986cdcdcffcf0/tokenizer.json', 'bytes': 435797, 'sha256': '5b3360be30cdf39081153cf9d30bfb0503c011462fc9f77f097127c83ef0f4d9'}, {'name': 'tokenizer_config.json', 'url': 'https://huggingface.co/distilbert/distilbert-base-cased/resolve/6ea81172465e8b0ad3fddeed32b986cdcdcffcf0/tokenizer_config.json', 'bytes': 49, 'sha256': '0f6d13e6f4da6f9e24f22ada6bc3be571123d858d7c0c05a8a7cd55a9c23c2e8'}, {'name': 'vocab.txt', 'url': 'https://huggingface.co/distilbert/distilbert-base-cased/resolve/6ea81172465e8b0ad3fddeed32b986cdcdcffcf0/vocab.txt', 'bytes': 213450, 'sha256': 'eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02'}]}}

PREDICTION_INITIAL_SHA256 = '76196b06c5288a74335a0bb32edf4123edd2e40d3abc14f27889ef61dfb85244'

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def normalize_bio(tags):
    """IOB1 or invalid initial I-X -> B-X; preserve explicit adjacent B-X."""
    out, previous = [], 'O'
    for tag in tags:
        if tag != 'O' and (tag[:2] not in ('B-', 'I-') or tag[2:] not in TYPES):
            raise ValueError(f'Unknown BIO tag {tag}')
        if tag.startswith('I-') and previous not in ('B-' + tag[2:], 'I-' + tag[2:]):
            tag = 'B-' + tag[2:]
        out.append(tag)
        previous = tag
    return out


def spans(tags):
    """Return {(start word inclusive, end exclusive, type)} after documented repair."""
    tags = normalize_bio(tags)
    result, start, typ = set(), None, None
    for i, tag in enumerate(tags + ['O']):
        if tag == 'O' or tag.startswith('B-'):
            if start is not None:
                result.add((start, i, typ))
                start = None
        if tag.startswith('B-'):
            start, typ = i, tag[2:]
    return result


def length_group(start, end):
    return str(end - start) if end - start < 4 else '4+'


def score(gold, predicted, entity_type=None, length=None):
    if len(gold) != len(predicted):
        raise ValueError('Sentence count mismatch')
    tp = fp = fn = 0
    for g, p in zip(gold, predicted):
        if len(g) != len(p):
            raise ValueError('Word count mismatch')
        def keep(s):
            return (entity_type is None or s[2] == entity_type) and (length is None or length_group(*s[:2]) == length)
        gs, ps = ({s for s in spans(t) if keep(s)} for t in (g, p))
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                support=tp + fn, tp=tp, fp=fp, fn=fn)


def align_labels(word_ids, word_labels):
    """Assign a target only at each complete word's first subtoken."""
    out, previous = [], None
    for wid in word_ids:
        out.append(word_labels[wid] if wid is not None and wid != previous else -100)
        previous = wid
    return out


def chunk_words(words, tokenizer, max_length=256):
    """Greedy binary search: complete words only, no discarded continuation pieces."""
    chunks, start = [], 0
    while start < len(words):
        if len(tokenizer(words[start:], is_split_into_words=True, truncation=False)['input_ids']) <= max_length:
            chunks.append((start, len(words)))
            break
        low, high, best = start + 1, len(words), start
        while low <= high:
            end = (low + high) // 2
            n = len(tokenizer(words[start:end], is_split_into_words=True, truncation=False)['input_ids'])
            if n <= max_length:
                best, low = end, end + 1
            else:
                high = end - 1
        if best == start:
            raise ValueError(f'Word at {start} alone exceeds max_length={max_length}; refusing data loss')
        chunks.append((start, best))
        start = best
    return chunks


def load_data(inspect=True):
    """Read downloaded Parquet without running remote dataset Python scripts."""
    import pyarrow.parquet as pq
    data, report = {}, {}
    for split in ('train', 'validation', 'test'):
        files = sorted((CACHE / 'data').rglob(f'{split}/*.parquet'))
        if not files:
            raise FileNotFoundError('Run python -B baseline_ner.py prepare first')
        rows, changes = [], 0
        for file in files:
            table = pq.read_table(file)
            metadata = json.loads(table.schema.metadata[b'huggingface'])
            names = metadata['info']['features']['ner_tags']['feature']['names']
            if names != LABELS:
                raise ValueError(f'Unexpected label mapping: {names}')
            for raw in table.to_pylist():
                original = [names[i] for i in raw['ner_tags']]
                tags = normalize_bio(original)
                changes += sum(a != b for a, b in zip(original, tags))
                rows.append(dict(id=len(rows), source_id=raw.get('id'), words=raw['tokens'],
                                 tags=tags, original_tags=original))
        data[split] = rows
        report[split] = dict(sentences=len(rows), words=sum(len(r['words']) for r in rows),
                             iob_to_iob2_changes=changes)
    if inspect:
        write_json(ART / 'dataset_inspection.json', dict(splits=report, labels=LABELS,
               scheme='IOB2 after deterministic normalization; see change counts', sample=data['train'][0]))
    return data


def save_predictions(name, rows, predictions, split='test'):
    path = ART / f'{name}_{split}_predictions.jsonl'
    path.parent.mkdir(exist_ok=True)
    if len(rows) != len(predictions):
        raise ValueError('Sentence count mismatch')
    with path.open('w', encoding='utf-8') as f:
        for row, pred in zip(rows, predictions):
            if len(row['words']) != len(pred):
                raise ValueError('Word count mismatch')
            f.write(json.dumps(dict(id=row['id'], split=split, words=row['words'],
                         gold=row['tags'], pred=normalize_bio(pred)), ensure_ascii=False) + '\n')


def load_predictions(name, split='test'):
    return [json.loads(line) for line in (ART / f'{name}_{split}_predictions.jsonl').read_text(encoding='utf-8').splitlines()]

def features(words, i):
    w = words[i]
    f = dict(word=w, lower=w.lower(), title=w.istitle(), upper=w.isupper(),
             digit=w.isdigit(), prefix=w[:3].lower(), suffix=w[-3:].lower(),
             first=i == 0, last=i == len(words) - 1)
    for delta, prefix in [(-1, 'prev'), (1, 'next')]:
        j = i + delta
        if 0 <= j < len(words):
            f[prefix + '_lower'] = words[j].lower()
            f[prefix + '_title'] = words[j].istitle()
    return f


def predict(bundle, rows):
    vectorizer, classifier = bundle
    result = []
    for row in rows:
        x = sparse32(vectorizer.transform([features(row['words'], i) for i in range(len(row['words']))]))
        result.append(classifier.predict(x).tolist())
    return result


def make_vectorizer():
    return DictVectorizer(dtype=np.float32)


def sparse32(x):
    if max(x.shape + (x.nnz,)) >= np.iinfo(np.int32).max:
        raise ValueError('Sparse matrix too large for SAGA int32 indices')
    x.indices = x.indices.astype(np.int32)
    x.indptr = x.indptr.astype(np.int32)
    return x


def select_validation_candidate(trials):
    return max(range(len(trials)), key=lambda i: (trials[i]['validation']['f1'], -trials[i]['C']))


def prediction_core(text):
    """Fingerprint the three original predictions independently of editorial headings."""
    return '\n'.join(line for line in text.splitlines() if line.startswith(('1. **', '2. **', '3. **')))


def run_baseline():
    data = load_data()
    start = time.perf_counter()
    vectorizer = make_vectorizer()
    x = sparse32(vectorizer.fit_transform([features(r['words'], i) for r in data['train'] for i in range(len(r['words']))]))
    y = [tag for r in data['train'] for tag in r['tags']]
    trials, fitted = [], []
    for c in (0.5, 1.0, 2.0):
        classifier = LogisticRegression(C=c, solver='saga', max_iter=200, tol=1e-3, random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            classifier.fit(x, y)
        val_pred = predict((vectorizer, classifier), data['validation'])
        trials.append(dict(C=c, validation=score([r['tags'] for r in data['validation']], val_pred),
                           warnings=[str(w.message) for w in caught]))
        fitted.append(classifier)
        print('validation candidate:', trials[-1], flush=True)
    selected = select_validation_candidate(trials)
    classifier = fitted[selected]
    bundle = (vectorizer, classifier)
    ART.mkdir(exist_ok=True)
    joblib.dump(bundle, ART / 'baseline.joblib')
    elapsed = time.perf_counter() - start
    report = dict(model='baseline', seed=42, hyperparameters=classifier.get_params(),
                  training_seconds=elapsed, features=x.shape[1], train_words=x.shape[0],
                  iterations=classifier.n_iter_.tolist(), warnings=trials[selected]['warnings'],
                  validation_trials=trials, selected_C=trials[selected]['C'],
                  selection='C grid (0.5,1,2) fixed before new run; maximum validation entity F1; ties use smaller C; test only after selection',
                  timing='feature extraction + candidate training + validation + model save; excludes final validation/test inference', device='CPU')
    for split in ('validation', 'test'):
        pred = predict(bundle, data[split])
        save_predictions('baseline', data[split], pred, split)
        report[split] = score([r['tags'] for r in data[split]], pred)
    write_json(ART / 'baseline_run.json', report)
    print(report)
    return report

import unittest

class CoreTests(unittest.TestCase):
    def test_failed_notebook_rerun_preserves_saved_evidence(self):
        import tempfile
        import nbformat
        from unittest.mock import patch
        module = sys.modules[execute_notebooks.__module__]
        with tempfile.TemporaryDirectory(dir=CACHE) as folder:
            temporary = Path(folder)
            path = temporary/'tokenizer_alignment.ipynb'
            original = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('1 + 1')])
            nbformat.write(original, path)
            before = path.read_bytes()
            class FailingClient:
                def __init__(self, notebook, **kwargs):
                    notebook.cells[0].source = 'partial execution must not replace saved evidence'
                def execute(self):
                    raise RuntimeError('kernel failure fixture')
            with patch.object(module, 'ROOT', temporary), patch.object(module, 'RUNTIME', temporary/'runtime'), \
                 patch.dict(os.environ), patch('nbclient.NotebookClient', FailingClient):
                with self.assertRaisesRegex(RuntimeError, 'kernel failure fixture'):
                    execute_notebooks()
            self.assertEqual(path.read_bytes(), before)

    def test_validation_selection_ignores_test_and_breaks_ties_by_smaller_c(self):
        trials = [dict(C=2.0, validation=dict(f1=0.8), test_f1=0.99),
                  dict(C=1.0, validation=dict(f1=0.8), test_f1=0.1),
                  dict(C=0.5, validation=dict(f1=0.7), test_f1=1.0)]
        self.assertEqual(select_validation_candidate(trials), 1)

    def test_baseline_vectorizer(self):
        from baseline_ner import make_vectorizer, features, sparse32
        words = ['Alice', 'works']
        x = sparse32(make_vectorizer().fit_transform([features(words, i) for i in range(2)]))
        self.assertEqual(x.shape[0], 2)
        self.assertGreater(x.shape[1], 0)
        self.assertEqual(str(x.indices.dtype), 'int32')

    def test_iob_conversion_and_adjacent(self):
        self.assertEqual(normalize_bio(['I-PER', 'I-PER', 'O', 'I-LOC']),
                         ['B-PER', 'I-PER', 'O', 'B-LOC'])
        self.assertEqual(spans(['B-PER', 'B-PER']), {(0, 1, 'PER'), (1, 2, 'PER')})

    def test_exact_boundary_type_missing_spurious(self):
        g = [['B-LOC', 'I-LOC', 'O', 'B-PER']]
        for p, expected in [([['B-LOC', 'I-LOC', 'O', 'B-PER']], (2, 0, 0)),
                            ([['B-LOC', 'O', 'O', 'B-PER']], (1, 1, 1)),
                            ([['B-ORG', 'I-ORG', 'O', 'B-PER']], (1, 1, 1)),
                            ([['O', 'O', 'O', 'B-PER']], (1, 0, 1)),
                            ([['B-LOC', 'I-LOC', 'B-ORG', 'B-PER']], (2, 1, 0))]:
            r = score(g, p)
            self.assertEqual((r['tp'], r['fp'], r['fn']), expected)
        self.assertEqual(score([['O']], [['O']])['f1'], 0.0)

    def test_length_fp_fn(self):
        g, p = [['B-LOC', 'I-LOC']], [['B-LOC', 'O']]
        self.assertEqual(score(g, p, length='1')['fp'], 1)
        self.assertEqual(score(g, p, length='2')['fn'], 1)

    def test_alignment(self):
        self.assertEqual(align_labels([None, 0, 0, 1, None, None], [1, 0]),
                         [-100, 1, -100, 0, -100, -100])

    def test_raw_offsets_and_adjacent_entities(self):
        from information_extraction import raw_words, to_json
        text = 'Alice Bob visited Paris.'
        words, offsets = raw_words(text)
        self.assertEqual(words, ['Alice', 'Bob', 'visited', 'Paris', '.'])
        out = to_json(text, words, offsets, ['B-PER', 'B-PER', 'O', 'B-LOC', 'O'])
        self.assertEqual([x['text'] for x in out['PERSON']], ['Alice', 'Bob'])
        self.assertEqual(out['LOCATION'], [{'text': 'Paris', 'start': 18, 'end': 23}])

    def test_raw_ignores_standalone_format_controls(self):
        from information_extraction import raw_words
        text = '\ufeffBarack\u200b Obama visited Paris.\ufffd\ue000'
        words, offsets = raw_words(text)
        self.assertEqual(words, ['Barack', 'Obama', 'visited', 'Paris', '.'])
        self.assertEqual(offsets[0], (1, 7))
        self.assertEqual(offsets[1], (9, 14))
        self.assertTrue(all(text[a:b] == w for w, (a,b) in zip(words, offsets)))

    def test_input_file_preserves_crlf(self):
        import tempfile
        from pathlib import Path
        from baseline_ner import CACHE
        from information_extraction import read_raw_file
        with tempfile.TemporaryDirectory(dir=CACHE) as folder:
            path = Path(folder) / 'input.txt'
            path.write_bytes(b'Hello\r\nParis.')
            self.assertEqual(read_raw_file(path), 'Hello\r\nParis.')

    def test_chunking_covers_every_word(self):
        class Tokenizer:
            def __call__(self, words, **kwargs):
                return {'input_ids': [0] + [5] * (2 * len(words)) + [0]}
        self.assertEqual(chunk_words(['a'] * 7, Tokenizer(), 8), [(0, 3), (3, 6), (6, 7)])
        with self.assertRaises(ValueError):
            chunk_words(['a'], Tokenizer(), 3)

def prepare_data():
    """Download only the pinned public files in SOURCE_MANIFEST; verify existing hashes."""
    import requests
    for kind, directory in [('dataset', CACHE/'data'), ('model', CACHE/'backbone')]:
        for item in SOURCE_MANIFEST[kind]['files']:
            path = directory/item['name']
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                print('Downloading', item['url'], flush=True)
                temporary = path.with_suffix(path.suffix+'.part')
                with requests.get(item['url'], stream=True, timeout=(30,180)) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as output:
                        for block in response.iter_content(1024*1024):
                            output.write(block)
                temporary.replace(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise RuntimeError(f'Source hash mismatch: {path}')
    write_json(ART/'sources.json', SOURCE_MANIFEST)
    print({split:len(rows) for split,rows in load_data().items()})


def run_self_checks():
    import unittest
    CACHE.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    return result.testsRun


def execute_notebooks(train=False):
    """Each notebook gets a fresh kernel, using this interpreter and external runtime."""
    import nbformat
    from nbclient import NotebookClient
    os.environ['LAB09_RUNTIME'] = str(RUNTIME)
    os.environ['LAB09_TRAIN'] = '1' if train else '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['JUPYTER_PATH'] = str(RUNTIME/'jupyter')
    os.environ['IPYTHONDIR'] = str(CACHE/'ipython')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(CACHE/'jupyter')
    write_json(RUNTIME/'jupyter/kernels/lab09/kernel.json',
        dict(argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}'],
             display_name='Python (LAB09 runtime)', language='python'))
    for name in ['tokenizer_alignment.ipynb','ner_finetuning.ipynb']:
        path = ROOT/name
        nb = nbformat.read(path,as_version=4)
        NotebookClient(nb,timeout=1200,kernel_name='lab09',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.write(nb,path)
        print('Executed',name,flush=True)

def verify_submission(check_notebooks=True):
    import nbformat
    import pandas as pd
    from PIL import Image as PILImage
    from information_extraction import char_spans, to_json
    test_count = run_self_checks()
    required = {'doc','README.md','calculations.md','prediction.md','baseline_ner.py',
                'tokenizer_alignment.ipynb','ner_finetuning.ipynb','results.csv','entity_metrics.csv',
                'entity_length_analysis.png','confusion_matrix.png','error_analysis.md',
                'information_extraction.py','reflection.md'}
    assert {p.name for p in ROOT.iterdir()} == required, 'Unexpected or missing submission entries'
    for name in ('baseline_ner.py','information_extraction.py'):
        compile((ROOT/name).read_text(encoding='utf8'),name,'exec')
    data = load_data(inspect=False)
    results = pd.read_csv(ROOT/'results.csv')
    typed = pd.read_csv(ROOT/'entity_metrics.csv')
    lengths = pd.read_csv(ART/'length_metrics.csv', dtype={'length':str})
    hashes = {}
    for name in ('baseline','transformer'):
        rows = load_predictions(name)
        assert [(r['id'],r['words'],r['gold']) for r in rows] == [(r['id'],r['words'],r['tags']) for r in data['test']]
        gold,pred = [r['gold'] for r in rows],[r['pred'] for r in rows]
        assert sum(len(r['words']) for r in rows)==46435
        actual = score(gold,pred)
        saved = results[results.model==name].iloc[0]
        for key in ('precision','recall','f1','support','tp','fp','fn'):
            assert np.isclose(actual[key],saved[key]), (name,key)
        for _,r in typed[typed.model==name].iterrows():
            s=score(gold,pred,entity_type=r['type'])
            for key in ('precision','recall','f1','support','tp','fp','fn'):
                assert np.isclose(s[key],r[key])
        for _,r in lengths[lengths.model==name].iterrows():
            s=score(gold,pred,length=r['length'])
            for key in ('precision','recall','f1','support','tp','fp','fn'):
                assert np.isclose(s[key],r[key])
        cm = np.array(json.loads((ART/'confusion_counts.json').read_text())['matrices'][name])
        expected = np.zeros((9,9),dtype=int)
        for g,p in zip(gold,pred):
            for a,b in zip(g,p):
                expected[LABELS.index(a),LABELS.index(b)] += 1
        assert np.array_equal(cm,expected) and cm.sum()==46435
    run=json.loads((ART/'transformer_run.json').read_text())
    from baseline_ner import PREDICTION_INITIAL_SHA256
    prefix=prediction_core((ROOT/'prediction.md').read_text(encoding='utf8'))
    assert hashlib.sha256(prefix.encode()).hexdigest()==PREDICTION_INITIAL_SHA256
    assert run['main_experiment'] and run['train_sentences']==14041
    assert run['best_validation_f1']==max(e['validation']['f1'] for e in run['history'])
    assert run['best_epoch']==max(run['history'],key=lambda e:e['validation']['f1'])['epoch']
    source=json.loads((ART/'sources.json').read_text())
    for kind,folder in [('dataset',CACHE/'data'),('model',CACHE/'backbone')]:
        for item in source[kind]['files']:
            assert hashlib.sha256((folder/item['name']).read_bytes()).hexdigest()==item['sha256']
    errors=json.loads((ART/'error_examples.json').read_text())
    preds=load_predictions('transformer')
    counts={group:sum(e['group']==group for e in errors) for group in ['FP','FN','boundary']}
    assert all(n>=5 for n in counts.values())
    for e in errors:
        row=preds[e['sentence_id']]
        g,p = tuple(e['gold']) if e['gold'] else None,tuple(e['prediction']) if e['prediction'] else None
        if g: assert g in spans(row['gold'])
        if p: assert p in spans(row['pred'])
        if e['group']=='FP': assert p not in spans(row['gold'])
        if e['group']=='FN': assert g not in spans(row['pred'])
        if e['group']=='boundary':
            assert g[2]==p[2] and g[:2]!=p[:2] and g[0]<p[1] and p[0]<g[1]
    amb=json.loads((ART/'ambiguity_examples.json').read_text())
    assert len({r['surface'] for r in amb})>=3
    for e in amb:
        row=data['test'][e['ID']]
        assert e['text']==' '.join(row['words'])
        assert any(' '.join(row['words'][a:b])==e['surface'] and t==e['type'] for a,b,t in spans(row['tags']))
    examples=json.loads((ART/'ie_examples.json').read_text())
    tp=fp=fn=0
    for ex in examples:
        output=ex['output']
        words=[r['word'] for r in output['word_predictions']]
        offsets=[(r['start'],r['end']) for r in output['word_predictions']]
        tags=[r['tag'] for r in output['word_predictions']]
        assert to_json(output['text'],words,offsets,tags)==output
        for w,(a,b) in zip(words,offsets): assert output['text'][a:b]==w
        gold={tuple(x) for x in ex['gold']}
        # Verify saved IE gold against independent corpus annotations again.
        row=data['test'][ex['id']]
        cursor=0
        gold_offsets=[]
        for w in row['words']:
            gold_offsets.append((cursor,cursor+len(w)))
            cursor+=len(w)+1
        assert gold=={(gold_offsets[a][0],gold_offsets[b-1][1],t) for a,b,t in spans(row['tags']) if t!='MISC'}
        predicted=char_spans(output)
        tp+=len(gold&predicted); fp+=len(predicted-gold); fn+=len(gold-predicted)
    ie=json.loads((ART/'ie_evaluation.json').read_text())
    assert len(examples)==30 and (tp,fp,fn)==(ie['tp'],ie['fp'],ie['fn'])
    assert np.isclose(2*tp/(2*tp+fp+fn),ie['f1'])
    notebooks={}
    for name in ('tokenizer_alignment.ipynb','ner_finetuning.ipynb'):
        nb=nbformat.read(ROOT/name,as_version=4)
        nbformat.validate(nb)
        cells=[c for c in nb.cells if c.cell_type=='code']
        if check_notebooks:
            assert [c.execution_count for c in cells]==list(range(1,len(cells)+1))
        assert not any(o.output_type=='error' for c in cells for o in c.outputs)
        notebooks[name]=dict(code_cells=len(cells),executed=check_notebooks,errors=0)
    tokens=pd.read_csv(ART/'tokenization_analysis.csv')
    assert len(tokens)>=20
    assert np.allclose(tokens.R,tokens.subword_tokens/tokens.words)
    assert (tokens.extra_pieces==tokens.subword_tokens-tokens.words).all()
    for name in ('entity_length_analysis.png','confusion_matrix.png'):
        with PILImage.open(ROOT/name) as im:
            im.verify()
    reflection=(ROOT/'reflection.md').read_text(encoding='utf-8').split('## Reflection (300–500 từ)')[1].split('## Individual')[0]
    assert 300<=len(reflection.split())<=500
    for path in sorted(ROOT.glob('*.csv'))+sorted(ROOT.glob('*.png'))+sorted(ROOT.glob('*.ipynb')):
        hashes[path.name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report=dict(status='passed',unit_tests=test_count,script_compile=True,notebooks=notebooks,
                evaluation_sentences=3453,evaluation_words=46435,tokenization_sentences=len(tokens),
                error_groups=counts,ambiguity_surface_forms=len({r['surface'] for r in amb}),
                ie_sentences=len(examples),reflection_whitespace_words=len(reflection.split()),
                prediction_frozen=True,source_hashes_verified=True,artifacts=hashes,
                checkpoint_sha256=hashlib.sha256((ART/'checkpoint/model.safetensors').read_bytes()).hexdigest(),
                original_assignment_sha256=hashlib.sha256((ROOT/'doc/lab_09_named_entity_recognition_and_information_extraction.md').read_bytes()).hexdigest(),
                authorship='AI assistance disclosed in README; no student authorship or confirmation asserted')
    assert report['original_assignment_sha256']=='4f9c23f939069084aadbf6c1d27d43ca37638abee9a4322043740e1d03a45c74'
    write_json(ART/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='train',choices=['prepare','train','checks','notebooks','verify'])
    parser.add_argument('--train',action='store_true',help='Explicitly retrain when executing notebooks')
    args = parser.parse_args()
    if args.command == 'prepare': prepare_data()
    elif args.command == 'checks': run_self_checks()
    elif args.command == 'notebooks':
        execute_notebooks(args.train)
        verify_submission()
    elif args.command == 'verify': verify_submission()
    else: run_baseline()
