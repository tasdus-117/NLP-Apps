"""LAB 10: compact AI-generated reference implementation, not student work.

Sections 10–16: strings/annotations -> overflowing features (memory only).
Sections 17–21: features -> start/end logits, explicit loss, spans and EM/F1.
train_run writes a checkpoint and training_metadata.json; other functions write
no files. Alignment and metrics are reference answers under section 29: students
must independently implement/explain these before comparing this code.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import string
import time


@dataclass
class Config:
    model_name: str = 'distilbert-base-uncased'
    max_length: int = 384
    stride: int = 64
    seed: int = 42
    batch_size: int = 32
    lr: float = 3e-5
    epochs: int = 2
    runtime: str = '.runtime'
    max_answer_length: int = 30
    question_max_tokens: int = 48
    precision: str = 'bf16'


def align_span(offsets, sequence_ids, answer_start, answer_text, cls_index):
    """16. Character [start,end) -> inclusive pair-token span or CLS sentinel.

    A substring strictly inside a token maps to its enclosing token. A partial
    answer at a window edge does not become an incorrectly shortened target.
    Input: offsets, sequence ownership, annotated text/start. Output: two ints.
    """
    if not answer_text or answer_start < 0:
        return cls_index, cls_index
    answer_end = answer_start + len(answer_text)
    start_token = end_token = None
    for i, (offset, sequence) in enumerate(zip(offsets, sequence_ids)):
        if sequence != 1 or offset is None:
            continue
        lo, hi = offset
        if lo <= answer_start < hi:
            start_token = i
        if lo < answer_end <= hi:
            end_token = i
    if start_token is None or end_token is None or start_token > end_token:
        return cls_index, cls_index
    return start_token, end_token


def prepare_features(examples, tokenizer, config):
    """10–16. SQuAD rows -> fixed-size overflow features, no disk output.

    A question cap of 48 is identical across length runs. With BERT/DistilBERT's
    three special tokens, even length 128 leaves 77 context tokens > stride 64.
    Character offsets always refer to the original, untouched context.
    """
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer is required for offset mapping.')
    if config.max_length <= 0 or config.question_max_tokens <= 0 or config.stride < 0:
        raise ValueError('Invalid sequence length, question cap, or stride.')
    if not examples:
        return []
    ids = [str(row['id']) for row in examples]
    if len(ids) != len(set(ids)):
        raise ValueError('Example IDs must be unique.')
    questions = []
    for row in examples:
        # Slice the original question at a token boundary instead of decode/reencode.
        question = row['question'].strip()
        q = tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
        if len(q['input_ids']) > config.question_max_tokens:
            question = question[:q['offset_mapping'][config.question_max_tokens - 1][1]]
        q_length = len(tokenizer(question, add_special_tokens=False)['input_ids'])
        context_budget = config.max_length - q_length - tokenizer.num_special_tokens_to_add(pair=True)
        if not 0 <= config.stride < context_budget:
            raise ValueError(f'stride={config.stride} must be below context budget={context_budget}.')
        questions.append(question)
    encoded = tokenizer(questions, [row['context'] for row in examples],
        truncation='only_second', max_length=config.max_length, stride=config.stride,
        padding='max_length', return_overflowing_tokens=True, return_offsets_mapping=True)
    features = []
    for window, sample in enumerate(encoded['overflow_to_sample_mapping']):
        row = examples[sample]
        sequences = encoded.sequence_ids(window)
        offsets = encoded['offset_mapping'][window]
        cls = encoded['input_ids'][window].index(tokenizer.cls_token_id)
        answers = row.get('answers', {'answer_start':[], 'text':[]})
        starts, texts = answers['answer_start'], answers['text']
        if len(starts) != len(texts):
            raise ValueError('Answer starts/text lengths differ.')
        label = (cls, cls)
        for start, text in zip(starts, texts):
            if text and (start < 0 or row['context'][start:start+len(text)] != text):
                raise ValueError(f'Annotation is not a context substring: {row["id"]}')
            candidate = align_span(offsets, sequences, start, text, cls)
            if candidate != (cls, cls):
                label = candidate
                break
        feature = {key:encoded[key][window] for key in tokenizer.model_input_names if key in encoded}
        feature.update(example_id=str(row['id']), sample_index=int(sample),
            overflow_to_sample_mapping=int(sample), sequence_ids=sequences,
            offset_mapping=[tuple(offset) if sequence == 1 and offset[1] > offset[0] else None
                            for offset, sequence in zip(offsets, sequences)],
            start_positions=label[0], end_positions=label[1])
        features.append(feature)
    return features


def decode_spans(examples, features, start_logits, end_logits, max_answer_length=30):
    """15/17. Join window logits -> best legal span per original sample.

    Score = start logit + end logit, compared globally over all its windows.
    No per-window softmax: its varying denominator distorts cross-window ranks.
    No top-k approximation: enumerate every bounded valid span in a window.
    Output offsets are half-open character offsets in the original context.
    """
    if max_answer_length < 1:
        raise ValueError('max_answer_length must be positive.')
    if not (len(features) == len(start_logits) == len(end_logits)):
        raise ValueError('Feature and logit window counts differ.')
    contexts = {str(row['id']):row['context'] for row in examples}
    predictions = {key:{'text':'', 'start':0, 'end':0, 'score':float('-inf')} for key in contexts}
    import numpy as np
    for feature, starts, ends in zip(features, start_logits, end_logits):
        key = feature['example_id']
        offsets = feature['offset_mapping']
        if len(offsets) != len(starts) or len(offsets) != len(ends):
            raise ValueError('Token/logit lengths differ.')
        if not offsets:
            continue
        n = len(offsets)
        valid_token = np.array([offset is not None for offset in offsets])
        bounds = np.array([offset if offset is not None else (0,0) for offset in offsets])
        starts_at = np.arange(n)[:,None]
        ends_at = starts_at + np.arange(min(n, max_answer_length))[None,:]
        clipped = np.minimum(ends_at, n-1)
        # Prefix counts also forbid crossing a non-context token inside the span.
        invalid_prefix = np.r_[0, np.cumsum(~valid_token)]
        valid_span = (ends_at < n) & (invalid_prefix[clipped+1] == invalid_prefix[starts_at])
        lo, hi = bounds[:,0,None], bounds[clipped,1]
        valid_span &= (lo >= 0) & (lo < hi) & (hi <= len(contexts[key]))
        scores = np.asarray(starts,dtype=np.float64)[:,None] + np.asarray(ends,dtype=np.float64)[clipped]
        scores = np.where(valid_span & np.isfinite(scores), scores, -np.inf)
        # Row-major argmax preserves exhaustive i-then-j order, including ties.
        i, distance = np.unravel_index(np.argmax(scores),scores.shape)
        score = float(scores[i,distance])
        if score > predictions[key]['score']:
            first, last = int(bounds[i,0]), int(bounds[i+distance,1])
            predictions[key] = {'text':contexts[key][first:last], 'start':first, 'end':last, 'score':score}
    return predictions


def normalize_answer(text):
    """18–19. Official SQuAD English normalization, output normalized string."""
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def squad_metrics(prediction, gold_texts):
    """18–20. One answer + all references -> EM/F1 in [0,1], no file output.

    Repeated words use Counter intersection. Maximize each metric independently
    over all accepted answers. Aggregate x100 outside this function for percent.
    """
    predicted = normalize_answer(prediction)
    pred_tokens = predicted.split()
    scores = []
    for gold in gold_texts or ['']:
        gold = normalize_answer(gold)
        gold_tokens = gold.split()
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if not pred_tokens or not gold_tokens:
            f1 = float(pred_tokens == gold_tokens)
        else:
            f1 = 2 * overlap / (len(pred_tokens) + len(gold_tokens))
        scores.append((float(predicted == gold), f1))
    return {'em':max(x[0] for x in scores), 'f1':max(x[1] for x in scores)}


def baseline(example):
    """21. Keyword overlap -> best context sentence as candidate span.

    Input: question/context. Output: exact context substring and char offsets.
    shortcut: whole-sentence answers are coarse; improve span rules only after
    measuring this intentionally simple heuristic baseline.
    """
    question_words = set(normalize_answer(example['question']).split())
    candidates = []
    for match in re.finditer(r'[^.!?\n]+[.!?]?', example['context']):
        raw = match.group()
        text = raw.strip()
        if text:
            lo = match.start() + len(raw) - len(raw.lstrip())
            score = len(question_words & set(normalize_answer(text).split()))
            candidates.append({'text':text, 'start':lo, 'end':lo+len(text), 'score':float(score)})
    return max(candidates, key=lambda row:row['score']) if candidates else {
        'text':'', 'start':0, 'end':0, 'score':0.0}


def qa_loss(start_logits, end_logits, start_positions, end_positions):
    """8/17. Logits + integer labels -> mean start/end cross-entropy scalar."""
    from torch.nn.functional import cross_entropy
    return (cross_entropy(start_logits, start_positions) +
            cross_entropy(end_logits, end_positions)) / 2


def _batches(features, model, batch_size, device, shuffle=False, generator=None):
    """Tensorize model-accepted inputs only; metadata never reaches the backbone."""
    import torch
    parameters = inspect.signature(model.forward).parameters
    keys = [key for key in ('input_ids', 'attention_mask', 'token_type_ids')
            if key in parameters and key in features[0]]
    indices = torch.randperm(len(features), generator=generator).tolist() if shuffle else list(range(len(features)))
    for lo in range(0, len(indices), batch_size):
        rows = [features[i] for i in indices[lo:lo+batch_size]]
        batch = {key:torch.tensor([row[key] for row in rows], dtype=torch.long, device=device) for key in keys}
        labels = [torch.tensor([row[key] for row in rows], dtype=torch.long, device=device)
                  for key in ('start_positions', 'end_positions')]
        yield batch, labels


def predict(model, tokenizer, examples, config, device=None):
    """17/27. Model + raw QA rows -> decoded spans, no disk output."""
    import torch
    features = prepare_features(examples, tokenizer, config)
    if not features:
        return {}
    device = device or str(next(model.parameters()).device)
    model.to(device)
    was_training = model.training
    model.eval()
    starts, ends = [], []
    with torch.no_grad():
        for inputs, _ in _batches(features, model, config.batch_size, device):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=str(device).startswith('cuda') and config.precision == 'bf16'):
                out = model(**inputs)
            starts.extend(out.start_logits.float().cpu().tolist())
            ends.extend(out.end_logits.float().cpu().tolist())
    model.train(was_training)
    return decode_spans(examples, features, starts, ends, config.max_answer_length)


def checkpoint_rank(record):
    """17. Rank validation-only checkpoints by F1, EM, then lower loss."""
    return record['validation_f1'], record['validation_em'], -record['validation_loss']


def train_run(train_examples, validation_examples, config, output_dir):
    """17/30. Train/validation rows -> checkpoint + measured epoch history.

    Input: fixed splits and Config. Operation: fresh pretrained QA head/backbone,
    AdamW, explicit start/end CE, all windows, epoch validation F1/EM. Only internal
    validation selects the best epoch. Never receives held-out evaluation rows.
    Output: fresh directory with improved epoch snapshots, final best root model,
    tokenizer, experiment_config.json, history.json and training_metadata.json.
    Existing nonempty directories are rejected. No training runs on import.
    """
    runtime = Path(config.runtime).resolve()
    workspace = Path(__file__).resolve().parent
    destination = Path(output_dir).resolve()
    if not runtime.is_relative_to(workspace) or not destination.is_relative_to(workspace):
        raise ValueError('Runtime and output paths must stay inside lab10.')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {destination}')
    if not train_examples or not validation_examples:
        raise ValueError('Training and validation splits must be nonempty.')
    if config.batch_size < 1 or config.epochs < 1 or config.lr <= 0:
        raise ValueError('Invalid batch size, epoch count or learning rate.')
    if config.precision not in ('bf16', 'fp32'):
        raise ValueError('precision must be bf16 or fp32.')
    os.environ.setdefault('HF_HOME', str(runtime / 'hf_cache'))
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, __version__ as transformers_version
    random.seed(config.seed)
    import numpy as np
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Warn rather than silently claim bit-identical results across different GPUs.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda' and config.precision == 'bf16'
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requested BF16 is unsupported on this GPU; keep all comparisons on one supported precision.')
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True,
        cache_dir=str(runtime / 'hf_cache'))
    model = AutoModelForQuestionAnswering.from_pretrained(config.model_name,
        cache_dir=str(runtime / 'hf_cache')).to(device)
    preprocessing_started = time.perf_counter()
    print(f'Preparing length={config.max_length}: train={len(train_examples)}, validation={len(validation_examples)}', flush=True)
    train_features = prepare_features(train_examples, tokenizer, config)
    # Training consumes labels/inputs only; retain offsets/ownership for validation.
    keep = set(tokenizer.model_input_names) | {'start_positions','end_positions'}
    for feature in train_features:
        for key in list(feature):
            if key not in keep:
                del feature[key]
    val_features = prepare_features(validation_examples, tokenizer, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=.01)
    generator = torch.Generator().manual_seed(config.seed)
    history = []
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(destination)
    (destination / 'experiment_config.json').write_text(json.dumps(asdict(config),indent=2),encoding='utf-8')
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    steps_per_epoch = math.ceil(len(train_features)/config.batch_size)
    metadata = {'status':'running', 'history':history, 'training_time':0.0, 'config':asdict(config),
        'device':device, 'precision_actual':'bf16_autocast' if use_amp else 'fp32',
        'checkpoint':str(destination), 'train_examples':len(train_examples),
        'validation_examples':len(validation_examples), 'train_features':len(train_features),
        'validation_features':len(val_features), 'optimizer_steps_per_epoch':steps_per_epoch,
        'optimizer_steps':0, 'optimizer':'torch.optim.AdamW', 'optimizer_defaults':optimizer.defaults,
        'gradient_clip_norm':1.0, 'scheduler':None, 'torch_version':torch.__version__,
        'transformers_version':transformers_version, 'numpy_version':np.__version__,
        'cuda_version':torch.version.cuda,
        'hardware':torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
        'cpu_threads':torch.get_num_threads(), 'preprocessing_time':preprocessing_seconds,
        'selection_rule':'highest internal-validation F1; ties EM then lower loss; earliest exact tie',
        'timing_scope':'training epochs including validation inference/decoding and epoch checkpoint writes; excluding preprocessing/final best copy',
        'determinism':'fixed seeds; deterministic algorithms warn_only; cross-hardware bitwise equality not guaranteed'}
    print(f'Features train={len(train_features)}, validation={len(val_features)}, steps/epoch={steps_per_epoch}, precision={metadata["precision_actual"]}', flush=True)
    best_record = None
    best_snapshot = None
    if device == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_sum = 0.0
        for step, (inputs, labels) in enumerate(_batches(train_features, model, config.batch_size, device, True, generator),1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                out = model(**inputs)
                loss = qa_loss(out.start_logits, out.end_logits, *labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}, step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.detach().item() * len(labels[0])
            if step % 100 == 0 or step == steps_per_epoch:
                print(f'length={config.max_length} epoch={epoch}/{config.epochs} step={step}/{steps_per_epoch} loss={loss.item():.5f} elapsed={time.perf_counter()-epoch_started:.1f}s',flush=True)
        model.eval()
        val_sum = 0.0
        val_starts, val_ends = [], []
        with torch.no_grad():
            for inputs, labels in _batches(val_features, model, config.batch_size, device):
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=use_amp):
                    out = model(**inputs)
                    loss = qa_loss(out.start_logits,out.end_logits,*labels)
                val_sum += loss.item() * len(labels[0])
                val_starts.extend(out.start_logits.float().cpu().tolist())
                val_ends.extend(out.end_logits.float().cpu().tolist())
        validation_predictions = decode_spans(validation_examples,val_features,val_starts,val_ends,config.max_answer_length)
        scores = [squad_metrics(validation_predictions[str(row['id'])]['text'],row['answers']['text'])
                  for row in validation_examples]
        record = {'epoch':epoch, 'train_loss':train_sum/len(train_features),
            'validation_loss':val_sum/len(val_features),
            'validation_em':sum(score['em'] for score in scores)/len(scores),
            'validation_f1':sum(score['f1'] for score in scores)/len(scores),
            'optimizer_steps':epoch*steps_per_epoch, 'epoch_seconds':time.perf_counter()-epoch_started}
        if best_record is None or checkpoint_rank(record) > checkpoint_rank(best_record):
            best_record = record
            best_snapshot = destination / f'epoch_{epoch}'
            model.save_pretrained(best_snapshot)
            tokenizer.save_pretrained(best_snapshot)
            (best_snapshot / 'experiment_config.json').write_text(json.dumps(asdict(config),indent=2),encoding='utf-8')
        history.append(record)
        metadata.update(best_epoch=best_record['epoch'],
            best_validation={'em':best_record['validation_em'],'f1':best_record['validation_f1'],'loss':best_record['validation_loss']},
            best_epoch_checkpoint=str(best_snapshot), optimizer_steps=epoch*steps_per_epoch,
            training_time=time.perf_counter()-started)
        (destination / 'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        (destination / 'training_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(record,flush=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    metadata['training_time'] = time.perf_counter() - started
    del optimizer, model
    if device == 'cuda':
        torch.cuda.empty_cache()
    model = AutoModelForQuestionAnswering.from_pretrained(best_snapshot,local_files_only=True).to(device)
    model.save_pretrained(destination)
    model.eval()
    metadata['status'] = 'complete'
    (destination / 'training_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return {'model':model, 'tokenizer':tokenizer, **metadata}
