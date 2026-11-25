"""27. Application: Document QA — AI-assisted reference implementation.

Input: question and supplied UTF-8 context/document plus a local QA checkpoint.
Operation: tokenize all overlapping windows, predict start/end, merge valid spans.
Output: JSON on stdout; no files written. Score is a raw logit sum, not confidence.
29. AI Usage: this implementation is AI-assisted under explicit user authorization.
"""

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--context", help="Context supplied directly")
    source.add_argument("--document", type=Path, help="Plain-text UTF-8 document")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Local directory saved by train_run")
    args = parser.parse_args()
    context = args.context if args.context is not None else args.document.read_text(encoding="utf-8")
    if not args.question.strip() or not context.strip():
        parser.error("question and context must be nonempty")
    if not args.checkpoint.is_dir():
        parser.error("checkpoint must be an existing local model directory")

    root = Path(__file__).resolve().parent
    runtime = root / ".runtime" / "full_run"
    os.environ.setdefault("HF_HOME", str(runtime / "hf_cache"))
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from qa_model import Config, predict

    saved = args.checkpoint / "experiment_config.json"
    metadata = args.checkpoint / "training_metadata.json"
    if saved.exists():
        config_data = json.loads(saved.read_text(encoding="utf-8"))
    elif metadata.exists():
        config_data = json.loads(metadata.read_text(encoding="utf-8"))["config"]
    else:
        parser.error("checkpoint needs experiment_config.json or training_metadata.json to reproduce preprocessing")
    valid_fields = {field.name for field in fields(Config)}
    config_data = {key: value for key, value in config_data.items() if key in valid_fields}
    config_data.update(model_name=str(args.checkpoint.resolve()), runtime=str(runtime))
    config = Config(**config_data)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoint, local_files_only=True)
    position_limit = getattr(model.config, "max_position_embeddings", None)
    if position_limit is not None and config.max_length > position_limit:
        parser.error(f"saved max_length={config.max_length} exceeds model position limit={position_limit}")
    example = {"id": "document", "question": args.question, "context": context,
               "answers": {"text": [], "answer_start": []}}
    answer = predict(model, tokenizer, [example], config)["document"]
    print(json.dumps({**answer, "score_type": "raw_start_plus_end_logits",
                      "warning": "SQuAD 1 core always selects a span; no calibrated no-answer detection."},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
