"""Independently verify a local routing archive; never load a model or collector."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

PASSES = ("original", "keep_08", "keep_06")
VARIANTS = PASSES + ("counterfactual_08", "counterfactual_06")
LAYERS, EXPERTS, K = 48, 128, 8


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path, compressed=False):
    with (gzip.open(path, "rt") if compressed else path.open()) as stream:
        for line in stream:
            yield load(line)


def compare(actual, expected, location="summary"):
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and actual.keys() == expected.keys(), f"{location}: fields differ")
        for key in actual:
            compare(actual[key], expected[key], f"{location}/{key}")
    elif isinstance(actual, list):
        require(isinstance(expected, list) and len(actual) == len(expected), f"{location}: length differs")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{location}/{i}")
    elif isinstance(actual, int):
        require(type(expected) is int and actual == expected, f"{location}: integer differs")
    else:
        require(type(expected) in (int, float) and np.isfinite(expected) and
                abs(actual - expected) <= 1e-9, f"{location}: float differs ({actual}, {expected})")


def accumulate(total, route, reference, partition, mask):
    for layer in range(LAYERS):
        ids, weights = route[layer]
        for p, label in enumerate(("prompt", "answer")):
            ix = partition == p
            selected, w = ids[ix], weights[ix]
            live = mask[layer, selected]
            n = int(ix.sum())
            overlap = int((selected[:, :, None] == reference[layer][0][ix, None, :]).any(axis=2).sum())
            values = dict(tokens=n,
                selected_counts=np.bincount(selected.ravel(), minlength=EXPERTS).tolist(),
                selected_weights=np.bincount(selected.ravel(), weights=w.ravel(), minlength=EXPERTS).tolist(),
                surviving_counts=np.bincount(selected[live], minlength=EXPERTS).tolist(),
                surviving_weights=np.bincount(selected[live], weights=w[live], minlength=EXPERTS).tolist(),
                surviving_histogram=np.bincount(live.sum(axis=1), minlength=K+1).tolist(),
                overlap_count=overlap)
            key = f"{layer}/{label}"
            if key not in total:
                total[key] = values
            else:
                for field, value in values.items():
                    if isinstance(value, list):
                        total[key][field] = [a+b for a, b in zip(total[key][field], value)]
                    else:
                        total[key][field] += value


def verify(archive, dataset=None, model=None):
    archive = Path(archive)
    manifest = load((archive / "manifest.json").read_text())
    summary = load((archive / "summary.json").read_text())
    require(set(manifest["selections"]) == set(PASSES[1:]), "selection names")
    masks = {"original": np.ones((LAYERS, EXPERTS), dtype=bool)}
    for name in PASSES[1:]:
        raw = np.asarray(manifest["selections"][name])
        require(raw.shape == (LAYERS, EXPERTS) and raw.dtype == np.bool_, "mask must be bool [48][128]")
        require(raw.any(axis=1).all(), "empty retained layer")
        masks[name] = raw
    compare({n: float(masks[n].mean()) for n in PASSES[1:]}, manifest["actual_retained_fractions"], "retained fractions")
    require(type(summary["questions"]) is int and summary["questions"] > 0, "invalid question count")
    limit = manifest["arguments"].get("limit")
    require(limit is None or type(limit) is int and limit > 0, "invalid limit")
    require(type(summary["pilot"]) is bool and summary["pilot"] == (limit is not None), "pilot mislabeled")
    require(limit is None or summary["questions"] <= limit, "pilot exceeds limit")
    require(summary.get("router_unchanged") is True, "router validation missing")
    require(summary.get("validation") == load((archive / "validation.json").read_text()), "validation mismatch")
    require(sha(archive / "tokens.jsonl") == summary["tokens_sha256"], "tokens checksum")
    tokens = list(rows(archive / "tokens.jsonl"))
    progress = list(rows(archive / "progress.jsonl"))
    n = summary["questions"]
    require(len(tokens) == len(progress) == n, "question count mismatch")
    expected_files = {f"question_{i:05d}.jsonl.gz" for i in range(n)}
    require({p.name for p in archive.glob("question_*")} == expected_files, "route file coverage")
    source_rows, tokenizer = None, None
    if dataset is not None:
        dataset = Path(dataset)
        require(sha(dataset) == manifest['inputs'][manifest['arguments']['dataset']], 'source dataset checksum')
        source_rows = list(rows(dataset))
        expected_n = len(source_rows) if limit is None else min(limit, len(source_rows))
        require(n == expected_n, 'source dataset completeness')
    if model is not None:
        require(source_rows is not None, '--model requires --dataset')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        expected_suffix = "Please reason step by step, and put your final numeric answer after '#### '."
        require(manifest['suffix'] == expected_suffix, 'unexpected prompt suffix')
    totals = {v: {} for v in VARIANTS}
    token_count = included = route_rows = 0
    for i, (token, checkpoint) in enumerate(zip(tokens, progress)):
        require(type(token["index"]) is int and type(checkpoint["index"]) is int and
                token["index"] == checkpoint["index"] == i, "nonsequential question index")
        if "dataset_line" in token:
            require(type(token["dataset_line"]) is int and token["dataset_line"] == i+1, "dataset line coverage")
        input_ids, parts = token["input_ids"], token["partition"]
        require(isinstance(input_ids, list) and input_ids and all(type(x) is int and x >= 0 for x in input_ids), "invalid token IDs")
        require(isinstance(parts, list) and len(parts) == len(input_ids) and
                all(type(x) is int and x in (-1, 0, 1) for x in parts) and 0 in parts and 1 in parts, "invalid partitions")
        require(type(checkpoint["tokens"]) is int and checkpoint["tokens"] == len(parts), "progress token count")
        if tokenizer is not None:
            source = source_rows[i]
            user = source['question'] + '\n\n' + expected_suffix
            text = tokenizer.apply_chat_template([dict(role='user', content=user)],
                                                 tokenize=False, add_generation_prompt=True)
            require(text.count(user) == 1, 'source prompt span')
            start = text.index(user)
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            answer = tokenizer.encode(source['answer'], add_special_tokens=False)
            expected_parts = [0 if a >= start and b <= start+len(user) and b > a else -1
                              for a, b in encoded['offset_mapping']] + [1]*len(answer)
            require(input_ids == encoded['input_ids'] + answer and parts == expected_parts,
                    'source token/partition mismatch')
        partition = np.asarray(parts)
        token_count += len(parts)
        included += int((partition >= 0).sum())
        path = archive / f"question_{i:05d}.jsonl.gz"
        require(sha(path) == checkpoint["routes_sha256"], f"routes checksum {i}")
        routes = {v: {} for v in PASSES}
        for row in rows(path, True):
            variant, layer = row["variant"], row["layer"]
            require(variant in PASSES and type(layer) is int and 0 <= layer < LAYERS, "unknown route")
            require(layer not in routes[variant], "duplicate route")
            ids, weights = np.asarray(row["ids"]), np.asarray(row["weights"])
            require(ids.shape == weights.shape == (len(parts), K), "route dimensions")
            require(ids.dtype.kind in "iu" and ((ids >= 0) & (ids < EXPERTS)).all(), "expert IDs")
            require((np.diff(np.sort(ids, axis=1), axis=1) != 0).all(), "duplicate expert IDs")
            require(weights.dtype.kind in "fiu" and np.isfinite(weights).all() and (weights > 0).all(), "nonpositive/nonfinite weights")
            weights = weights.astype(np.float64)
            require(((weights.sum(axis=1) >= .99) & (weights.sum(axis=1) <= 1.01)).all(), "native weight normalization")
            f32 = weights.astype(np.float32)
            require(np.array_equal(f32.astype(np.float64), weights) and
                    ((f32.view(np.uint32) & 65535) == 0).all(), "weights not native BF16")
            routes[variant][layer] = ids, weights
            route_rows += 1
        require(all(set(r) == set(range(LAYERS)) for r in routes.values()), "layer coverage")
        for variant in PASSES[1:]:
            require(all(np.array_equal(a, b) for a, b in zip(routes[variant][0], routes["original"][0])), "first-layer pre-drop routes differ")
        for variant in PASSES:
            accumulate(totals[variant], routes[variant], routes["original"], partition, masks[variant])
        for suffix in ("08", "06"):
            accumulate(totals[f"counterfactual_{suffix}"], routes["original"], routes["original"], partition, masks[f"keep_{suffix}"])
    for stats in totals.values():
        for s in stats.values():
            count, kept = sum(s["selected_counts"]), sum(s["surviving_counts"])
            weight, mass = sum(s["selected_weights"]), sum(s["surviving_weights"])
            require(count == s["tokens"]*K and sum(s["surviving_histogram"]) == s["tokens"] and
                    kept == sum(j*c for j, c in enumerate(s["surviving_histogram"])), "count conservation")
            s.update(dropped_count=count-kept, dropped_weight=weight-mass,
                     selection_denominator=count, weight_denominator=weight,
                     dropped_selection_fraction=(count-kept)/count,
                     dropped_weight_fraction=(weight-mass)/weight,
                     mean_survivors=kept/s["tokens"], overlap_fraction=s["overlap_count"]/(s["tokens"]*K))
    compare(totals, summary["variants"])
    return dict(verified=True, questions=n, pilot=summary["pilot"], tokens=token_count,
                included_tokens=included, route_rows=route_rows, layers=LAYERS, experts=EXPERTS, top_k=K,
                checks={k: True for k in ("coverage", "dimensions", "unique_ids", "positive_bf16_weights",
                    "normalization", "tokens_checksum", "route_checksums", "first_layer_equal",
                    "retained_fractions", "question_counts", "pilot_label", "all_summary_statistics",
                    "mask_only_counterfactuals", "count_conservation")},
                float_absolute_tolerance=1e-9,
                source_dataset_verified=source_rows is not None,
                source_tokenization_verified=tokenizer is not None,
                scope="Raw archive statistics independently recomputed; source dataset/tokenization verified when supplied. Not a kernel-execution or cross-backend equivalence check.",
                manifest_sha256=sha(archive / "manifest.json"), summary_sha256=sha(archive / "summary.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, help="Independently check source hash and full split coverage")
    parser.add_argument("--model", type=Path, help="With --dataset, re-tokenize the local checkpoint template")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output already exists")
    result = verify(args.archive, args.dataset, args.model)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
