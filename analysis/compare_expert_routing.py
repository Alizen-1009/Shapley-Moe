"""Independent, teacher-forced Qwen3 routing experiment (not an execution benchmark).

Prompt spans contain only the literal user question + suffix; answer spans contain
only gold answer tokens. All chat-template/BOS tokens are excluded. No assistant
EOS is appended. Zero-weight experts may still execute; no zero-reference numerical
equivalence is asserted. Imports of numpy/torch/transformers are runtime-only.
"""
import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path
import sys

SUFFIX = "Please reason step by step, and put your final numeric answer after '#### '."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    with open(path, "x") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def load_selection(path, layers, experts):
    raw = json.loads(Path(path).read_text())
    require(isinstance(raw, dict), "selection must be a layer mapping")
    require(set(raw) - {"_metadata"} == {str(i) for i in range(layers)},
            "selection missing/extra layers")
    masks = []
    for i in range(layers):
        ids = raw[str(i)]
        require(isinstance(ids, list) and ids and
                all(type(x) is int and 0 <= x < experts for x in ids) and
                len(set(ids)) == len(ids), f"invalid selection layer {i}")
        masks.append([e in ids for e in range(experts)])
    return masks, raw.get("_metadata", {})


def token_sequence(tokenizer, question, answer, max_length):
    user = question + "\n\n" + SUFFIX
    messages = [{"role": "user", "content": user}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=False)
    # Locate literal user content in the rendered template by offsets, not by
    # searching for a possibly repeated token subsequence.
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    require(rendered.count(user) == 1, "ambiguous user span")
    start = rendered.index(user)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    require(encoded["input_ids"] == prefix, "chat template/tokenization mismatch")
    gold = tokenizer.encode(answer, add_special_tokens=False)
    require(gold, "empty answer tokens")
    partition = [0 if a >= start and b <= start + len(user) and b > a else -1
                 for a, b in encoded["offset_mapping"]] + [1] * len(gold)
    ids = prefix + gold
    require(0 in partition and len(ids) <= max_length, "empty prompt or sequence exceeds max-length; no truncation")
    return ids, partition


def gate_hook(mask, partition, records, layer, experts, topk):
    """Capture native BF16 pre-drop weights; change weights only, never IDs/logits."""
    def hook(module, args, output):
        import torch
        require(isinstance(output, tuple) and len(output) == 3, "unsupported gate contract")
        logits, weights, ids = output
        require(layer not in records, "gate called twice in one forward")
        require(tuple(ids.shape) == (len(partition), topk) and weights.shape == ids.shape,
                "unexpected gate dimensions")
        require(weights.dtype == torch.bfloat16 and bool(torch.isfinite(logits).all()) and
                bool(torch.isfinite(weights).all()) and bool((weights >= 0).all()), "invalid weights/logits or non-BF16 gate")
        require(bool(((ids >= 0) & (ids < experts)).all()), "invalid expert IDs")
        require(bool((ids.sort(-1).values.diff(dim=-1) != 0).all()), "duplicate top-k IDs")
        records[layer] = (ids.detach().cpu().numpy().astype("int16"),
                          weights.detach().float().cpu().numpy())
        if mask is None:
            return output
        retained = torch.tensor(mask, device=ids.device, dtype=torch.bool)[ids]
        return logits, weights * retained, ids
    return hook


def collect(backbone, gates, ids, partition, masks, experts, topk, device):
    import torch
    records, handles = {}, []
    try:
        for layer, gate in enumerate(gates):
            handles.append(gate.register_forward_hook(gate_hook(
                None if masks is None else masks[layer], partition, records, layer, experts, topk)))
        output = backbone(input_ids=torch.tensor([ids], device=device), use_cache=False,
                          return_dict=True).last_hidden_state
        require(set(records) == set(range(len(gates))), "incomplete layer coverage")
        require(bool(torch.isfinite(output).all()), "nonfinite backbone output")
        return records, output
    finally:
        for handle in handles:
            handle.remove()


def statistics(routes, partition, masks, original, experts, topk):
    import numpy as np
    result = {}
    partition = np.asarray(partition)
    for layer, (ids, weights) in routes.items():
        mask = np.ones(experts, dtype=bool) if masks is None else np.asarray(masks[layer])
        survived = mask[ids]
        for p, name in enumerate(("prompt", "answer")):
            ix = partition == p
            selected_ids, w, live = ids[ix], weights[ix], survived[ix]
            n = int(ix.sum())
            require(n > 0, "empty partition")
            counts = np.bincount(selected_ids.ravel(), minlength=experts)
            sums = np.bincount(selected_ids.ravel(), weights=w.ravel(), minlength=experts)
            kept = np.bincount(selected_ids[live], minlength=experts)
            mass = np.bincount(selected_ids[live], weights=w[live], minlength=experts)
            hist = np.bincount(live.sum(axis=1), minlength=topk + 1)
            overlap = (selected_ids[:, :, None] == original[layer][0][ix, None, :]).any(axis=2).sum()
            require(int(counts.sum()) == n * topk and int(hist.sum()) == n and
                    int(kept.sum()) == int((hist * np.arange(topk + 1)).sum()) and
                    bool((counts >= kept).all()) and np.isfinite(sums).all() and sums.sum() > 0,
                    "count/weight conservation failed")
            result[f"{layer}/{name}"] = dict(tokens=n, selected_counts=counts.tolist(),
                selected_weights=sums.tolist(), surviving_counts=kept.tolist(),
                surviving_weights=mass.tolist(), surviving_histogram=hist.tolist(),
                overlap_count=int(overlap))
    return result


def merge(total, question):
    for key, values in question.items():
        if key not in total:
            total[key] = values
        else:
            for field, value in values.items():
                if isinstance(value, list):
                    total[key][field] = [a + b for a, b in zip(total[key][field], value)]
                else:
                    total[key][field] += value


def finalize(total, topk):
    for s in total.values():
        count, kept = sum(s["selected_counts"]), sum(s["surviving_counts"])
        weight, mass = sum(s["selected_weights"]), sum(s["surviving_weights"])
        s.update(dropped_count=count-kept, dropped_weight=weight-mass,
                 selection_denominator=count, weight_denominator=weight,
                 dropped_selection_fraction=(count-kept)/count,
                 dropped_weight_fraction=(weight-mass)/weight,
                 mean_survivors=kept/s["tokens"], overlap_fraction=s["overlap_count"]/(s["tokens"]*topk))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "dataset", "selection-08", "selection-06", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--experts-implementation", choices=("grouped_mm", "eager"), default="grouped_mm")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    require(args.limit is None or args.limit > 0, "limit must be positive")
    require(args.max_length > 0 and args.device.startswith("cuda:"), "invalid length/device")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # even partial outputs may never be reused
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require(torch.cuda.is_available(), "CUDA required")
    # Local-only loading: an absent checkpoint must never trigger remote access.
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation=args.experts_implementation).to(args.device).eval()
    require(model.config.model_type == "qwen3_moe", "expected Qwen3 MoE checkpoint")
    backbone = model.model
    layers, experts, topk = model.config.num_hidden_layers, model.config.num_experts, model.config.num_experts_per_tok
    gates = [layer.mlp.gate for layer in backbone.layers]
    require(len(gates) == layers, "layer coverage mismatch")
    require(all(type(layer.mlp).__name__ == "Qwen3MoeSparseMoeBlock" for layer in backbone.layers),
            "unsupported MoE module layout")
    selections, metadata = {}, {}
    for name, path in (("keep_08", args.selection_08), ("keep_06", args.selection_06)):
        selections[name], metadata[name] = load_selection(path, layers, experts)
    def router_hash():
        h = hashlib.sha256()
        for gate in gates:
            for parameter in gate.parameters():
                h.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return h.hexdigest()
    router_before = router_hash()
    checkpoint_files = sorted(p for p in Path(args.model).rglob("*") if p.is_file() and
                              (p.suffix == ".safetensors" or p.name.endswith(".json") or p.name.endswith(".model")))
    require(any(p.suffix == ".safetensors" for p in checkpoint_files), "local safetensors checkpoint required")
    manifest = dict(arguments=vars(args), command=sys.argv, source_sha256=digest(__file__),
        inputs={str(p): digest(p) for p in (args.dataset, args.selection_08, args.selection_06)},
        checkpoint={str(p): digest(p) for p in checkpoint_files},
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                      python=sys.version, cuda=torch.version.cuda),
        gpu=dict(name=torch.cuda.get_device_name(args.device), properties=str(torch.cuda.get_device_properties(args.device))),
        router_sha256=router_before, selections=selections, selection_metadata=metadata,
        model_commit=getattr(model.config, "_commit_hash", None),
        tokenizer_commit=tokenizer.init_kwargs.get("_commit_hash"),
        route_schema={"format": "per-question jsonl.gz; one row per variant/layer",
                      "fields": ["variant", "layer", "ids[token][k]", "weights[token][k]"],
                      "weights": "native pre-drop BF16 represented as JSON float; apply manifest masks to reproduce survivors",
                      "alignment": "tokens.jsonl index matches question filename; partition -1 excluded, 0 prompt, 1 answer"},
        actual_retained_fractions={n: sum(map(sum, m))/(layers*experts) for n, m in selections.items()},
        token_protocol="user literal question + two newlines + suffix only; gold answer only; exclude template/BOS; no assistant EOS",
        suffix=SUFFIX, overlap_definition="intersection of pre-drop top-k IDs / top-k at aligned token positions",
        caveat="surviving contributions are not measured executed work; counterfactuals are mask-only, not model passes")
    save_json(out / "manifest.json", manifest)
    with open(out / "collector_source.py", "x") as snapshot:
        snapshot.write(Path(__file__).read_text())
    totals = {n: {} for n in ("original", "keep_08", "keep_06", "counterfactual_08", "counterfactual_06")}
    completed, saw_drop, validation = 0, False, {}
    with open(args.dataset) as dataset, open(out / "tokens.jsonl", "x") as tokens, open(out / "progress.jsonl", "x") as progress, torch.inference_mode():
        for line_number, line in enumerate(dataset, 1):
            if args.limit is not None and completed >= args.limit:
                break
            row = json.loads(line)
            require(all(isinstance(row.get(k), str) and row[k].strip() for k in ("question", "answer")), "invalid question/answer")
            ids, partition = token_sequence(tokenizer, row["question"], row["answer"], args.max_length)
            if completed == 0:
                plain = backbone(input_ids=torch.tensor([ids], device=args.device), use_cache=False,
                                 return_dict=True).last_hidden_state
            original, hidden = collect(backbone, gates, ids, partition, None, experts, topk, args.device)
            if completed == 0:
                torch.testing.assert_close(hidden, plain, rtol=0, atol=0)
                del plain
                all_retained = [[True]*experts for _ in gates]
                control, control_hidden = collect(backbone, gates, ids, partition, all_retained, experts, topk, args.device)
                torch.testing.assert_close(control_hidden, hidden, rtol=0, atol=0)
                require(all(np.array_equal(control[l][j], original[l][j]) for l in original for j in (0, 1)), "no-drop routes changed")
                validation["no_drop_backbone"] = "exact BF16 output and route equality; rtol=atol=0"
                del control, control_hidden
                # Fixed-hidden gate probes validate pre-drop IDs and a real dropped selection.
                probe = backbone.embed_tokens(torch.tensor([ids], device=args.device)).reshape(-1, model.config.hidden_size)
                for l, gate in enumerate(gates):
                    native = gate(probe)
                    for masks in selections.values():
                        captured = {}
                        hooked = gate_hook(masks[l], partition, captured, l, experts, topk)(gate, (probe,), native)
                        require(hooked[0] is native[0] and hooked[2] is native[2], "router outputs replaced")
                        expected = native[1] * torch.tensor(masks[l], device=args.device)[native[2]]
                        torch.testing.assert_close(hooked[1], expected, rtol=0, atol=0)
                        saw_drop |= bool((hooked[1] != native[1]).any())
                require(saw_drop, "validation probe did not exercise dropped contributions")
                validation["fixed_hidden_mask_probe"] = "native logits/IDs preserved; weights exactly masked; dropped contribution exercised"
                del probe, native, hooked, expected
                save_json(out / "validation.json", validation)
                del original, hidden
                original, hidden = collect(backbone, gates, ids, partition, None, experts, topk, args.device)
            del hidden
            archive = []
            for name, masks in [("original", None)] + list(selections.items()):
                if name == "original":
                    routes = original
                else:
                    routes, hidden = collect(backbone, gates, ids, partition, masks, experts, topk, args.device)
                    del hidden
                merge(totals[name], statistics(routes, partition, masks, original, experts, topk))
                for l, (route_ids, weights) in routes.items():
                    archive.append((name, l, route_ids, weights))
                if name != "original":
                    merge(totals[name.replace("keep", "counterfactual")], statistics(original, partition, masks, original, experts, topk))
            route_path = out / f"question_{completed:05d}.jsonl.gz"
            with gzip.open(route_path, "xt") as stream:
                for name, l, route_ids, weights in archive:
                    stream.write(json.dumps(dict(variant=name, layer=l, ids=route_ids.tolist(),
                                                 weights=weights.tolist()), allow_nan=False) + "\n")
            tokens.write(json.dumps(dict(index=completed, dataset_line=line_number, input_ids=ids, partition=partition)) + "\n")
            tokens.flush()
            os.fsync(tokens.fileno())
            progress.write(json.dumps(dict(index=completed, tokens=len(ids), routes_sha256=digest(route_path))) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
            completed += 1
            print(f"Completed {completed} questions ({len(ids)} tokens)", flush=True)
            del archive, routes, original
    require(completed > 0, "empty dataset")
    require(router_before == router_hash(), "router weights changed")
    for total in totals.values():
        require(len(total) == layers * 2, "aggregate coverage failure")
        finalize(total, topk)
    summary = dict(questions=completed, pilot=args.limit is not None, validation=validation,
                   router_unchanged=True, tokens_sha256=digest(out / "tokens.jsonl"), variants=totals)
    save_json(out / "summary.pending.json", summary)
    os.replace(out / "summary.pending.json", out / "summary.json")


if __name__ == "__main__":
    main()
