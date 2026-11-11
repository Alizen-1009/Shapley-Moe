"""Plot archived selected routes and surviving contributions, not executed work."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

VARIANTS = ("original", "keep_08", "keep_06")
LABELS = ("Original", "Keep 0.8", "Keep 0.6")
PARTITIONS = ("prompt", "answer")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.archive / "manifest.json").read_text())
    summary = json.loads((args.archive / "summary.json").read_text())
    masks = {v: np.asarray(manifest["selections"][v], dtype=bool) for v in VARIANTS[1:]}
    layers, experts = masks["keep_08"].shape
    stats = summary["variants"]
    if layers != 48 or experts != 128 or masks["keep_06"].shape != (layers, experts):
        raise ValueError("expected 48 layers and 128 experts")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "svg.fonttype": "none"})
    args.output.mkdir(parents=True, exist_ok=False)
    pilot = " [PILOT]" if summary["pilot"] else ""

    def save(fig, name):
        fig.savefig(args.output / (name + ".svg"))
        fig.savefig(args.output / (name + ".png"), dpi=200)
        plt.close(fig)

    def matrix(v, p, field):
        return np.asarray([stats[v][f"{l}/{p}"][field] for l in range(layers)], dtype=float)

    def frequency(v, p, kind):
        return matrix(v, p, kind + "_counts") / matrix(v, p, "tokens")[:, None]

    def heatmaps(arrays, titles, name, title, label, diverging=False):
        # Every heatmap is an editable vector QuadMesh, with a shared scale.
        fig, axes = plt.subplots(1, len(arrays), figsize=(7, 3.3), layout="constrained", squeeze=False)
        if diverging:
            bound = max(max(float(np.abs(a).max()) for a in arrays), 1e-12)
            norm, cmap = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), "coolwarm"
        else:
            norm, cmap = Normalize(0, max(max(float(a.max()) for a in arrays), 1e-12)), "viridis"
        for ax, a, caption in zip(axes[0], arrays, titles):
            mesh = ax.pcolormesh(np.arange(experts+1), np.arange(layers+1), a,
                                 cmap=cmap, norm=norm, shading="flat", rasterized=False)
            ax.set(title=caption, xlabel="Expert ID", xlim=(0, experts), ylim=(layers, 0))
            ax.set_xticks([0, 32, 64, 96, 127])
            ax.set_yticks([0, 12, 24, 36, 47])
            ax.set_ylabel("Layer ID")
        colorbar = fig.colorbar(mesh, ax=axes[0].tolist(), label=label, shrink=.85)
        colorbar.solids.set_rasterized(False)
        fig.suptitle(title + pilot)
        save(fig, name)

    # Shared across all three variants AND both partitions, for each quantity.
    for kind, title in (("selected", "Selected pre-drop frequency"),
                        ("surviving", "Surviving contribution frequency (not executed work)")):
        all_arrays = [frequency(v, p, kind) for p in PARTITIONS for v in VARIANTS]
        upper = max(float(a.max()) for a in all_arrays)
        fig, axes = plt.subplots(2, 3, figsize=(7, 5.8), layout="constrained")
        for ax, a, (p, label) in zip(axes.flat, all_arrays,
                                    [(p, label) for p in PARTITIONS for label in LABELS]):
            mesh = ax.pcolormesh(np.arange(experts+1), np.arange(layers+1), a,
                                 cmap="viridis", vmin=0, vmax=max(upper, 1e-12), rasterized=False)
            ax.set(title=f"{label}: {p}", xlabel="Expert ID", ylabel="Layer ID", ylim=(layers, 0))
            ax.set_xticks([0, 64, 127])
            ax.set_yticks([0, 24, 47])
        colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Selections per token" if kind == "selected" else "Surviving contributions per token", shrink=.8)
        colorbar.solids.set_rasterized(False)
        fig.suptitle(title + pilot)
        save(fig, kind + "_frequency")
        # One figure per partition; two intervention deltas share a symmetric scale.
        for p in PARTITIONS:
            heatmaps([frequency(v, p, kind)-frequency("original", p, kind) for v in VARIANTS[1:]],
                     ["Keep 0.8 − original", "Keep 0.6 − original"], f"{kind}_delta_{p}",
                     f"{title}: {p} change", "Change per token", True)
    heatmaps([masks[v].astype(float) for v in VARIANTS[1:]],
             [f"{label}: {manifest['actual_retained_fractions'][v]:.3%} retained"
              for v, label in zip(VARIANTS[1:], LABELS[1:])],
             "retained_masks", "Retained expert masks", "Retained (1) / removed (0)")

    metrics = [("mean_survivors", "Mean surviving contributions / token"),
               ("dropped_weight_fraction", "Dropped pre-drop weight fraction"),
               ("overlap_fraction", "Top-8 overlap with original"),
               ("dropped_selection_fraction", "Dropped selected contribution fraction")]
    colors = ("#21918c", "#443983")
    for p in PARTITIONS:
        fig, axes = plt.subplots(2, 2, figsize=(7, 5), layout="constrained")
        for ax, (field, label) in zip(axes.flat, metrics):
            for v, caption, color, marker in zip(VARIANTS[1:], LABELS[1:], colors, ('o', 's')):
                ax.plot(np.arange(layers), matrix(v, p, field), label=caption, color=color,
                        marker=marker, markevery=6, markersize=3)
                cf = v.replace("keep", "counterfactual")
                ax.plot(np.arange(layers), matrix(cf, p, field), color=color, linestyle="--",
                        label=caption + " mask-only")
            ax.set(xlabel="Layer ID", ylabel=label, xlim=(0, layers-1))
            ax.set_ylim(0, 8 if field == 'mean_survivors' else 1)
            ax.grid(alpha=.2)
        axes[0, 0].legend()
        fig.suptitle(f"{p.capitalize()}: surviving contributions, not executed work" + pilot)
        save(fig, "layer_metrics_" + p)

    per_layer, aggregates = [], {}
    for v, variant in stats.items():
        aggregates[v] = {}
        for p in PARTITIONS:
            records = [variant[f"{l}/{p}"] for l in range(layers)]
            for l, s in enumerate(records):
                per_layer.append(dict(variant=v, partition=p, layer=l,
                    **{k: value for k, value in s.items() if not isinstance(value, list)}))
            tokens = sum(s["tokens"] for s in records)
            selected = sum(sum(s["selected_counts"]) for s in records)
            kept = sum(sum(s["surviving_counts"]) for s in records)
            weight = sum(sum(s["selected_weights"]) for s in records)
            surviving_weight = sum(sum(s["surviving_weights"]) for s in records)
            overlap = sum(s["overlap_count"] for s in records)
            aggregates[v][p] = dict(token_layer_count=tokens, selected_count=selected,
                surviving_count=kept, selected_weight=weight, surviving_weight=surviving_weight,
                overlap_count=overlap, mean_survivors=kept/tokens,
                dropped_selection_fraction=(selected-kept)/selected,
                dropped_weight_fraction=(weight-surviving_weight)/weight,
                overlap_fraction=overlap/selected)
    for variant in aggregates.values():
        combined = {key: sum(variant[p][key] for p in PARTITIONS)
                    for key in ('token_layer_count', 'selected_count', 'surviving_count',
                                'selected_weight', 'surviving_weight', 'overlap_count')}
        combined.update(mean_survivors=combined['surviving_count']/combined['token_layer_count'],
                        dropped_selection_fraction=1-combined['surviving_count']/combined['selected_count'],
                        dropped_weight_fraction=1-combined['surviving_weight']/combined['selected_weight'],
                        overlap_fraction=combined['overlap_count']/combined['selected_count'])
        variant['combined'] = combined
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.3), layout='constrained', sharey=True)
    for ax, p in zip(axes, PARTITIONS):
        x = np.arange(3)
        for j, (v, label, color, hatch) in enumerate(zip(VARIANTS[1:], LABELS[1:], colors, ('', '//'))):
            a = aggregates[v][p]
            values = 100*np.asarray([1-manifest['actual_retained_fractions'][v],
                                     a['dropped_selection_fraction'], a['dropped_weight_fraction']])
            bars = ax.bar(x+(j-.5)*.36, values, .36, label=label, color=color,
                          hatch=hatch, edgecolor='white', linewidth=.4)
            ax.bar_label(bars, labels=[f'{z:.2f}%' for z in values], fontsize=8, padding=2)
        ax.set_xticks(x, ['Stored experts\nremoved', 'Selected slots\ndropped', 'Routing weight\ndropped'])
        upper = max(55, 1.2*max(100*z for v in VARIANTS[1:]
                    for z in (1-manifest['actual_retained_fractions'][v],
                              aggregates[v][p]['dropped_selection_fraction'],
                              aggregates[v][p]['dropped_weight_fraction'])))
        ax.set(title=p.capitalize(), ylim=(0, upper), ylabel='Reduction (%)')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend()
    fig.suptitle('Expert count reduction is not routed-contribution reduction' + pilot)
    save(fig, 'count_vs_contribution_reduction')
    with (args.output / "per_layer.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_layer[0]))
        writer.writeheader()
        writer.writerows(per_layer)
    with (args.output / "global_aggregates.json").open("x") as stream:
        json.dump(dict(questions=summary["questions"], pilot=summary["pilot"],
            actual_retained_fractions=manifest["actual_retained_fractions"],
            interpretation="Selected pre-drop routes and surviving contributions; not executed work. Aggregates are token-layer/count/weight weighted, not means of layer fractions.",
            variants=aggregates), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
