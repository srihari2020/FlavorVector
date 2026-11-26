import json
import math
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.engine import FlavorEngine  # noqa: E402

ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
FENCE = "`" * 3

BENCHMARK_CASES = [
    {"query": "what can replace buttermilk and give the same tanginess in pancakes?", "vegan_only": False, "expected_any": ["lemon juice", "yogurt"]},
    {"query": "vegan substitute for buttermilk in cornbread", "vegan_only": True, "expected_any": ["soy milk", "apple cider vinegar"]},
    {"query": "replace egg as a binder in meatballs", "vegan_only": True, "expected_any": ["flaxseed"]},
    {"query": "replace egg for airy lift and leavening in sponge cake", "vegan_only": True, "expected_any": ["baking soda"]},
    {"query": "vegan substitute for whipped egg whites in meringue", "vegan_only": True, "expected_any": ["aquafaba"]},
    {"query": "what can I use instead of sugar in a banana shake?", "vegan_only": True, "expected_any": ["dates", "maple syrup", "jaggery"]},
    {"query": "unrefined Indian sweetener replacement for brown sugar in kheer", "vegan_only": True, "expected_any": ["jaggery"]},
    {"query": "substitute for tamarind paste in sambar", "vegan_only": True, "expected_any": ["amchur", "lime juice"]},
    {"query": "what replaces kokum in coastal fish curry?", "vegan_only": True, "expected_any": ["tamarind", "pomegranate"]},
    {"query": "dry citrus spice alternative for ground sumac in fattoush", "vegan_only": True, "expected_any": ["lemon zest", "citric acid"]},
    {"query": "vegan substitute for fish sauce umami in stir-fry", "vegan_only": True, "expected_any": ["soy sauce", "nori", "mushroom"]},
    {"query": "vegetarian replacement for oyster sauce in noodles", "vegan_only": True, "expected_any": ["mushroom stir-fry", "hoisin"]},
    {"query": "what can replace miso paste in ramen broth?", "vegan_only": True, "expected_any": ["tahini", "nutritional yeast"]},
    {"query": "dairy-free substitute for grated parmesan cheese on pasta", "vegan_only": True, "expected_any": ["nutritional yeast"]},
    {"query": "vegan replacement for anchovy fillets in puttanesca sauce", "vegan_only": True, "expected_any": ["capers", "miso"]},
    {"query": "substitute for asafoetida hing in dal tadka", "vegan_only": True, "expected_any": ["garlic powder", "onion powder"]},
    {"query": "what can replace gochujang chili paste in bibimbap?", "vegan_only": True, "expected_any": ["red miso", "gochugaru", "cayenne"]},
    {"query": "substitute for chipotle in adobo in tacos", "vegan_only": True, "expected_any": ["smoked paprika"]},
    {"query": "what replaces kasuri methi dried fenugreek leaves in butter masala?", "vegan_only": True, "expected_any": ["celery leaves", "fennel"]},
    {"query": "substitute for fresh curry leaves in South Indian tadka", "vegan_only": True, "expected_any": ["bay leaf", "lime zest"]},
    {"query": "what can I use instead of coconut chutney with masala dosa?", "vegan_only": True, "expected_any": ["peanut", "chana dal", "kara chutney"]},
    {"query": "vegan substitute for paneer in palak paneer", "vegan_only": True, "expected_any": ["tofu"]},
    {"query": "dairy-free heavy cream substitute for tomato gravy", "vegan_only": True, "expected_any": ["cashews", "coconut cream"]},
    {"query": "glossy sauce thickener replacement for cornstarch", "vegan_only": True, "expected_any": ["arrowroot", "potato starch"]},
    {"query": "vegan meaty substitute for boneless chicken in biryani", "vegan_only": True, "expected_any": ["soya chunks", "jackfruit"]},
]


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


def compute_rank_metrics(ranked_subs: list[str], expected_any: list[str]) -> tuple[int, float, float]:
    found_rank = 0
    for idx, sub in enumerate(ranked_subs[:3], start=1):
        sub_l = sub.lower()
        if any(kw.lower() in sub_l for kw in expected_any):
            found_rank = idx
            break
    rr = (1.0 / found_rank) if found_rank > 0 else 0.0
    ndcg = (1.0 / math.log2(found_rank + 1)) if found_rank > 0 else 0.0
    return found_rank, rr, ndcg


def evaluate_models(engine: FlavorEngine) -> dict:
    engine.search_substitutes("warm up query", force_live=False, top_k=3)
    engine.suggest_pairings(["garlic", "lemon"], force_live=False, top_k=5)
    engine.find_flavor_bridges(["tomato", "garlic", "paneer"], top_k=5)

    bi_h1 = bi_h3 = 0
    bi_rrs, bi_ndcgs = [], []

    rerank_h1 = rerank_h3 = 0
    rerank_rrs, rerank_ndcgs = [], []

    stage1_ms_list = []
    stage2_ms_list = []
    total_sub_ms_list = []

    for case in BENCHMARK_CASES:
        q = case["query"]
        where_filter = {"vegan": True} if case["vegan_only"] else None

        t0 = time.perf_counter()
        q_vec = engine.bi_encoder.encode([q], normalize_embeddings=True).tolist()
        raw = engine.sub_collection.query(query_embeddings=q_vec, n_results=15, where=where_filter)
        t1 = time.perf_counter()

        docs = raw["documents"][0]
        metas = raw["metadatas"][0]

        bi_subs = [m.get("substitute", "") for m in metas[:3]]
        r_bi, rr_bi, ndcg_bi = compute_rank_metrics(bi_subs, case["expected_any"])
        if r_bi == 1:
            bi_h1 += 1
        if 1 <= r_bi <= 3:
            bi_h3 += 1
        bi_rrs.append(rr_bi)
        bi_ndcgs.append(ndcg_bi)

        t2 = time.perf_counter()
        scores = engine.cross_encoder.predict([(q, d) for d in docs])
        ranked = sorted(zip(metas, scores), key=lambda x: float(x[1]), reverse=True)
        t3 = time.perf_counter()

        ce_subs = [m.get("substitute", "") for m, _ in ranked[:3]]
        r_ce, rr_ce, ndcg_ce = compute_rank_metrics(ce_subs, case["expected_any"])
        if r_ce == 1:
            rerank_h1 += 1
        if 1 <= r_ce <= 3:
            rerank_h3 += 1
        rerank_rrs.append(rr_ce)
        rerank_ndcgs.append(ndcg_ce)

        s1 = (t1 - t0) * 1000.0
        s2 = (t3 - t2) * 1000.0
        stage1_ms_list.append(s1)
        stage2_ms_list.append(s2)
        total_sub_ms_list.append(s1 + s2)

    pair_queries = [
        ["garlic", "lemon", "thyme"],
        ["cumin", "coriander", "turmeric"],
        ["dark chocolate", "espresso"],
        ["ginger", "soy sauce", "sesame oil"],
        ["tomato", "basil", "olive oil"],
    ] * 3
    pair_ms = []
    for ings in pair_queries:
        t0 = time.perf_counter()
        engine.suggest_pairings(ings, force_live=False, top_k=6)
        pair_ms.append((time.perf_counter() - t0) * 1000.0)

    bridge_queries = [
        ["tomato", "garlic", "paneer"],
        ["lemon", "olive oil", "chickpeas"],
        ["ginger", "coconut milk", "lime"],
        ["butter", "thyme", "mushroom"],
        ["cardamom", "cinnamon", "milk"],
    ] * 3
    bridge_ms = []
    for pantry in bridge_queries:
        t0 = time.perf_counter()
        engine.find_flavor_bridges(pantry, top_k=5)
        bridge_ms.append((time.perf_counter() - t0) * 1000.0)

    cpu_mean_ms = None
    if engine.device == "cuda":
        cpu_bi = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        cpu_ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        sample_qs = [c["query"] for c in BENCHMARK_CASES[:5]]
        cpu_times = []
        for q in sample_qs:
            t0 = time.perf_counter()
            qv = cpu_bi.encode([q], normalize_embeddings=True).tolist()
            raw = engine.sub_collection.query(query_embeddings=qv, n_results=15)
            cpu_ce.predict([(q, d) for d in raw["documents"][0]])
            cpu_times.append((time.perf_counter() - t0) * 1000.0)
        cpu_mean_ms = round(float(np.mean(cpu_times)), 2)

    n = len(BENCHMARK_CASES)
    return {
        "bi_encoder": {
            "hit_at_1": round(bi_h1 / n * 100, 1),
            "hit_at_3": round(bi_h3 / n * 100, 1),
            "mrr": round(float(np.mean(bi_rrs)) * 100, 1),
            "ndcg_at_3": round(float(np.mean(bi_ndcgs)) * 100, 1),
        },
        "two_stage": {
            "hit_at_1": round(rerank_h1 / n * 100, 1),
            "hit_at_3": round(rerank_h3 / n * 100, 1),
            "mrr": round(float(np.mean(rerank_rrs)) * 100, 1),
            "ndcg_at_3": round(float(np.mean(rerank_ndcgs)) * 100, 1),
        },
        "latency": {
            "sub_stage1_mean_ms": round(float(np.mean(stage1_ms_list)), 2),
            "sub_stage2_mean_ms": round(float(np.mean(stage2_ms_list)), 2),
            "sub_total_p50_ms": round(float(np.percentile(total_sub_ms_list, 50)), 2),
            "sub_total_p95_ms": round(float(np.percentile(total_sub_ms_list, 95)), 2),
            "sub_total_mean_ms": round(float(np.mean(total_sub_ms_list)), 2),
            "pair_p50_ms": round(float(np.percentile(pair_ms, 50)), 2),
            "pair_p95_ms": round(float(np.percentile(pair_ms, 95)), 2),
            "pair_mean_ms": round(float(np.mean(pair_ms)), 2),
            "bridge_p50_ms": round(float(np.percentile(bridge_ms, 50)), 2),
            "bridge_p95_ms": round(float(np.percentile(bridge_ms, 95)), 2),
            "bridge_mean_ms": round(float(np.mean(bridge_ms)), 2),
            "cpu_sub_mean_ms": cpu_mean_ms,
        },
        "throughput_qps": {
            "substitutions": round(1000.0 / float(np.mean(total_sub_ms_list)), 1),
            "pairings": round(1000.0 / float(np.mean(pair_ms)), 1),
            "bridges": round(1000.0 / float(np.mean(bridge_ms)), 1),
        },
    }


def collect_atlas_stats(engine: FlavorEngine) -> dict:
    sub_metas = engine.sub_collection.get(include=["metadatas"]).get("metadatas") or []
    pair_metas = engine.pair_collection.get(include=["metadatas"]).get("metadatas") or []

    taste_counts = Counter()
    cuisine_counts = Counter()

    for m in sub_metas + pair_metas:
        for t in m.get("tastes", "").split(","):
            t_clean = t.strip().lower()
            if t_clean:
                taste_counts[t_clean] += 1
        c_clean = m.get("cuisine", "").strip()
        if c_clean:
            cuisine_counts[c_clean] += 1

    return {
        "sub_count": len(sub_metas),
        "pair_count": len(pair_metas),
        "total_vectors": len(sub_metas) + len(pair_metas),
        "tastes": dict(taste_counts),
        "cuisines": dict(cuisine_counts),
    }


def plot_benchmark_dashboard(report: dict, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#e6edf3",
            "text.color": "#e6edf3",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#21262d",
            "font.sans-serif": "DejaVu Sans",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), dpi=300)
    fig.subplots_adjust(hspace=0.36, wspace=0.25, top=0.87, bottom=0.08, left=0.07, right=0.96)

    device_label = report["system"]["device"].upper()
    gpu_name = report["system"].get("gpu_name") or device_label
    fig.suptitle(
        f"FlavorVector Base Engine — Speed, Accuracy & System Benchmarks ({gpu_name})",
        fontsize=15,
        fontweight="bold",
        color="#f0f6fc",
        y=0.95,
    )
    fig.text(
        0.5,
        0.91,
        f"25 Culinary Edge Cases · {report['atlas']['total_vectors']} Indexed Vectors · "
        f"Cold Boot: {report['system']['cold_boot_sec']}s · Index Footprint: {report['system']['chroma_mb']} MB",
        ha="center",
        fontsize=10.5,
        color="#8b949e",
    )

    ax1 = axes[0, 0]
    metrics = ["Hit Rate@1", "Hit Rate@3", "MRR", "NDCG@3"]
    bi_vals = [
        report["accuracy"]["bi_encoder"]["hit_at_1"],
        report["accuracy"]["bi_encoder"]["hit_at_3"],
        report["accuracy"]["bi_encoder"]["mrr"],
        report["accuracy"]["bi_encoder"]["ndcg_at_3"],
    ]
    ce_vals = [
        report["accuracy"]["two_stage"]["hit_at_1"],
        report["accuracy"]["two_stage"]["hit_at_3"],
        report["accuracy"]["two_stage"]["mrr"],
        report["accuracy"]["two_stage"]["ndcg_at_3"],
    ]
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax1.bar(x - w / 2, bi_vals, w, label="Stage 1: BGE-Base Only", color="#388bfd")
    b2 = ax1.bar(x + w / 2, ce_vals, w, label="Two-Stage: + Cross-Encoder", color="#3fb950")
    ax1.set_title("Retrieval Accuracy Ablation (%)", fontsize=12, fontweight="bold", pad=10, color="#f0f6fc")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9.5)
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=8.5, loc="lower right")
    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{bar.get_height():.0f}%", ha="center", fontsize=8.5, color="#79c0ff")
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{bar.get_height():.0f}%", ha="center", fontsize=8.5, fontweight="bold", color="#56d364")

    ax2 = axes[0, 1]
    lat = report["latency"]
    endpoints = ["Substitutions\n(/substitute)", "Flavor Pairing\n(/pair)", "Flavor Bridge\n(/bridge)"]
    p50_vals = [lat["sub_total_p50_ms"], lat["pair_p50_ms"], lat["bridge_p50_ms"]]
    p95_vals = [lat["sub_total_p95_ms"], lat["pair_p95_ms"], lat["bridge_p95_ms"]]
    x2 = np.arange(len(endpoints))
    b3 = ax2.bar(x2 - w / 2, p50_vals, w, label="p50 Median Latency (ms)", color="#a371f7")
    b4 = ax2.bar(x2 + w / 2, p95_vals, w, label="p95 Tail Latency (ms)", color="#f78166")
    ax2.set_title("End-to-End Query Latency (Lower is Better)", fontsize=12, fontweight="bold", pad=10, color="#f0f6fc")
    ax2.set_ylabel("Milliseconds (ms)", fontsize=9.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(endpoints, fontsize=9.5)
    ax2.set_ylim(0, max(p95_vals) * 1.28)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=8.5, loc="upper left")
    for bar in b3:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{bar.get_height():.1f}ms", ha="center", fontsize=8.5, color="#d2a8ff")
    for bar in b4:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{bar.get_height():.1f}ms", ha="center", fontsize=8.5, color="#ffa198")

    ax3 = axes[1, 0]
    if lat["cpu_sub_mean_ms"]:
        labels_hw = [
            f"GPU ({device_label})\nStage 1 + Stage 2",
            "CPU Fallback\nStage 1 + Stage 2",
        ]
        s1_gpu = lat["sub_stage1_mean_ms"]
        s2_gpu = lat["sub_stage2_mean_ms"]
        cpu_total = lat["cpu_sub_mean_ms"]
        speedup = cpu_total / max(0.1, (s1_gpu + s2_gpu))

        ax3.barh([0], [s1_gpu], height=0.45, color="#388bfd", label="Stage 1: Bi-Encoder + ChromaDB")
        ax3.barh([0], [s2_gpu], left=[s1_gpu], height=0.45, color="#3fb950", label="Stage 2: Cross-Encoder Rerank")
        ax3.barh([1], [cpu_total], height=0.45, color="#d29922", label=f"CPU End-to-End ({speedup:.1f}x slower)")
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(labels_hw, fontsize=9.5)
        ax3.set_xlabel("Mean Query Time (ms)", fontsize=9.5)
        ax3.set_xlim(0, cpu_total * 1.22)
        ax3.set_title(f"Stage Breakdown & GPU Acceleration ({speedup:.1f}x Speedup)", fontsize=12, fontweight="bold", pad=10, color="#f0f6fc")
        ax3.text(s1_gpu + s2_gpu + 1.5, 0, f"{s1_gpu + s2_gpu:.1f} ms", va="center", fontsize=9, fontweight="bold", color="#56d364")
        ax3.text(cpu_total + 1.5, 1, f"{cpu_total:.1f} ms", va="center", fontsize=9, fontweight="bold", color="#e3b341")
    else:
        s1 = lat["sub_stage1_mean_ms"]
        s2 = lat["sub_stage2_mean_ms"]
        ax3.barh(["Substitution\nPipeline"], [s1], height=0.4, color="#388bfd", label="Stage 1: Bi-Encoder + ChromaDB")
        ax3.barh(["Substitution\nPipeline"], [s2], left=[s1], height=0.4, color="#3fb950", label="Stage 2: Cross-Encoder Rerank")
        ax3.set_xlabel("Mean Latency (ms)")
        ax3.set_xlim(0, (s1 + s2) * 1.3)
        ax3.set_title("Two-Stage Retrieval Latency Split", fontsize=12, fontweight="bold", pad=10, color="#f0f6fc")
    ax3.grid(axis="x", linestyle="--", alpha=0.6)
    ax3.legend(frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=8, loc="lower right")

    ax4 = axes[1, 1]
    qps = report["throughput_qps"]
    qps_labels = ["Substitutions", "Flavor Pairing", "Flavor Bridge"]
    qps_vals = [qps["substitutions"], qps["pairings"], qps["bridges"]]
    colors_qps = ["#3fb950", "#388bfd", "#a371f7"]
    bars_qps = ax4.bar(qps_labels, qps_vals, width=0.48, color=colors_qps)
    ax4.set_title("Sustained Query Throughput (Queries / Sec)", fontsize=12, fontweight="bold", pad=10, color="#f0f6fc")
    ax4.set_ylabel("Queries per Second (QPS)", fontsize=9.5)
    ax4.set_ylim(0, max(qps_vals) * 1.28)
    ax4.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars_qps:
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{bar.get_height():.1f} QPS", ha="center", fontsize=9, fontweight="bold", color="#f0f6fc")

    sys_info = report["system"]
    foot_text = (
        f"Peak RAM: {sys_info['peak_ram_mb']} MB  |  "
        f"GPU VRAM: {sys_info['vram_mb']} MB  |  "
        f"ChromaDB Disk: {sys_info['chroma_mb']} MB"
    )
    ax4.text(
        0.5,
        0.90,
        foot_text,
        transform=ax4.transAxes,
        ha="center",
        fontsize=8.3,
        color="#c9d1d9",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d1117", edgecolor="#30363d"),
    )

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_taste_coverage(atlas: dict, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#e6edf3",
            "text.color": "#e6edf3",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#21262d",
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.14, left=0.13, right=0.97)
    fig.suptitle(
        "FlavorVector Knowledge Graph — 10-Taste Sensory & Global Cuisine Coverage",
        fontsize=14,
        fontweight="bold",
        color="#f0f6fc",
        y=0.94,
    )

    taste_order = [
        ("sweet", "Sweet"),
        ("salty", "Salty"),
        ("sour", "Sour"),
        ("bitter", "Bitter"),
        ("umami", "Umami"),
        ("spicy_pungent", "Spicy / Pungent"),
        ("astringent", "Astringent"),
        ("cooling", "Cooling"),
        ("fatty_rich", "Fatty / Rich"),
        ("kokumi", "Kokumi"),
    ]
    t_labels = [lbl for _, lbl in reversed(taste_order)]
    t_vals = [atlas["tastes"].get(k, 0) for k, _ in reversed(taste_order)]
    palette_t = ["#388bfd", "#58a6ff", "#3fb950", "#56d364", "#a371f7", "#d2a8ff", "#f78166", "#ffa657", "#e3b341", "#79c0ff"]

    bars1 = ax1.barh(t_labels, t_vals, color=palette_t, height=0.62)
    ax1.set_title("Indexed Profiles by Taste Dimension", fontsize=11.5, fontweight="bold", pad=10, color="#f0f6fc")
    ax1.set_xlabel("Tagged Vector Profiles", fontsize=9.5)
    ax1.set_xlim(0, max(t_vals + [5]) * 1.2)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)
    for bar in bars1:
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, str(int(bar.get_width())), va="center", fontsize=8.8, color="#e6edf3")

    sorted_cuisines = sorted(atlas["cuisines"].items(), key=lambda x: x[1])
    c_labels = [k for k, _ in sorted_cuisines]
    c_vals = [v for _, v in sorted_cuisines]
    bars2 = ax2.barh(c_labels, c_vals, color="#2ea043", height=0.58)
    ax2.set_title("Indexed Profiles by Global Culinary Corner", fontsize=11.5, fontweight="bold", pad=10, color="#f0f6fc")
    ax2.set_xlabel("Tagged Vector Profiles", fontsize=9.5)
    ax2.set_xlim(0, max(c_vals + [5]) * 1.2)
    ax2.grid(axis="x", linestyle="--", alpha=0.6)
    for bar in bars2:
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, str(int(bar.get_width())), va="center", fontsize=8.8, color="#e6edf3")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def update_readme(report: dict) -> None:
    acc = report["accuracy"]["two_stage"]
    bi_acc = report["accuracy"]["bi_encoder"]
    lat = report["latency"]
    qps = report["throughput_qps"]
    sys_info = report["system"]
    atlas = report["atlas"]

    readme = f"""# FlavorVector

![FlavorVector Benchmarks](assets/benchmark_dashboard.png)

![FlavorVector 10-Taste & Cuisine Coverage](assets/taste_coverage.png)

A local-first, self-expanding **Semantic Cooking Substitution & Flavor-Pairing Engine** powered by food-science chemistry, volatile aroma compounds, and two-stage neural retrieval.

## Benchmark Summary (Base Engine)
Evaluated locally on **`{sys_info['gpu_name'] or sys_info['device'].upper()}`** across **25 culinary edge cases** covering all **10 taste & mouthfeel dimensions** and **7 global culinary regions**:

| Category | Metric | Stage 1 (Bi-Encoder) | Two-Stage (+ Cross-Encoder) |
| :--- | :--- | :---: | :---: |
| **Accuracy** | **Hit Rate@1** | {bi_acc['hit_at_1']}% | **{acc['hit_at_1']}%** |
| **Accuracy** | **Hit Rate@3** | {bi_acc['hit_at_3']}% | **{acc['hit_at_3']}%** |
| **Ranking Quality** | **Mean Reciprocal Rank (MRR)** | {bi_acc['mrr']}% | **{acc['mrr']}%** |
| **Ranking Quality** | **NDCG@3** | {bi_acc['ndcg_at_3']}% | **{acc['ndcg_at_3']}%** |
| **Latency** | **Substitutions (`/substitute` p50 / p95)** | {lat['sub_stage1_mean_ms']} ms (Stage 1) | **{lat['sub_total_p50_ms']} ms / {lat['sub_total_p95_ms']} ms** |
| **Latency** | **Flavor Pairing (`/pair` p50 / p95)** | — | **{lat['pair_p50_ms']} ms / {lat['pair_p95_ms']} ms** |
| **Latency** | **Pantry Flavor Bridge (`/bridge` p50 / p95)** | — | **{lat['bridge_p50_ms']} ms / {lat['bridge_p95_ms']} ms** |
| **Throughput** | **Sustained QPS** | — | **{qps['substitutions']} QPS** (`/substitute`) · **{qps['pairings']} QPS** (`/pair`) |
| **Footprint** | **Vectors / Disk / VRAM** | — | **{atlas['total_vectors']} vectors** · **{sys_info['chroma_mb']} MB** · **{sys_info['vram_mb']} MB VRAM** |

## Core Capabilities
- **Role-Based Substitution Search:** Matches chemical & structural function (acidity for baking soda leavening, mucilage binding vs. foam whipping, glutamate-ribonucleotide umami synergy) rather than naive ingredient similarity.
- **10-Taste Sensory & Global Cuisine Atlas:** Indexes **Sweet, Salty, Sour, Bitter, Umami, Spicy/Pungent, Astringent, Cooling, Fatty/Rich, and Kokumi** profiles across Indian & South Asian, East & Southeast Asian, Mediterranean, Latin American, Middle Eastern, and Baking kitchens.
- **Pantry Flavor-Bridge Solver:** Finds the single missing bridge ingredient that connects 2–8 ingredients already in your kitchen via shared volatile aroma compounds.
- **Kitchen Fraction Scaler & Live Web Learning:** Scales recipes cleanly (`0.5x`–`3x`) into kitchen fractions and auto-indexes unseen culinary queries from Wikipedia, TheMealDB, Open Food Facts, and local Ollama.

## Quickstart
{FENCE}bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the Universal 10-Taste & Global Cuisine ChromaDB Index
python3 scripts/expand_global_atlas.py

# 2. Launch the Interactive Streamlit App (http://localhost:8501)
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501

# 3. Or launch the FastAPI REST Service (http://localhost:8000/docs)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
{FENCE}
"""
    (PROJECT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_boot_start = time.perf_counter()
    engine = FlavorEngine()
    cold_boot_sec = round(time.perf_counter() - t_boot_start, 2)

    eval_data = evaluate_models(engine)
    atlas_stats = collect_atlas_stats(engine)

    peak_ram_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0.0
    )
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    report = {
        "system": {
            "device": engine.device,
            "gpu_name": gpu_name,
            "cold_boot_sec": cold_boot_sec,
            "peak_ram_mb": peak_ram_mb,
            "vram_mb": vram_mb,
            "chroma_mb": dir_size_mb(PROJECT_ROOT / "chroma_db"),
        },
        "atlas": atlas_stats,
        **eval_data,
    }

    json_out = PROJECT_ROOT / "tests" / "full_benchmark.json"
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    dash_png = ASSETS_DIR / "benchmark_dashboard.png"
    taste_png = ASSETS_DIR / "taste_coverage.png"
    plot_benchmark_dashboard(report, dash_png)
    plot_taste_coverage(atlas_stats, taste_png)
    update_readme(report)

    print(f"Generated {dash_png}")
    print(f"Generated {taste_png}")
    print("Updated README.md with top-level PNG benchmark charts.")


if __name__ == "__main__":
    main()
