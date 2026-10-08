"""Generate the 15-chart enterprise figure suite from saved joint-model artifacts.

Reads only completed outputs (no API access, no refitting). For each chart this
module writes a static matplotlib PNG + SVG into figures/ and, where meaningful,
an offline interactive Plotly HTML into visualizations/. Results are indexed in
figures_manifest.json and figure_gallery.html.

Data caveats carried on every chart: records are NOT deduplicated; topic -1 is
the retained outlier class; 2026 is incomplete; topic labels are interpretive.
"""
from __future__ import annotations

import argparse
import json
import io
import re
import time
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
import pycountry
import figure_ops as ops
from text_utils import tokenize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DIR = OUT / "joint_bge_m3"
FIG = DIR / "figures"
VIZ = DIR / "visualizations"
EMBEDDINGS_PATH = OUT / "joint_bge_m3_embeddings.npy"

# Visual identity shared with the existing report.
COLORS = ["#176b87", "#d17a45", "#599977", "#9078a8", "#bd6277", "#8c8a48", "#4e93ac", "#725847",
          "#746ec9", "#c4944e", "#3f8d8c", "#ab6d43", "#5f7f9f", "#a45f7c", "#6f9a58", "#8a6db0",
          "#c07a5a", "#4c8c86", "#9c8a3f", "#7f6f9f", "#b0566f", "#56872f", "#3f7fa8", "#a8763f",
          "#6d8f8f", "#8f6d75", "#728a4f", "#6d7f9f", "#a05f4f", "#4f8f7f", "#9f7f4f", "#7f4f8f",
          "#5f8fa0", "#a08f5f", "#8f5f5f", "#5f6f8f", "#7f9f5f", "#9f5f8f", "#5f9f8f", "#9f9f5f",
          "#7f5f6f", "#6f5f9f"]
INK = "#173242"
MUTED = "#61727e"
GRID = "#dbe3e7"
NEUTRAL = "#b6bcc3"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False, "svg.fonttype": "none", "svg.hashsalt": "bertopic-figures-v2",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white",
})
FONT_PATH = None
for candidate in [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")]:
    if candidate.exists():
        font_manager.fontManager.addfont(str(candidate))
        name = font_manager.FontProperties(fname=str(candidate)).get_name()
        plt.rcParams["font.family"] = name
        FONT_PATH = str(candidate)
        break

FOOTNOTE = ("口径：四库合并未去重记录（N=22,549）；主题 -1 为保留离群类；2026 年不完整；主题标签为解释性命名。")
GENERATED_FILES: dict[str, list[str]] = {}
INTERACTIVE_CONTRACTS: dict = {}


def log(stage: str, **kwargs):
    print(json.dumps({"stage": stage, **kwargs}, ensure_ascii=False), flush=True)


def read_table(name):
    return pd.read_json(DIR / f"{name}.json", dtype={"doi": str, "pmid": str, "wos_id": str})


def save_figure(figure, name):
    (FIG / name).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white", metadata={"Date": None})
    figure.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white",
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)
    GENERATED_FILES.setdefault(name, []).extend([f"figures/{name}.{suffix}" for suffix in ("png", "svg", "pdf")])


def write_interactive(figure, name, height=None):
    figure.update_layout(
        template="plotly_white", font=dict(family="Microsoft YaHei, Arial", size=12, color=INK),
        margin=dict(l=60, r=40, t=70, b=60), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        height=height or figure.layout.height or 680)
    figure.write_html(
        VIZ / f"{name}.html", include_plotlyjs="directory", full_html=True, div_id=name,
        config={"responsive": True, "displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
    GENERATED_FILES.setdefault(name, []).append(f"visualizations/{name}.html")
    INTERACTIVE_CONTRACTS.setdefault(name, {})["traces"] = len(figure.data)


def footnote(ax=None, text=FOOTNOTE):
    target = ax if ax is not None else plt.gca()
    target.figure.text(0.01, 0.002, text, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def topic_color(topic: int) -> str:
    return NEUTRAL if topic == -1 else COLORS[int(topic) % len(COLORS)]


def short(label: str, width=26) -> str:
    text = str(label)
    return text if len(text) <= width else text[:max(1, width - 1)] + "…"


def ellip(label: str, width=14) -> str:
    text = str(label)
    return text if len(text) <= width else text[:width] + "…"


class Ctx:
    """All artifacts needed by the figure functions, loaded once."""

    def __init__(self):
        self.df = read_table("document_topics")
        self.summary = read_table("topic_summary")
        self.keywords = read_table("topic_keywords")
        self.by_year = read_table("topics_by_year")
        self.by_source = read_table("topics_by_source")
        self.by_language = read_table("topics_by_language")
        self.metrics = json.loads((DIR / "selected_model_metrics.json").read_text(encoding="utf-8"))
        self.stability = json.loads((DIR / "stability.json").read_text(encoding="utf-8"))
        self.candidates = read_table("parameter_comparison")
        self.ids = [int(v) for v in json.loads((DIR / "topic_ids.json").read_text(encoding="utf-8"))]
        labels = json.loads((DIR / "topic_labels.json").read_text(encoding="utf-8"))
        manual = json.loads((ROOT / "topic_names.json").read_text(encoding="utf-8")).get(DIR.name, {})
        self.labels = {int(k): v for k, v in {**labels, **manual}.items()}
        self.sim = np.load(DIR / "topic_similarity.npy")
        self.linkage = np.load(DIR / "hierarchy_linkage.npy")
        self.centroids = np.load(DIR / "topic_centroids.npy")
        self.df["year"] = pd.to_numeric(self.df.year, errors="coerce")
        self.ordered = self.summary.sort_values("count", ascending=False).topic.astype(int).tolist()
        self.clusters = dict(zip(self.ids, fcluster(self.linkage, t=8, criterion="maxclust").tolist()))
        self._mds = None
        self._wos_meta = None

    def name(self, topic: int, width=None) -> str:
        label = self.labels.get(int(topic), "")
        return short(label, width) if width else label

    def display(self, topic: int, width=None) -> str:
        text = f"T{int(topic)} {self.name(topic)}"
        return short(text, width) if width else text

    @property
    def mds(self):
        """Deterministic 2-D MDS projection of topic centroids (cosine distance)."""
        if self._mds is None:
            distance = np.clip(1 - self.sim, 0, 2)
            np.fill_diagonal(distance, 0)
            estimator = MDS(n_components=2, metric="precomputed", init="random", random_state=42,
                            normalized_stress=True, n_init=4, max_iter=1000, eps=1e-6)
            self._mds = estimator.fit_transform(distance)
            self.mds_stress = float(estimator.stress_)
        return self._mds

    @property
    def wos_meta(self):
        """Country + first institution per WoS record, extracted from Addresses."""
        if self._wos_meta is None:
            data_manifest = json.loads((OUT / "data_manifest.json").read_text(encoding="utf-8"))
            wos = pd.read_csv(Path(data_manifest["input_directory"]) / "wos.csv", dtype=str, keep_default_na=False,
                              usecols=["Addresses"])
            addresses = wos["Addresses"].tolist()
            if len(addresses) != int(self.df.source.eq("WoS").sum()):
                raise RuntimeError("wos.csv row count no longer matches the corpus")
            mapping = {f"WoS_{i + 1:06d}": extract_country_institution(a) for i, a in enumerate(addresses)}
            self.wos_memberships = {f"WoS_{i + 1:06d}": extract_all_addresses(a) for i, a in enumerate(addresses)}
            self._wos_meta = pd.DataFrame(
                {"record_id": list(mapping), "country": [v[0] for v in mapping.values()],
                 "institution": [v[1] for v in mapping.values()]})
        return self._wos_meta


COUNTRY_MAP = {
    "peoples r china": "中国", "china": "中国", "taiwan": "中国台湾", "hong kong": "中国香港",
    "usa": "美国", "japan": "日本", "india": "印度", "south korea": "韩国", "germany": "德国",
    "france": "法国", "australia": "澳大利亚", "brazil": "巴西", "thailand": "泰国",
    "england": "英国", "scotland": "英国", "wales": "英国", "uk": "英国", "canada": "加拿大",
    "italy": "意大利", "spain": "西班牙", "netherlands": "荷兰", "philippines": "菲律宾",
    "vietnam": "越南", "pakistan": "巴基斯坦", "bangladesh": "孟加拉国", "indonesia": "印度尼西亚",
    "malaysia": "马来西亚", "egypt": "埃及", "iran": "伊朗", "turkey": "土耳其", "mexico": "墨西哥",
    "russia": "俄罗斯", "nigeria": "尼日利亚", "kenya": "肯尼亚", "argentina": "阿根廷",
    "poland": "波兰", "sweden": "瑞典", "denmark": "丹麦", "finland": "芬兰", "norway": "挪威",
    "switzerland": "瑞士", "austria": "奥地利", "belgium": "比利时", "czech republic": "捷克",
    "singapore": "新加坡", "myanmar": "缅甸", "nepal": "尼泊尔", "sri lanka": "斯里兰卡",
    "cote ivoire": "科特迪瓦", "ivory coast": "科特迪瓦", "uzbekistan": "乌兹别克斯坦",
}


COUNTRY_ALIASES = {country.name.casefold(): country.name for country in pycountry.countries}
for country in pycountry.countries:
    if hasattr(country, "official_name"):
        COUNTRY_ALIASES[country.official_name.casefold()] = country.name
COUNTRY_ALIASES.update(COUNTRY_MAP)
COUNTRY_ALIASES.update({"united states": "美国", "united states of america": "美国",
                        "u s a": "美国", "united kingdom": "英国", "north ireland": "英国",
                        "northern ireland": "英国", "republic of korea": "韩国",
                        "south africa": "南非", "new zealand": "新西兰",
                        "peoples republic of china": "中国", "p r china": "中国"})
COUNTRY_SUFFIXES = sorted(COUNTRY_ALIASES, key=len, reverse=True)


def parse_address(address):
    parts = [part.strip() for part in address.split(",") if part.strip()]
    institution = re.sub(r"\s+", " ", parts[0]) if len(parts) >= 2 else ""
    raw = re.sub(r"[.]+", " ", parts[-1]).casefold() if parts else ""
    raw = re.sub(r"\s+", " ", raw).strip()
    country = next((COUNTRY_ALIASES[key] for key in COUNTRY_SUFFIXES
                    if raw == key or raw.endswith(" " + key)), "")
    return country, institution


def extract_all_addresses(address):
    # Author lists contain semicolons: remove brackets before splitting addresses.
    cleaned = re.sub(r"\[[^\]]*\]", "", str(address or ""))
    pairs = [parse_address(part.strip()) for part in cleaned.split(";") if part.strip()]
    return list(dict.fromkeys(pairs))


def extract_country_institution(address):
    pairs = extract_all_addresses(address)
    return pairs[0] if pairs else ("", "")


# ---------------------------------------------------------------------------
# 1. 主题总览图：主题规模 + 主题关系（气泡间距 = 语义距离，气泡大小 = 规模）
# ---------------------------------------------------------------------------
def fig01_topic_overview(c: Ctx):
    xy = c.mds
    counts = c.summary.set_index("topic")["count"].to_dict()
    sizes = np.array([counts[t] for t in c.ids], dtype=float)
    marker = 80 * np.sqrt(sizes / sizes.max())
    cluster_of = [c.clusters[t] for t in c.ids]
    palette = plt.get_cmap("tab10")

    static, ax = plt.subplots(figsize=(12.5, 8.5))
    seen = set()
    for i, topic in enumerate(c.ids):
        cluster = cluster_of[i]
        label = f"簇 {cluster}" if cluster not in seen else None
        seen.add(cluster)
        ax.scatter(*xy[i], s=marker[i] ** 2, color=palette(cluster % 10), alpha=0.75,
                   edgecolor="white", linewidth=1.2, label=label, zorder=3)
    for i, topic in enumerate(c.ids):
        if counts[topic] >= 120 or topic in c.ordered[:12]:
            ax.annotate(f"T{topic}", xy[i], ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold", zorder=4)
    ax.set(title=f"主题总览：{len(c.ids)} 个主题的规模与语义关系（MDS · 余弦距离）",
           xlabel="MDS 维度 1（主题中心语义投影）", ylabel="MDS 维度 2")
    ax.grid(color=GRID, lw=0.5, alpha=0.6)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=palette(cluster % 10),
                      markeredgecolor="white", markersize=9, label=f"层次簇 {cluster}")
               for cluster in sorted(set(cluster_of))]
    ax.legend(handles=handles, title="平均连接 8 簇", loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9)
    footnote()
    save_figure(static, "fig01_topic_overview")

    texts = [c.display(t) for t in c.ids]
    kw = c.summary.set_index("topic")["keywords"].to_dict()
    npm = c.summary.set_index("topic")["npmi_top10"].to_dict()
    share = c.summary.set_index("topic")["share_all_documents"].to_dict()
    fig = go.Figure(go.Scatter(
        x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=[f"T{t}" for t in c.ids],
        textposition="middle center", textfont=dict(size=8, color="white"),
        marker=dict(size=marker, color=[matplotlib.colors.to_hex(palette(cluster % 10)) for cluster in cluster_of],
                    opacity=0.8, line=dict(width=1.2, color="white")),
        customdata=[[texts[i], int(sizes[i]), float(share[t]), kw.get(t, "")[:60], float(npm[t])] for i, t in enumerate(c.ids)],
        hovertemplate="<b>%{customdata[0]}</b><br>记录数 %{customdata[1]:,.0f}（%{customdata[2]:.2%}）"
                      "<br>前 5 关键词：%{customdata[3]}<br>NPMI %{customdata[4]:.3f}<extra></extra>"))
    fig.update_layout(title=f"主题总览：气泡面积与记录数成正比；近似 MDS 投影（stress={c.mds_stress:.3f}）",
                      xaxis_title="MDS 维度 1", yaxis_title="MDS 维度 2")
    write_interactive(fig, "fig01_topic_overview", height=760)


# ---------------------------------------------------------------------------
# 2. 主题关键词条形图：每个主题核心关键词（c-TF-IDF 权重）
# ---------------------------------------------------------------------------
def fig02_topic_keywords(c: Ctx):
    top16 = c.ordered[:16]
    static, axes = plt.subplots(4, 4, figsize=(17, 13))
    for ax, topic in zip(axes.ravel(), top16):
        sub = c.keywords.loc[c.keywords.topic.eq(topic)].head(8).iloc[::-1]
        ax.barh(sub.keyword, sub.ctfidf_weight, color=topic_color(topic), height=0.65)
        ax.set_title(f"T{topic} · {c.name(topic, 30)}（{int(c.summary.set_index('topic').loc[topic, 'count']):,} 条）",
                     fontsize=10)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlim(0, sub.ctfidf_weight.max() * 1.15 if len(sub) else 1)
    static.suptitle("主题关键词条形图：前 16 大主题的核心关键词（c-TF-IDF 权重）", fontsize=15, y=0.995)
    static.tight_layout(rect=(0, 0.01, 1, 0.985))
    footnote()
    save_figure(static, "fig02_topic_keywords")

    # Interactive: one figure with a dropdown to switch between all 42 topics.
    frames, buttons = [], []
    for topic in c.ids:
        sub = c.keywords.loc[c.keywords.topic.eq(topic) & c.keywords.ctfidf_weight.gt(0)].head(10).iloc[::-1]
        frames.append(go.Bar(y=sub.keyword.tolist() or [""], x=sub.ctfidf_weight.tolist() or [0],
                             orientation="h", marker_color=topic_color(topic),
                             name=f"T{topic}"))
        buttons.append(dict(method="update", label=f"T{topic} {c.name(topic, 18)}",
                            args=[{"x": [frames[-1].x], "y": [frames[-1].y], "marker": [{"color": topic_color(topic)}]},
                                  {"title": f"主题关键词：T{topic} {c.name(topic)}"}]))
    fig = go.Figure(frames[0])
    fig.update_layout(title=f"主题关键词条形图：T{c.ids[0]} {c.name(c.ids[0])}",
                      xaxis_title="c-TF-IDF 权重", updatemenus=[
                          dict(type="dropdown", buttons=buttons, direction="down",
                               x=1.0, xanchor="right", y=1.16, yanchor="top", showactive=True)])
    write_interactive(fig, "fig02_topic_keywords", height=620)


# ---------------------------------------------------------------------------
# 3. 主题词云图：以 c-TF-IDF 权重为词频
# ---------------------------------------------------------------------------
def fig03_wordcloud(c: Ctx):
    topics = c.ordered[:12]
    static, axes = plt.subplots(3, 4, figsize=(17, 11))
    for ax, topic in zip(axes.ravel(), topics):
        sub = c.keywords.loc[c.keywords.topic.eq(topic) & c.keywords.ctfidf_weight.gt(0)].head(40)
        frequencies = {row.keyword: float(row.ctfidf_weight) for row in sub.itertuples()}
        wc = WordCloud(font_path=FONT_PATH, width=640, height=420, background_color="white",
                       colormap="viridis" if topic % 2 else "cividis", prefer_horizontal=0.9,
                       random_state=42, margin=4).generate_from_frequencies(frequencies)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"T{topic} · {c.name(topic, 26)}", fontsize=10)
        ax.axis("off")
    static.suptitle("主题词云图：前 12 大主题的高权重关键词（c-TF-IDF，中英文同图）", fontsize=15, y=0.995)
    static.tight_layout(rect=(0, 0.01, 1, 0.985))
    footnote()
    save_figure(static, "fig03_wordcloud")


def topic_detail_figures(c: Ctx, numbers, previous=None):
    """Complete static topic coverage; cloud frequencies use the model tokenizer."""
    rows = previous or []
    indexed = {row["topic"]: dict(row) for row in rows}
    lexical = []
    for topic in c.ids:
        sub = c.df.loc[c.df.topic.eq(topic)]
        row = indexed.setdefault(topic, {"topic": topic, "label": c.name(topic), "records": len(sub), "files": []})
        if 2 in numbers:
            keywords = c.keywords.loc[c.keywords.topic.eq(topic)].sort_values("rank").head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(10, 8.2))
            ax.barh(keywords.keyword, keywords.ctfidf_weight, color=topic_color(topic), height=0.72)
            ax.set_title(f"T{topic} · {c.name(topic)}\n前 20 关键词 · {len(sub):,} 条记录", fontsize=12)
            ax.set_xlabel("c-TF-IDF 权重")
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(axis="x", color=GRID, alpha=0.5)
            footnote()
            name = f"topics/T{topic:02d}_keywords"
            save_figure(fig, name)
            row["files"] = [path for path in row["files"] if "_keywords." not in path] + GENERATED_FILES[name]
        if 3 in numbers:
            frequency = Counter()
            for text in sub.text:
                frequency.update(tokenize(text))
            ordered = dict(sorted(frequency.items(), key=lambda value: (-value[1], value[0]))[:120])
            if not ordered:
                raise ValueError(f"T{topic} 分词后无有效词项")
            wc = WordCloud(font_path=FONT_PATH, width=1000, height=600, max_words=100,
                           background_color="white", colormap="viridis", random_state=42,
                           prefer_horizontal=0.9, margin=6).generate_from_frequencies(ordered)
            fig, ax = plt.subplots(figsize=(10, 6.8))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"T{topic} · {c.name(topic)}\n真实分词频次词云 · {len(sub):,} 条记录", fontsize=12)
            footnote(text=FOOTNOTE + " 字号由标题+摘要中的词项出现次数决定；使用与模型相同的分词及停用词。")
            name = f"topics/T{topic:02d}_frequency_cloud"
            save_figure(fig, name)
            # Native wordcloud SVG contains selectable vector text rather than an embedded PNG.
            import html
            vector = wc.to_svg(embed_font=False)
            vector = vector.replace('height="600"', 'height="680"', 1).replace('</svg>',
                f'<text x="20" y="630" font-family="Microsoft YaHei" font-size="20">{html.escape(f"T{topic} {c.name(topic)} · 真实分词频次")}</text>'
                '<text x="20" y="662" font-family="Microsoft YaHei" font-size="15">未去重；标题+摘要分词；主题标签为解释性命名</text></svg>')
            (FIG / f"{name}.svg").write_text(vector, encoding="utf-8")
            row["files"] = [path for path in row["files"] if "_frequency_cloud." not in path] + GENERATED_FILES[name]
            lexical.append({"topic": topic, "records": len(sub), "total_tokens": sum(frequency.values()),
                            "distinct_tokens": len(frequency), "frequencies": ordered})
        log("topic_details_done", topic=topic)
    if lexical:
        ops.write_json(DIR / "topic_lexical_frequencies.json", lexical)
    return [indexed[topic] for topic in c.ids]


# ---------------------------------------------------------------------------
# 4. 主题层级图：合并层级结构（平均连接 · 余弦距离）
# ---------------------------------------------------------------------------
def fig04_hierarchy(c: Ctx):
    order = leaves_list(c.linkage).astype(int)
    names = {i: t for i, t in enumerate(c.ids)}
    leaf_labels = [f"T{names[i]} {c.name(names[i], 30)}" for i in order]
    from scipy.cluster.hierarchy import set_link_color_palette
    set_link_color_palette([COLORS[i % len(COLORS)] for i in range(10)])

    static, ax = plt.subplots(figsize=(15, 7.5))
    dendrogram(c.linkage, labels=[f"T{t}" for t in c.ids], leaf_rotation=90,
               leaf_font_size=7.5, ax=ax, above_threshold_color=MUTED)
    ax.set(title="主题层级图：42 个主题的合并层级（平均连接法，主题中心余弦距离）",
           xlabel="层次树叶节点（主题编号，按语义距离排序）", ylabel="主题间余弦距离")
    n_clusters = 8
    cut_height = float((c.linkage[-n_clusters, 2] + c.linkage[-(n_clusters - 1), 2]) / 2)
    actual_clusters = len(set(fcluster(c.linkage, t=cut_height, criterion="distance")))
    if actual_clusters != n_clusters:
        raise ValueError(f"切割线形成 {actual_clusters} 簇，预期 {n_clusters} 簇")
    ax.axhline(cut_height, color="#bd6277", lw=0.9, ls="--")
    ax.annotate(f"{n_clusters} 簇切割线 {cut_height:.3f}", xy=(0.005, cut_height),
                xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points",
                fontsize=8.5, color="#bd6277")
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.5)
    footnote()
    save_figure(static, "fig04_topic_hierarchy")
    set_link_color_palette(None)

    # Interactive dendrogram drawn from the linkage matrix with full labels.
    n = len(c.ids)
    pos = {leaf: i for i, leaf in enumerate(order)}
    spans = {}
    lines_x, lines_y = [], []
    for row in range(c.linkage.shape[0]):
        node = n + row
        left, right, dist, _ = c.linkage[row]
        for child in (int(left), int(right)):
            if child < n:
                spans[child] = (pos[child], pos[child], 0.0)
        l0, l1, ldist = spans[int(left)]
        r0, r1, rdist = spans[int(right)]
        yc_l, yc_r = (l0 + l1) / 2, (r0 + r1) / 2
        spans[node] = (min(l0, r0), max(l1, r1), dist)
        lines_x += [dist, ldist, None, dist, rdist, None, dist, dist, None]
        lines_y += [yc_l, yc_l, None, yc_r, yc_r, None, yc_l, yc_r, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lines_x, y=lines_y, mode="lines",
                             line=dict(color=MUTED, width=1.1), hoverinfo="skip", showlegend=False))
    counts = c.summary.set_index("topic")["count"].to_dict()
    fig.add_trace(go.Scatter(
        x=[0] * n, y=list(range(n)), mode="markers+text",
        marker=dict(size=6 + 16 * np.sqrt(np.array([counts[names[i]] for i in order]) / max(counts.values())),
                    color=[topic_color(names[i]) for i in order]),
        text=[f"T{names[i]}" for i in order], textposition="middle right", textfont=dict(size=8),
        customdata=[[c.display(names[i]), int(counts[names[i]])] for i in order],
        hovertemplate="<b>%{customdata[0]}</b><br>记录数 %{customdata[1]:,.0f}<extra></extra>",
        showlegend=False))
    fig.update_layout(title="主题层级图（交互）：可缩放的合并树，气泡大小 = 主题规模",
                      xaxis_title="主题间余弦距离", yaxis=dict(showticklabels=False, range=[-3, n + 2]),
                      xaxis_range=[-0.02, float(c.linkage[:, 2].max()) * 1.02])
    write_interactive(fig, "fig04_topic_hierarchy", height=900)


# ---------------------------------------------------------------------------
# 5. 文档语义分布图：文献在向量空间中的聚类分布
# ---------------------------------------------------------------------------
def fig05_document_map(c: Ctx):
    top12 = c.ordered[:12]
    counts = c.summary.set_index("topic")["count"].to_dict()

    static, ax = plt.subplots(figsize=(11, 8.5))
    ax.scatter(c.df.umap_1, c.df.umap_2, c="#d3d9de", s=1.5, alpha=0.35, rasterized=True,
               label="其他主题 / 离群", zorder=1)
    for topic in top12:
        sub = c.df.loc[c.df.topic.eq(topic)]
        ax.scatter(sub.umap_1, sub.umap_2, s=3.5, alpha=0.6, color=topic_color(topic),
                   label=f"T{topic}（{counts[topic]:,}）", rasterized=True, zorder=2)
        if len(sub) >= 10:
            try:
                hull = ConvexHull(np.column_stack([sub.umap_1, sub.umap_2]))
                ring = np.append(hull.vertices, hull.vertices[0])
                pts = np.column_stack([sub.umap_1, sub.umap_2])[ring]
                ax.fill(pts[:, 0], pts[:, 1], color=topic_color(topic), alpha=0.06, zorder=0)
                ax.plot(pts[:, 0], pts[:, 1], color=topic_color(topic), lw=0.8, alpha=0.5, zorder=0)
            except Exception:
                pass
    ax.set(title="文档语义分布图：22,549 条记录的 UMAP 投影与前 12 大主题",
           xlabel="UMAP 坐标 1", ylabel="UMAP 坐标 2")
    ax.legend(ncol=3, markerscale=3.4, frameon=False, fontsize=8, loc="upper right")
    footnote()
    save_figure(static, "fig05_document_map")

    biggest = c.ordered
    display = c.df.copy()
    display["plot_topic"] = display.topic.map(
        lambda t: f"T{int(t)} {c.name(t, 24)}" if t in biggest else ("离群文献 / Outliers" if t == -1 else "其他主题"))
    display["title_preview"] = display.title.str.slice(0, 160)
    fig = go.Figure()
    for topic in biggest + [-1]:
        sub = display.loc[display.topic.eq(topic)]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=sub.umap_1, y=sub.umap_2, mode="markers", name=f"T{topic}" if topic != -1 else "离群文献",
            marker=dict(size=3.5, color=topic_color(topic), opacity=0.62),
            text=sub.title_preview, customdata=np.stack(
                [sub.record_id, sub.source, sub.year.astype("Int64").astype(str), sub.topic.astype(str),
                 sub.cluster_membership_strength.round(3)], axis=-1),
            hovertemplate="<b>%{text}</b><br>ID %{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}<br>主题 %{customdata[3]}"
                          " · 成员强度 %{customdata[4]}<extra></extra>"))
    fig.update_layout(title="文档语义分布图（交互）：五维 UMAP 前两坐标，图上距离仅作探索性参考",
                      xaxis_title="UMAP 坐标 1", yaxis_title="UMAP 坐标 2",
                      legend=dict(itemsizing="constant", font=dict(size=9)))
    plotted_ids = [row[0] for trace in fig.data for row in trace.customdata]
    if len(plotted_ids) != len(c.df) or set(plotted_ids) != set(c.df.record_id):
        raise ValueError("文档语义交互图未完整覆盖原始记录")
    INTERACTIVE_CONTRACTS["fig05_document_map"] = {"documents": len(plotted_ids), "topic_classes": len(fig.data)}
    write_interactive(fig, "fig05_document_map", height=820)


# ---------------------------------------------------------------------------
# 6. 主题-关键词热力图
# ---------------------------------------------------------------------------
def fig06_topic_keyword_heatmap(c: Ctx):
    topics = c.ordered[:15]
    columns: list[str] = []
    per_topic = {}
    for topic in topics:
        sub = c.keywords.loc[c.keywords.topic.eq(topic) & c.keywords.ctfidf_weight.gt(0)].head(5)
        per_topic[topic] = dict(zip(sub.keyword, sub.ctfidf_weight))
        for word in sub.keyword:
            if word not in columns:
                columns.append(word)
    matrix = np.full((len(topics), len(columns)), np.nan)
    for i, topic in enumerate(topics):
        peak = max(per_topic[topic].values())
        for j, word in enumerate(columns):
            value = per_topic[topic].get(word)
            if value is not None:
                matrix[i, j] = value / peak  # normalise per topic for comparable colouring

    static, ax = plt.subplots(figsize=(1.9 + 0.36 * len(columns), 0.42 * len(topics) + 2.2))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(topics)), [c.display(t, 30) for t in topics], fontsize=8)
    ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=7.5)
    for i in range(len(topics)):
        for j in range(len(columns)):
            if not np.isnan(matrix[i, j]) and matrix[i, j] > 0.55:
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if matrix[i, j] > 0.75 else INK)
    static.colorbar(image, ax=ax, label="主题内相对权重（该主题峰值 = 1）", shrink=0.6)
    ax.set_title("主题-关键词热力图：前 15 大主题 × 各自前 5 关键词")
    footnote()
    save_figure(static, "fig06_topic_keyword_heatmap")

    raw = np.full_like(matrix, np.nan)
    for i, topic in enumerate(topics):
        for j, word in enumerate(columns):
            v = per_topic[topic].get(word)
            if v is not None:
                raw[i, j] = v
    fig = go.Figure(go.Heatmap(
        z=matrix, x=columns, y=[c.display(t, 26) for t in topics], colorscale="Blues", zmin=0, zmax=1,
        customdata=raw,
        hovertemplate="%{y}<br>%{x}<br>相对权重 %{z:.2f} · c-TF-IDF %{customdata:.3f}<extra></extra>",
        colorbar=dict(title="主题内相对权重")))
    fig.update_layout(title="主题-关键词热力图（交互）：悬停显示原始 c-TF-IDF 权重",
                      yaxis_autorange="reversed", height=max(600, 22 * len(topics) + 220))
    write_interactive(fig, "fig06_topic_keyword_heatmap")
    all_columns = list(dict.fromkeys(c.keywords.loc[c.keywords["rank"].le(5), "keyword"]))
    all_raw = c.keywords.pivot(index="topic", columns="keyword", values="ctfidf_weight").reindex(index=c.ids, columns=all_columns).fillna(0)
    all_relative = all_raw.div(all_raw.max(axis=1), axis=0)
    all_fig = go.Figure(go.Heatmap(z=all_relative.values, x=all_columns, y=[c.display(t) for t in c.ids],
        customdata=all_raw.values, colorscale="Blues", zmin=0, zmax=1,
        hovertemplate="%{y}<br>%{x}<br>主题内相对权重 %{z:.3f} · c-TF-IDF %{customdata:.4f}<extra></extra>"))
    all_fig.update_layout(title="全部 42 个主题 × 各主题前 5 关键词并集（已有前 20 词表内的权重；未收录项显示 0）", yaxis_autorange="reversed", xaxis_tickangle=-60)
    write_interactive(all_fig, "fig06_all_topic_keyword_heatmap", height=1200)


# ---------------------------------------------------------------------------
# 7. 主题词秩图：关键词权重随名次的衰减
# ---------------------------------------------------------------------------
def fig07_word_rank(c: Ctx):
    static, axes = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    pivot = c.keywords.loc[c.keywords.ctfidf_weight.gt(0)].pivot(index="rank", columns="topic", values="ctfidf_weight")
    for topic in pivot.columns:
        ax.plot(pivot.index, pivot[topic], color=GRID, lw=0.7, alpha=0.75, zorder=1)
    mean_curve = pivot.mean(axis=1)
    ax.plot(mean_curve.index, mean_curve, color=COLORS[0], lw=2.6, label="42 主题平均", zorder=3)
    for topic in c.ordered[:3]:
        curve = pivot[topic].dropna()
        ax.plot(curve.index, curve, lw=1.6, color=topic_color(topic),
                label=f"T{topic} {ellip(c.name(topic), 16)}", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="主题词秩图：c-TF-IDF 权重随词名次的衰减（每主题前 20 词）",
           xlabel="关键词名次（对数刻度）", ylabel="c-TF-IDF 权重（对数刻度）")
    ax.grid(color=GRID, lw=0.5, alpha=0.6, which="both")
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor=GRID,
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    ax = axes[1]
    rows = []
    for topic in c.ids:
        sub = c.keywords.loc[c.keywords.topic.eq(topic) & c.keywords.ctfidf_weight.gt(0)].set_index("rank").ctfidf_weight
        if 1 in sub.index and 10 in sub.index and sub[10] > 0:
            rows.append({"topic": topic, "ratio": sub[1] / sub[10]})
    frame = pd.DataFrame(rows).sort_values("ratio")
    ax.barh([f"T{t}" for t in frame.topic], frame.ratio, color=[topic_color(t) for t in frame.topic], height=0.7)
    ax.axvline(frame.ratio.mean(), color=INK, lw=1, ls="--")
    ax.annotate(f"均值 {frame.ratio.mean():.1f}", xy=(frame.ratio.mean(), len(frame) - 0.5),
                fontsize=8, color=INK, xytext=(4, 0), textcoords="offset points", va="center")
    ax.set(title="首位 / 第 10 位权重比：主题关键词集中度", xlabel="第 1 名权重 ÷ 第 10 名权重")
    footnote()
    save_figure(static, "fig07_word_rank")

    fig = go.Figure()
    for topic in c.ids:
        curve = pivot[topic].dropna()
        major = topic in c.ordered[:3]
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve, mode="lines", name=f"T{topic}",
            line=dict(color=topic_color(topic) if major else GRID, width=2.4 if major else 0.8),
            opacity=1 if major else 0.7, visible=True if major else "legendonly",
            hovertemplate=f"<b>T{topic} {c.name(topic, 24)}</b><br>名次 %{{x}} · 权重 %{{y:.3f}}<extra></extra>"))
    fig.update_layout(title="主题词秩图（交互）：点击图例查看单个主题的权重衰减曲线",
                      xaxis_title="关键词名次（对数）", yaxis_title="c-TF-IDF 权重（对数）",
                      xaxis_type="log", yaxis_type="log")
    write_interactive(fig, "fig07_word_rank", height=640)


# ---------------------------------------------------------------------------
# 8. 主题时间趋势图：主题占比随年份变化
# ---------------------------------------------------------------------------
def fig08_topic_trends(c: Ctx):
    top12 = c.ordered[:12]
    eligible = c.df.loc[c.df.year.notna() & c.df.year.between(1985, 2026)]
    counts = eligible.groupby(["year", "topic"]).size().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)

    static, axes = plt.subplots(2, 1, figsize=(13, 10.2), sharex=True,
                                gridspec_kw={"height_ratios": [1.15, 1]})
    ax = axes[0]
    for topic in top12:
        ax.plot(shares.index, 100 * shares.get(topic, pd.Series(0, index=shares.index)),
                lw=1.5, color=topic_color(topic), label=f"T{topic} {ellip(c.name(topic), 16)}")
    ax.set(title="主题时间趋势图：前 12 大主题占当年全部记录的比例", ylabel="当年记录占比（%）")
    ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax = axes[1]
    ax = axes[1]
    stack_topics = c.ordered[:8]
    others = eligible.loc[~eligible.topic.isin(stack_topics) & eligible.topic.ne(-1)].groupby("year").size()
    outlier = eligible.loc[eligible.topic.eq(-1)].groupby("year").size()
    base = pd.Series(0.0, index=shares.index)
    for topic in stack_topics:
        values = 100 * shares.get(topic, pd.Series(0, index=shares.index))
        ax.fill_between(shares.index, base, base + values, alpha=0.85, lw=0,
                        color=topic_color(topic))
        base = base + values
    for name, series, color in [("其他主题", others, NEUTRAL), ("离群文献", outlier, "#e5e9ec")]:
        values = 100 * (series / counts.sum(axis=1)).reindex(shares.index).fillna(0)
        ax.fill_between(shares.index, base, base + values, alpha=0.9, lw=0, color=color, label=name)
        base = base + values
    ax.set(title="主题构成堆叠图：前 8 大主题 + 其他 + 离群（合计 100%）",
           xlabel="发表年份（2026 不完整；2027 及缺失年份已排除）", ylabel="当年记录占比（%）",
           xlim=(1985, 2026))
    handles, labels = [], []
    for axes_ in axes:
        sub_handles, sub_labels = axes_.get_legend_handles_labels()
        handles += sub_handles
        labels += sub_labels
    static.legend(handles, labels, ncol=5, fontsize=8.5, frameon=True, framealpha=0.95,
                  edgecolor=GRID, loc="upper center", bbox_to_anchor=(0.5, -0.055))
    footnote()
    save_figure(static, "fig08_topic_trends")

    fig = go.Figure()
    for topic in c.ordered:
        fig.add_trace(go.Scatter(
            x=shares.index.astype(int), y=shares.get(topic, 0), mode="lines", name=f"T{topic} {c.name(topic, 14)}",
            line=dict(color=topic_color(topic), width=2), visible=True if topic in top12 else "legendonly",
            hovertemplate=f"<b>T{topic} {c.name(topic, 24)}</b><br>%{{x}} 年：%{{y:.2%}}<extra></extra>"))
    fig.update_layout(title="主题时间趋势图（交互）：占当年全部记录比例；点击图例可显示全部 42 主题",
                      xaxis_title="发表年份", yaxis_title="当年记录占比", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.18, font=dict(size=9)))
    write_interactive(fig, "fig08_topic_trends", height=640)


# ---------------------------------------------------------------------------
# 9. 主题分类对比图：数据库 / 期刊 / 国家 / 机构 / 文献类型 / 语言
# ---------------------------------------------------------------------------
def normalized_type(value: str) -> str:
    text = str(value).lower()
    if not text:
        return "未标注"
    if "review" in text:
        return "综述"
    if "proceedings" in text or "conference" in text or "meeting" in text:
        return "会议论文"
    if "article" in text or "期刊" in text or "journal" in text:
        return "期刊论文"
    if "chapter" in text or "book" in text:
        return "书籍章节"
    return "其他类型"


def clc_comparison(c: Ctx):
    """Chinese raw classification codes; never infer missing WoS subject areas."""
    data_manifest = json.loads((OUT / "data_manifest.json").read_text(encoding="utf-8"))
    input_dir = Path(data_manifest["input_directory"])
    cnki = pd.read_html(io.StringIO((input_dir / "CNKI.xls").read_text(encoding="utf-8-sig")))[0]
    cnki.columns = cnki.iloc[0].astype(str)
    cnki = cnki.iloc[1:].fillna("")
    wanfang = pd.read_excel(input_dir / "WanFangdata.xlsx", header=1, dtype=str).fillna("")
    rows = []
    for source, frame, column in (("CNKI", cnki, "CLC-中图分类号"), ("WanFang", wanfang, "中图分类号")):
        expected = c.df.loc[c.df.source.eq(source), "record_id"].tolist()
        if len(frame) != len(expected):
            raise ValueError(f"{source} 中图分类原始行数与模型不一致")
        for record_id, code in zip(expected, frame[column]):
            # Keep the raw alphabetic prefix plus first digit; it is not a predicted label.
            groups = sorted(set(re.findall(r"(?<![A-Z])([A-Z]{1,2}\d)", str(code).upper())))
            rows.extend({"record_id": record_id, "clc_group": group, "clc_raw": str(code)} for group in groups)
    data = pd.DataFrame(rows)
    columns = data.clc_group.value_counts().head(10).index.tolist()
    models = [("联合 bge-m3（中图分类有值的中文记录）", c.df, c.labels),
              ("中文 bge-large-zh-v1.5 补充模型（编号独立）",
               pd.read_json(OUT / "chinese_bge_zh/document_topics.json", dtype={"doi": str, "pmid": str, "wos_id": str}),
               {int(k): v for k, v in json.loads((OUT / "chinese_bge_zh/topic_labels.json").read_text(encoding="utf-8")).items()})]
    models[1][2].update({int(k): v for k, v in json.loads((ROOT / "topic_names.json").read_text(encoding="utf-8")).get("chinese_bge_zh", {}).items()})
    static, axes = plt.subplots(1, 2, figsize=(18, 10))
    counts_export = []
    for index, (title, documents, labels) in enumerate(models):
        merged = data.merge(documents[["record_id", "topic"]], on="record_id", validate="many_to_one")
        topics = documents.loc[documents.topic.ge(0)].topic.value_counts().index.tolist()
        grouped = merged.loc[merged.clc_group.isin(columns)].groupby(["clc_group", "topic"]).size()
        denominator = merged.groupby("clc_group").size()
        matrix = grouped.unstack(level=0, fill_value=0).reindex(index=topics, columns=columns).fillna(0)
        shares = matrix.div(denominator.reindex(columns), axis=1)
        ax = axes[index]
        im = ax.imshow(shares, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
        ax.set_yticks(range(len(topics)), [f"T{topic} {short(labels.get(int(topic), ''), 26)}" for topic in topics], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("原始中图分类号前缀（每条记录在同一前缀计一次）")
        static.colorbar(im, ax=ax, label="该分类前缀组内全部记录占比", shrink=0.7)
        fig = go.Figure(go.Heatmap(z=shares.values, x=columns,
            y=[f"T{topic} {labels.get(int(topic), '')}" for topic in topics], colorscale="Blues", zmin=0, zmax=1,
            hovertemplate="%{y}<br>中图分类 %{x} · 组内占比 %{z:.2%}<extra></extra>", colorbar=dict(tickformat=".0%")))
        fig.update_layout(title=title + "；分母包含离群记录", yaxis_autorange="reversed")
        write_interactive(fig, "fig09_by_clc_joint" if index == 0 else "fig09_by_clc_chinese", height=1000)
        counts_export.extend({"model": "joint_bge_m3" if index == 0 else "chinese_bge_zh",
                              "clc_group": group, "topic": int(topic), "records": int(count),
                              "group_records_including_noise": int(denominator[group])}
                             for (group, topic), count in grouped.items())
    static.suptitle("原始中图分类补充对比：仅 CNKI / 万方，两个模型的主题编号不能直接对应", fontsize=14)
    static.tight_layout(rect=(0, 0.065, 1, 0.96))
    footnote(text=f"未去重；分类覆盖 {data.record_id.nunique():,}/715 条中文记录；分类内占比分母包含离群；多分类记录可进入多个组。分类前缀不是推断学科标签。")
    save_figure(static, "fig09_clc_comparison")
    ops.write_json(DIR / "metadata_clc_group_counts.json", counts_export)
    return {"clc_records": int(data.record_id.nunique()), "clc_source_counts": data.drop_duplicates("record_id").record_id.str.split("_").str[0].value_counts().astype(int).to_dict(),
            "clc_grouping": "Raw CLC prefix plus first digit; multiple groups per record; CNKI/WanFang only"}


def fig09_group_comparison(c: Ctx):
    df = c.df
    meta = c.wos_meta
    df = df.merge(meta, on="record_id", how="left")
    df["type_group"] = df.document_type.map(normalized_type)
    df.loc[df.source.ne("WoS"), ["country", "institution"]] = ""
    memberships = [{"record_id": record_id, "country": country, "institution": institution}
                   for record_id, pairs in c.wos_memberships.items() for country, institution in pairs]
    membership_frame = pd.DataFrame(memberships)
    country_data = membership_frame.loc[membership_frame.country.ne(""), ["record_id", "country"]].drop_duplicates().merge(c.df[["record_id", "topic"]], on="record_id", validate="many_to_one")
    institution_data = membership_frame.loc[membership_frame.institution.ne(""), ["record_id", "institution"]].drop_duplicates().merge(c.df[["record_id", "topic"]], on="record_id", validate="many_to_one")
    top_topics = c.ordered[:15]
    topic_counts = df.topic.value_counts()
    total = len(df)

    def share_frame(column, top_values, min_count=0, topic_selection=None):
        selected_topics = top_topics if topic_selection is None else topic_selection
        frame = country_data if column == "country" else institution_data if column == "institution" else df
        table = frame.loc[frame[column].isin(top_values)].groupby([column, "topic"]).size().rename("n").reset_index()
        base = frame.loc[frame[column].isin(top_values)].groupby(column).size()
        table["share"] = table.n / table[column].map(base)
        table = table.loc[table.topic.isin(selected_topics)]
        pivot = table.pivot(index="topic", columns=column, values="share").reindex(selected_topics).fillna(0)
        return pivot

    journals = [j for j in df.journal.value_counts().head(10).index if j][:8]
    countries = country_data.country.value_counts().head(8).index.tolist()
    institutions = institution_data.institution.value_counts().head(8).index.tolist()
    sources = [s for s in ["WoS", "PubMed", "WanFang", "CNKI"] if s in set(df.source)]
    types = [t for t in ["期刊论文", "综述", "会议论文", "书籍章节", "其他类型"] if t in set(df.type_group)]
    languages = ["en", "zh"]
    language_display = {"en": "英文", "zh": "中文"}
    panels = [
        ("数据库（各库内占比）", share_frame("source", sources), "库内占比"),
        ("国家/地区（WoS 全地址；组内占比）", share_frame("country", countries), "组内占比"),
        ("期刊（前 8 刊，刊内占比）", share_frame("journal", journals), "刊内占比"),
        ("机构（WoS 全地址；机构内占比）", share_frame("institution", institutions), "机构内占比"),
        ("文献类型（类型内占比）", share_frame("type_group", types), "类型内占比"),
        ("语言（语言内占比）", share_frame("language_of_text", languages), "语言内占比"),
    ]

    static, axes = plt.subplots(2, 3, figsize=(20, 11.5))
    for ax, (title, pivot, bar_label) in zip(axes.ravel(), panels):
        image = ax.imshow(100 * pivot.values, aspect="auto", cmap="Blues", vmin=0,
                          vmax=max(12, 100 * pivot.values.max()))
        column_labels = [language_display.get(str(col), str(col)) if pivot.columns.name == "language_of_text"
                         else str(col) for col in pivot.columns]
        ax.set_xticks(range(len(pivot.columns)), [label[:16] for label in column_labels], rotation=38,
                      ha="right", fontsize=7.5)
        ax.set_yticks(range(len(pivot)), [c.display(t, 24) for t in pivot.index], fontsize=7.5)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                if pivot.values[i, j] >= 0.08:
                    ax.text(j, i, f"{100 * pivot.values[i, j]:.0f}", ha="center", va="center",
                            fontsize=6.5, color="white" if 100 * pivot.values[i, j] > 0.5 * max(12, 100 * pivot.values.max()) else INK)
        ax.set_title(title, fontsize=10.5)
        static.colorbar(image, ax=ax, label="占比（%）", shrink=0.75)
    static.suptitle(f"主题分类对比图：前 15 大主题在不同分类维度上的分布（分母为各分组内全部记录）", fontsize=15, y=0.995)
    static.tight_layout(rect=(0, 0.012, 1, 0.982))
    footnote()
    save_figure(static, "fig09_group_comparison")

    filenames = {"source": "fig09_by_source", "country": "fig09_by_country", "journal": "fig09_by_journal",
                 "institution": "fig09_by_institution", "type_group": "fig09_by_type", "language_of_text": "fig09_by_language"}
    column_titles = {"source": "数据库", "country": "国家/地区（WoS 全地址）", "journal": "期刊（前 8 刊）",
                     "institution": "机构（WoS 全地址前 8）", "type_group": "文献类型", "language_of_text": "语言"}
    for (title, pivot, bar_label), ax_col in zip(panels, filenames):
        pivot = share_frame(ax_col, pivot.columns.tolist(), topic_selection=c.ids)
        column_labels = [language_display.get(str(col), str(col)) if ax_col == "language_of_text"
                         else str(col) for col in pivot.columns]
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=[label[:24] for label in column_labels],
            y=[c.display(t, 26) for t in pivot.index], colorscale="Blues", zmin=0,
            hovertemplate="%{y}<br>%{x}<br>" + bar_label + " %{z:.2%}<extra></extra>",
            colorbar=dict(title="占比", tickformat=".0%")))
        fig.update_layout(title=f"主题分类对比（{column_titles[ax_col]}）：{title}",
                          yaxis_autorange="reversed", height=1100, xaxis_tickangle=-30)
        write_interactive(fig, filenames[ax_col])
    address_count = sum(bool(pairs) for pairs in c.wos_memberships.values())
    country_count = int(country_data.record_id.nunique())
    coverage = {"scope": "WoS Addresses only", "counting": "all addresses; one membership per record per group; full counting",
                "wos_records": int(c.df.source.eq("WoS").sum()), "with_addresses": address_count,
                "with_recognized_country": country_count, "with_institution": int(institution_data.record_id.nunique()),
                "address_country_unresolved_records": address_count - country_count,
                "institution_disambiguation": "Whitespace normalization only; names are not entity-resolved"}
    coverage.update(clc_comparison(c))
    return {"metadata_coverage": coverage,
            "top_countries": {str(k): int(v) for k, v in country_data.country.value_counts().head(11).items()},
            "top_institutions": {str(k): int(v) for k, v in institution_data.institution.value_counts().head(11).items()}}


# ---------------------------------------------------------------------------
# 10. 主题相似度热力图（按层次叶序排列）
# ---------------------------------------------------------------------------
def fig10_topic_similarity(c: Ctx):
    order = np.array([c.ids[i] for i in leaves_list(c.linkage).astype(int)])
    index = [c.ids.index(t) for t in order]
    matrix = c.sim[np.ix_(index, index)]

    static, ax = plt.subplots(figsize=(11.5, 9.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), [f"T{t}" for t in order], rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), [f"T{t}" for t in order], fontsize=7)
    static.colorbar(image, ax=ax, label="主题中心余弦相似度", shrink=0.75)
    ax.set_title("主题相似度热力图：42 个主题两两余弦相似度（按层次聚类叶序排列）")
    footnote()
    save_figure(static, "fig10_topic_similarity")

    names = [c.display(t, 34) for t in order]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=[f"T{t}" for t in order], y=[f"T{t}" for t in order], colorscale="Blues", zmin=0, zmax=1,
        customdata=[[f"{names[i]} × {names[j]}" for j in range(len(order))] for i in range(len(order))],
        hovertemplate="%{customdata}<br>余弦相似度 %{z:.3f}<extra></extra>",
        colorbar=dict(title="余弦相似度")))
    fig.update_layout(title="主题相似度热力图（交互）：主题中心向量余弦相似度，行列按层次叶序排列",
                      yaxis_autorange="reversed", height=850)
    write_interactive(fig, "fig10_topic_similarity")


# ---------------------------------------------------------------------------
# 11. 主题关系网络图
# ---------------------------------------------------------------------------
def fig11_topic_network(c: Ctx):
    xy = c.mds
    counts = c.summary.set_index("topic")["count"].to_dict()
    n = len(c.ids)
    pair_sims = c.sim[np.triu_indices(n, k=1)]
    threshold = max(0.55, float(np.quantile(pair_sims, 0.97)))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            value = float(c.sim[i, j])
            if value >= threshold:
                edges.append((i, j, value))

    static, ax = plt.subplots(figsize=(11.5, 9))
    for i, j, value in edges:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=MUTED,
                lw=0.6 + 2.4 * (value - threshold) / max(1e-9, pair_sims.max() - threshold),
                alpha=0.25 + 0.5 * (value - threshold) / max(1e-9, pair_sims.max() - threshold), zorder=1)
    sizes = 30 + 620 * np.sqrt(np.array([counts[t] for t in c.ids]) / max(counts.values()))
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=[topic_color(t) for t in c.ids], alpha=0.85,
               edgecolor="white", linewidth=1.2, zorder=3)
    for i, topic in enumerate(c.ids):
        if counts[topic] >= 150 or topic in c.ordered[:16]:
            ax.annotate(f"T{topic}", xy[i], ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold", zorder=4)
    ax.set(title=f"主题关系网络图：余弦相似度 ≥ {threshold:.2f} 的关联结构（{len(edges)} 条边）",
           xlabel="MDS 维度 1", ylabel="MDS 维度 2")
    ax.grid(color=GRID, lw=0.5, alpha=0.5)
    footnote()
    save_figure(static, "fig11_topic_network")

    kw = c.summary.set_index("topic")["keywords"].to_dict()
    fig = go.Figure()
    for i, j, value in edges:
        fig.add_trace(go.Scatter(
            x=[xy[i, 0], xy[j, 0]], y=[xy[i, 1], xy[j, 1]], mode="lines", hoverinfo="skip",
            line=dict(color=MUTED, width=0.6 + 2.4 * (value - threshold) / max(1e-9, pair_sims.max() - threshold)),
            opacity=0.55, showlegend=False))
    customdata = [[c.display(t), int(counts[t]), kw.get(t, "")[:50]] for t in c.ids]
    fig.add_trace(go.Scatter(
        x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=[f"T{t}" for t in c.ids],
        textposition="middle center", textfont=dict(size=8, color="white"),
        marker=dict(size=np.sqrt(sizes) * 1.6, color=[topic_color(t) for t in c.ids], opacity=0.88,
                    line=dict(width=1, color="white")),
        customdata=customdata,
        hovertemplate="<b>%{customdata[0]}</b><br>记录数 %{customdata[1]:,.0f}"
                      "<br>关键词 %{customdata[2]}<extra></extra>", showlegend=False))
    fig.update_layout(title=f"主题关系网络图（交互）：边 = 主题中心余弦相似度 ≥ {threshold:.2f}，节点大小 = 主题规模",
                      xaxis_title="MDS 维度 1", yaxis_title="MDS 维度 2")
    write_interactive(fig, "fig11_topic_network", height=800)


# ---------------------------------------------------------------------------
# 12. 主题规模分布图（含帕累托曲线与长尾）
# ---------------------------------------------------------------------------
def fig12_topic_size_distribution(c: Ctx):
    frame = c.summary.sort_values("count", ascending=False)
    cumulative = frame["count"].cumsum() / frame["count"].sum()

    static, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar([f"T{t}" for t in frame.topic], frame["count"], color=[topic_color(t) for t in frame.topic], width=0.7)
    ax.set_ylabel("文献记录数（未去重）")
    ax.set_xlabel("主题（按规模降序，含全部 42 个非离群主题）")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7.5)
    pareto = ax.twinx()
    pareto.plot(range(len(frame)), 100 * cumulative, color="#bd6277", lw=1.8, marker="o", ms=2.5)
    pareto.set_ylabel("累计占比（%）", color="#bd6277")
    pareto.tick_params(axis="y", colors="#bd6277")
    pareto.set_ylim(0, 105)
    pareto.spines["right"].set_visible(True)
    for rect, count in zip(bars[:1], frame["count"][:1]):
        ax.annotate(f"最大主题 T{int(frame.topic.iloc[0])}：{int(count):,} 条", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(6, -4), textcoords="offset points", fontsize=8.5, ha="left", va="top")
    ax.set_title("主题规模分布图：各主题文献记录数与帕累托累计曲线")
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
    footnote()
    save_figure(static, "fig12_topic_size_distribution")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"T{t}" for t in frame.topic], y=frame["count"], name="记录数",
                         marker_color=[topic_color(t) for t in frame.topic],
                         customdata=[[str(row.label), float(row.share_all_documents)] for row in frame.itertuples()],
                         hovertemplate="<b>T%{x}</b> %{customdata[0]}<br>记录数 %{y:,}（%{customdata[1]:.2%}）<extra></extra>"))
    fig.add_trace(go.Scatter(x=[f"T{t}" for t in frame.topic], y=100 * cumulative, name="累计占比（%）",
                             yaxis="y2", mode="lines+markers", line=dict(color="#bd6277", width=2),
                             hovertemplate="累计 %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="主题规模分布图（交互）：42 个主题规模与帕累托曲线（另含离群 8,562 条）",
                      yaxis=dict(title="文献记录数"), yaxis2=dict(title="累计占比（%）", overlaying="y",
                                                                   side="right", range=[0, 105], tickfont=dict(color="#bd6277")),
                      xaxis=dict(title="主题", tickangle=-90), barmode="group",
                      legend=dict(orientation="h", y=1.08))
    write_interactive(fig, "fig12_topic_size_distribution", height=680)


# ---------------------------------------------------------------------------
# 13. 噪声主题占比图
# ---------------------------------------------------------------------------
def fig13_outlier_share(c: Ctx):
    df = c.df
    total, outliers = len(df), int(df.topic.eq(-1).sum())
    assigned = total - outliers

    static, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax = axes[0, 0]
    wedges, _, autotexts = ax.pie(
        [assigned, outliers], labels=[f"已归类\n{assigned:,} 条", f"离群（主题 -1）\n{outliers:,} 条"],
        colors=[COLORS[0], "#e5e9ec"], autopct=lambda p: f"{p:.2f}%", startangle=90,
        pctdistance=0.72, wedgeprops=dict(width=0.42, edgecolor="white"), textprops=dict(fontsize=10))
    autotexts[0].set_color("white")
    autotexts[1].set_color(INK)
    ax.set_title("全语料归类 vs 离群")

    ax = axes[0, 1]
    rates = df.groupby("source").topic.apply(lambda s: (s == -1).mean()).sort_values()
    ax.barh(rates.index.astype(str), 100 * rates, color=[COLORS[1]] * len(rates), height=0.6)
    for i, (name, rate) in enumerate(rates.items()):
        ax.text(100 * rate + 0.6, i, f"{100 * rate:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(46, 100 * rates.max() + 6))
    ax.set_title("各数据库的离群率")
    ax.set_xlabel("离群记录占比（%）")

    ax = axes[1, 0]
    eligible = df.loc[df.year.between(1995, 2026)].copy()
    yearly = eligible.groupby("year").agg(n=("topic", "size"), out=("topic", lambda s: (s == -1).sum()))
    yearly["rate"] = 100 * yearly.out / yearly.n
    ax.plot(yearly.index, yearly.rate, color=COLORS[0], lw=1.8, marker="o", ms=3)
    ax.set_title("离群率随年份变化（1995–2026）")
    ax.set_xlabel("发表年份")
    ax.set_ylabel("离群记录占比（%）")
    ax.grid(color=GRID, lw=0.5, alpha=0.6)

    ax = axes[1, 1]
    nearest = df.loc[df.topic.eq(-1), "nearest_topic_cosine"]
    strength = df.loc[df.topic.ne(-1), "nearest_topic_cosine"]
    ax.hist(nearest, bins=40, alpha=0.75, color=COLORS[1], label=f"离群记录 · 最近主题余弦（均值 {nearest.mean():.2f}）", density=True)
    ax.hist(strength, bins=40, alpha=0.75, color=COLORS[0], label=f"已归类 · 最近主题余弦（均值 {strength.mean():.2f}）", density=True)
    ax.set_title("两类记录的最近主题余弦相似度分布")
    ax.set_xlabel("与最近主题中心的余弦相似度（两组相同度量）")
    ax.set_ylabel("密度")
    ax.legend(fontsize=8.5, frameon=False)
    static.suptitle("噪声主题占比图：离群（主题 -1）记录的规模、来源与年份结构", fontsize=14.5, y=0.995)
    static.tight_layout(rect=(0, 0.012, 1, 0.98))
    footnote()
    save_figure(static, "fig13_outlier_share")

    fig = go.Figure(go.Pie(
        labels=[f"已归类主题（{assigned:,} 条）", f"离群主题 -1（{outliers:,} 条）"],
        values=[assigned, outliers], hole=0.55, marker=dict(colors=[COLORS[0], "#e5e9ec"]),
        texttemplate="%{label}<br>%{percent}", hovertemplate="%{label}<br>%{value:,} 条（%{percent}）<extra></extra>"))
    fig.update_layout(title=f"噪声主题占比图（交互）：{outliers / total:.2%} 的记录未归入任何主题（HDBSCAN 保留离群，未强制改写）")
    write_interactive(fig, "fig13_outlier_share", height=560)


# ---------------------------------------------------------------------------
# 14. 主题质量诊断图
# ---------------------------------------------------------------------------
def fig14_quality_diagnostics(c: Ctx):
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    assigned = np.flatnonzero(c.df.topic.to_numpy() >= 0)
    rng = np.random.default_rng(42)
    sample = rng.choice(assigned, min(2500, len(assigned)), replace=False)
    labels = c.df.topic.to_numpy()[sample]
    sil_samples = silhouette_samples(embeddings[sample], labels, metric="cosine")
    per_topic_sil = pd.Series(sil_samples).groupby(labels).mean()
    overall_sil = float(np.mean(sil_samples))

    static, axes = plt.subplots(2, 3, figsize=(21, 16))
    ax = axes[0, 0]
    frame = c.summary.dropna(subset=["npmi_top10"]).sort_values("npmi_top10")
    ax.barh([f"T{t}" for t in frame.topic], frame.npmi_top10, color=[topic_color(t) for t in frame.topic], height=0.7)
    ax.axvline(frame.npmi_top10.mean(), color=INK, lw=1, ls="--")
    ax.set_title("各主题 NPMI 一致性（前十关键词）")
    ax.set_xlabel(f"NPMI（文档级共现） · 主题等权均值 {frame.npmi_top10.mean():.3f}")
    plt.setp(ax.get_yticklabels(), fontsize=9.5)

    ax = axes[0, 1]
    sil = per_topic_sil.reindex(c.ordered).dropna()
    colors = [topic_color(t) for t in sil.index]
    ax.barh([f"T{t}" for t in sil.index], sil.values, color=colors, height=0.7)
    ax.axvline(overall_sil, color=INK, lw=1, ls="--")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_title(f"各主题轮廓系数（余弦，抽样 {len(sample):,} 条）")
    ax.set_xlabel(f"主题样本平均轮廓系数 · 全部样本均值 {overall_sil:.3f}")
    plt.setp(ax.get_yticklabels(), fontsize=9.5)

    ax = axes[0, 2]
    cand = c.candidates.sort_values("min_cluster_size")
    x = np.arange(len(cand))
    ax.bar(x - 0.2, cand.topics, width=0.4, color=COLORS[0], label="主题数")
    ax.bar(x + 0.2, cand.outlier_rate * 100, width=0.4, color=COLORS[1], label="离群率（%）")
    ax2 = ax.twinx()
    ax2.plot(x, cand.selection_score, color="#bd6277", lw=1.8, marker="D", ms=5, label="选择分数")
    ax2.set_ylabel("启发式选择分数", color="#bd6277")
    ax2.tick_params(axis="y", colors="#bd6277")
    ax2.set_ylim(0.5, 0.8)
    ax.set_xticks(x, [f"最小簇规模 {int(v)}" for v in cand.min_cluster_size], fontsize=8.5)
    ax.set_ylabel("主题数 / 离群率（%）")
    ax.set_title("候选参数比较（最终选用 100）")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="lower right")

    ax = axes[1, 0]
    st = c.stability
    names = ["ARI\n全部记录", "NMI\n全部记录", "ARI\n两种子均归类"]
    values = [st["ari_all_including_noise"], st["nmi_all_including_noise"], st["ari_both_assigned"]]
    bars = ax.bar(names, values, color=[COLORS[0], COLORS[2], COLORS[4]], width=0.55)
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"随机种子稳定性（种子 42 vs 7；{st['both_assigned_documents']:,} 条共同归类）")
    ax.set_ylabel("一致性（0–1，置换不变）")

    ax = axes[1, 1]
    ax.scatter(c.summary.mean_centroid_cosine, c.summary.mean_membership_strength,
               s=18 + 180 * c.summary["count"] / c.summary["count"].max(),
               c=[topic_color(t) for t in c.summary.topic], alpha=0.8, edgecolor="white")
    for _, row in c.summary.nlargest(8, "count").iterrows():
        ax.annotate(f"T{int(row.topic)}", (row.mean_centroid_cosine, row.mean_membership_strength),
                    fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    ax.set_title("主题内聚力：成员强度 × 主题中心余弦")
    ax.set_xlabel("主题内平均中心余弦相似度")
    ax.set_ylabel("平均 HDBSCAN 成员强度")
    ax.grid(color=GRID, lw=0.5, alpha=0.6)

    ax = axes[1, 2]
    top10 = c.keywords.loc[c.keywords["rank"].le(10)]
    appearances = top10.groupby("keyword").topic.nunique()
    diversity = top10.assign(exclusive=top10.keyword.map(appearances).eq(1)).groupby("topic").exclusive.mean()
    overall = c.metrics["topic_diversity_top10"]
    div = diversity.reindex(c.ordered)
    ax.barh([f"T{t}" for t in div.index], div.values, color=[topic_color(t) for t in div.index], height=0.7)
    ax.axvline(div.mean(), color=INK, lw=1, ls="--")
    ax.set_xlim(0, 1.02)
    ax.set_title(f"各主题独享关键词占比（全局多样性 {overall:.3f}）")
    ax.set_xlabel(f"前十词仅出现在该主题的比例 · 主题等权均值 {div.mean():.3f}")
    plt.setp(ax.get_yticklabels(), fontsize=9.5)

    static.suptitle("主题质量诊断图：一致性、轮廓系数、参数敏感性、种子稳定性、内聚力与关键词多样性", fontsize=15, y=0.995)
    static.tight_layout(rect=(0, 0.012, 1, 0.98))
    footnote(text=FOOTNOTE + " 轮廓系数为原始嵌入空间余弦距离下 2,500 条抽样记录的估计，主题级均值为样本均值。")
    save_figure(static, "fig14_quality_diagnostics")

    fig = go.Figure(go.Bar(
        x=[f"T{t}" for t in frame.topic], y=frame.npmi_top10, marker_color=[topic_color(t) for t in frame.topic],
        customdata=[c.name(t, 30) for t in frame.topic],
        hovertemplate="<b>%{x}</b> %{customdata}<br>NPMI %{y:.3f}<extra></extra>"))
    fig.update_layout(title="主题质量诊断（交互）：各主题 NPMI 一致性排序", yaxis_title="NPMI（前十关键词）",
                      xaxis_title="主题（按 NPMI 升序）")
    write_interactive(fig, "fig14_quality_npmi", height=640)
    return {"silhouette_sample": float(overall_sil), "silhouette_sample_size": len(sample),
            "silhouette_topic_sample_counts": {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
            "keyword_exclusivity_mean": float(div.mean()), "stability_scope": "Two seeds (42,7); not a confidence interval"}


# ---------------------------------------------------------------------------
# 15. 主题演化桑基图（时间段 → 主题；数据库 → 主题）
# ---------------------------------------------------------------------------
def sankey_flows(df: pd.DataFrame, source_key: str, top_n=10):
    counts = df.groupby([source_key, "topic"]).size().rename("n").reset_index()
    top_topics = df.topic.value_counts()
    top_topics = [t for t in top_topics.index if t != -1][:top_n]
    counts["target"] = counts.topic.map(
        lambda t: t if (t in top_topics or t == -1) else "其他")
    grouped = counts.groupby([source_key, "target"]).n.sum().reset_index()
    left_order = list(dict.fromkeys(counts[source_key]))
    right_order = top_topics + ["其他", -1]
    return grouped, left_order, right_order


def draw_sankey_static(ax, grouped, left_order, right_order, title, c: Ctx):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    total = grouped.n.sum()
    left_totals = grouped.groupby(grouped.columns[0]).n.sum().reindex(left_order)
    right_totals = grouped.groupby("target").n.sum().reindex(right_order)
    span = 1.0  # vertical layout is expressed in axes-data units within [0, 1]
    gap = 0.025 * span
    used_l = span - gap * (len(left_order) - 1)
    used_r = span - gap * (len(right_order) - 1)
    left_pos, y = {}, 1.0
    for key in left_order:
        height = used_l * left_totals[key] / total
        left_pos[key] = (y - height, y)
        y -= height + gap
    right_pos, y = {}, 1.0
    for key in right_order:
        height = used_r * right_totals[key] / total
        right_pos[key] = (y - height, y)
        y -= height + gap
    offsets_l = {k: v[1] for k, v in left_pos.items()}
    offsets_r = {k: v[1] for k, v in right_pos.items()}
    scale_l, scale_r = used_l / total, used_r / total
    for row in grouped.itertuples():
        lk, rk, value = row[1], row[2], row.n
        a0, a1 = offsets_l[lk] - value * scale_l, offsets_l[lk]
        b0, b1 = offsets_r[rk] - value * scale_r, offsets_r[rk]
        offsets_l[lk], offsets_r[rk] = a0, b0
        color = topic_color(rk) if isinstance(rk, (int, np.integer)) else NEUTRAL
        verts = [(0.12, a1), (0.5, a1), (0.5, b1), (0.88, b1),
                 (0.88, b0), (0.5, b0), (0.5, a0), (0.12, a0), (0.12, a1)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, alpha=0.5,
                               lw=0.3, edgecolor="white", joinstyle="round"))
    for key, (y0, y1) in left_pos.items():
        ax.text(-0.015, (y0 + y1) / 2, f"{key}　{left_totals[key]:,} 条", ha="right", va="center", fontsize=9)
        ax.add_patch(plt.Rectangle((0.10, y0), 0.02, y1 - y0, color="#9fb0ba", lw=0))
    for key, (y0, y1) in right_pos.items():
        label = c.display(key, 26) if isinstance(key, (int, np.integer)) else (f"离群文献" if key == -1 else "其他主题")
        ax.text(0.905, (y0 + y1) / 2, f"{label}　{right_totals[key]:,} 条", ha="left", va="center", fontsize=9)
        ax.add_patch(plt.Rectangle((0.88, y0), 0.02, y1 - y0,
                                   color=topic_color(key) if isinstance(key, (int, np.integer)) else (NEUTRAL if key == "其他" else "#e5e9ec"), lw=0))
    ax.set_xlim(-0.42, 1.45)
    ax.set_ylim(-0.03, 1.04)
    ax.axis("off")
    ax.set_title(title, fontsize=13)


def fig15_topic_sankey(c: Ctx):
    df = c.df.copy()
    df = df.loc[df.year.notna() & df.year.le(2026)]
    df["period"] = df.year.map(lambda y: "≤2015" if y <= 2015 else ("2016–2020" if y <= 2020 else "2021–2026"))

    grouped, left_order, right_order = sankey_flows(df, "period")
    left_order = [period for period in ("≤2015", "2016–2020", "2021–2026") if period in left_order]
    static, ax = plt.subplots(figsize=(13, 9))
    draw_sankey_static(ax, grouped, left_order, right_order,
                       "主题演化桑基图：三个时间段 → 主题的记录流向（同一联合主题定义，非逐年独立建模）", c)
    footnote()
    save_figure(static, "fig15_sankey_time")

    grouped_src, left_src, right_src = sankey_flows(c.df, "source")
    static, ax = plt.subplots(figsize=(13, 9))
    draw_sankey_static(ax, grouped_src, left_src, right_src,
                       "主题来源桑基图：四个数据库 → 主题的记录流向（未去重）", c)
    footnote()
    save_figure(static, "fig15_sankey_source")

    def plotly_sankey(grouped, left_order, right_order, title, name):
        palette = [topic_color(t) if isinstance(t, (int, np.integer)) else (NEUTRAL if t == "其他" else "#e5e9ec") for t in right_order]
        node_labels = ([f"{k}（{int(grouped.groupby(grouped.columns[0]).n.sum()[k]):,} 条）" for k in left_order]
                       + [((c.display(t, 30) if isinstance(t, (int, np.integer)) else ("离群文献" if t == -1 else "其他主题"))
                           + f"（{int(grouped.groupby('target').n.sum()[t]):,} 条）") for t in right_order])
        left_index = {k: i for i, k in enumerate(left_order)}
        right_index = {k: len(left_order) + i for i, k in enumerate(right_order)}
        fig = go.Figure(go.Sankey(
            node=dict(pad=14, thickness=16, line=dict(color="white", width=0.5),
                      label=node_labels, color=["#9fb0ba"] * len(left_order) + palette),
            link=dict(source=[left_index[r[1]] for r in grouped.itertuples()],
                      target=[right_index[r[2]] for r in grouped.itertuples()],
                      value=[int(r.n) for r in grouped.itertuples()],
                      color=[topic_color(r[2]) if isinstance(r[2], (int, np.integer)) else "rgba(150,160,168,0.35)" for r in grouped.itertuples()],
                      hovertemplate="%{source.label} → %{target.label}<br>%{value:,} 条记录<extra></extra>")))
        fig.update_layout(title=title)
        write_interactive(fig, name, height=760)

    plotly_sankey(grouped, left_order, right_order,
                  "主题演化桑基图（交互）：时间段 → 主题。记录不会跨主题移动；该图展示各时期研究记录流入哪些主题", "fig15_sankey_time")
    plotly_sankey(grouped_src, left_src, right_src,
                  "主题来源桑基图（交互）：数据库 → 主题的记录流向（未去重，WoS 占 89%）", "fig15_sankey_source")


# ---------------------------------------------------------------------------
# Gallery + manifest
# ---------------------------------------------------------------------------
CATALOG = [
    ("fig01_topic_overview", "1. 主题总览图", "42 个主题的规模与语义关系：气泡面积与主题记录数成正比；二维 MDS 近似表示主题中心余弦距离，颜色为层次簇。"),
    ("fig02_topic_keywords", "2. 主题关键词条形图", "前 16 大主题各自前 8 关键词的 c-TF-IDF 权重；交互版可下拉切换全部 42 个主题。"),
    ("fig03_wordcloud", "3. 主题词云图", "总览显示前 12 大主题的关键词权重词云（字号依据 c-TF-IDF，不是词频）；下方全部主题独立图提供真实分词频次词云。"),
    ("fig04_topic_hierarchy", "4. 主题层级图", "主题合并层级树（平均连接 · 主题中心余弦距离）；交互版支持缩放并显示完整主题标签。"),
    ("fig05_document_map", "5. 文档语义分布图", "22,549 条记录的五维 UMAP 前两坐标投影，前 12 大主题着色并绘制凸包边界；交互版可悬停查看每条文献。"),
    ("fig06_topic_keyword_heatmap", "6. 主题-关键词热力图", "前 15 大主题 × 各自前 5 关键词的权重矩阵（主题内归一化着色，悬停显示原始 c-TF-IDF）。"),
    ("fig07_word_rank", "7. 主题词秩图", "各主题关键词权重随名次的衰减曲线（双对数），以及首位/第 10 位权重比反映的关键词集中度。"),
    ("fig08_topic_trends", "8. 主题时间趋势图", "前 12 大主题占当年全部记录比例的逐年变化（1985–2026），叠加前 8 主题的 100% 堆叠构成图。"),
    ("fig09_group_comparison", "9. 主题分类对比图", "前 15 大主题在数据库、国家/地区（WoS 全地址）、期刊、机构（WoS 全地址）、文献类型、语言六个维度上的分布。国家/机构按记录在各组完整计数，合作记录可属于多个组；机构名称仅做空白规范化。学科领域缺失未推断填充。"),
    ("fig10_topic_similarity", "10. 主题相似度热力图", "42 个主题两两中心余弦相似度矩阵，按层次聚类叶序排列，高相似区块一目了然。"),
    ("fig11_topic_network", "11. 主题关系网络图", "以余弦相似度阈值构建的主题关联网络：节点大小 = 主题规模，边宽 = 相似度，布局与总览图一致。"),
    ("fig12_topic_size_distribution", "12. 主题规模分布图", "全部 42 个主题的记录数降序柱状图 + 帕累托累计曲线，展示主题规模的长尾结构。"),
    ("fig13_outlier_share", "13. 噪声主题占比图", "离群（主题 -1）记录的总体占比环形图、分库离群率、逐年离群率与最近主题相似度分布。"),
    ("fig14_quality_diagnostics", "14. 主题质量诊断图", "模型质量六联图：各主题 NPMI、轮廓系数、候选参数比较、随机种子稳定性、主题内聚力与关键词多样性。"),
    ("fig15_sankey_time", "15. 主题演化桑基图", "时间段 → 主题的记录流向桑基图（同一联合主题定义）；配套数据库 → 主题来源桑基图。"),
]

EXTRA_INTERACTIVE = {
    "fig06_topic_keyword_heatmap": [("fig06_all_topic_keyword_heatmap", "全部 42 主题关键词矩阵")],
    "fig09_group_comparison": [("fig09_by_source", "按数据库"), ("fig09_by_country", "按国家（WoS）"),
                               ("fig09_by_journal", "按期刊"), ("fig09_by_institution", "按机构（WoS）"),
                               ("fig09_by_type", "按文献类型"), ("fig09_by_language", "按语言"),
                               ("fig09_by_clc_joint", "中图分类（联合）"), ("fig09_by_clc_chinese", "中图分类（中文补充）"),
                               ("fig09_clc_comparison", "原始中图分类补充图")],
    "fig14_quality_diagnostics": [("fig14_quality_npmi", "各主题 NPMI")],
    "fig15_sankey_time": [("fig15_sankey_source", "数据库 → 主题桑基图")],
}


FUNCTIONS = {
    1: fig01_topic_overview, 2: fig02_topic_keywords, 3: fig03_wordcloud, 4: fig04_hierarchy,
    5: fig05_document_map, 6: fig06_topic_keyword_heatmap, 7: fig07_word_rank, 8: fig08_topic_trends,
    9: fig09_group_comparison, 10: fig10_topic_similarity, 11: fig11_topic_network,
    12: fig12_topic_size_distribution, 13: fig13_outlier_share, 14: fig14_quality_diagnostics,
    15: fig15_topic_sankey,
}


def main():
    global DIR, FIG, VIZ, FOOTNOTE
    parser = argparse.ArgumentParser(description="生成完整且可校验的离线 BERTopic 图库（无需 API）")
    parser.add_argument("--only", type=str, default="", help="逗号分隔的图编号，例如 1,4,9；默认全部")
    parser.add_argument("--check-only", action="store_true", help="只校验当前发布的图库与数据指纹")
    parser.add_argument("--recover-lock", action="store_true", help="仅在原进程已退出时恢复遗留锁")
    args = parser.parse_args()
    try:
        requested = ops.parse_selection(args.only)
    except ValueError as error:
        parser.error(str(error))
    destination = DIR
    start = time.time()
    with ops.BuildLock(destination, recover=args.recover_lock):
        fingerprint, inputs = ops.input_fingerprint(ROOT, destination)
        code_hashes = {name: ops.sha256(ROOT / name) for name in ("make_figures.py", "figure_ops.py")}
        previous_path = destination / "figures_manifest.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
        if args.check_only:
            if previous.get("schema_version") != ops.SCHEMA_VERSION:
                raise ValueError("旧版图库清单，请先完整运行一次")
            if previous["data_fingerprint"] != fingerprint or previous["code_sha256"] != code_hashes:
                raise ValueError("分析数据或绘图代码已变化，请完整重绘")
            result = ops.validate_release(destination, previous)
            log("figures_verified", **result)
            return
        ctx = Ctx()
        checks = ops.preflight(ctx, ROOT)
        if FONT_PATH is None:
            raise ValueError("未找到中文字体；请安装 Microsoft YaHei 或 SimHei")
        FOOTNOTE = f"口径：四库合并未去重记录（N={len(ctx.df):,}）；主题 -1 为保留离群类；2026 年不完整；主题标签为解释性命名。"
        full = set(requested) == set(FUNCTIONS)
        if not full:
            if previous.get("schema_version") != ops.SCHEMA_VERSION:
                raise ValueError("首次升级必须完整重绘，之后可使用 --only")
            if previous["data_fingerprint"] != fingerprint or previous["code_sha256"] != code_hashes:
                raise ValueError("数据或代码已变更；单图重绘会混用版本，请完整重绘")
            ops.validate_release(destination, previous)
        stage = Path(tempfile.mkdtemp(prefix=".figure_stage_", dir=destination))
        try:
            if not full:
                for item in previous["artifacts"]:
                    target = stage / item["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination / item["path"], target)
                INTERACTIVE_CONTRACTS.update(previous["interactive_contracts"])
            DIR, FIG, VIZ = stage, stage / "figures", stage / "visualizations"
            FIG.mkdir(parents=True, exist_ok=True)
            VIZ.mkdir(parents=True, exist_ok=True)
            extras = {} if full else dict(previous["extras"])
            log("figures_start", topics=len(ctx.ids), records=len(ctx.df), requested=requested)
            for number in requested:
                t0 = time.time()
                result = FUNCTIONS[number](ctx)
                if isinstance(result, dict):
                    extras.update(result)
                log("figure_done", number=number, seconds=round(time.time() - t0, 1))
            details = topic_detail_figures(ctx, requested, None if full else previous["topic_details"])
            artifacts = [ops.artifact_info(stage, path) for path in sorted(stage.rglob("*"))
                         if path.is_file() and path.name not in ("figure_gallery.html", "figures_validation.json")]
            charts = []
            for name, title, _ in CATALOG:
                number_prefix = name.split("_", 1)[0] + "_"
                paths = [item["path"] for item in artifacts if Path(item["path"]).name.startswith(number_prefix)]
                charts.append({"id": name, "title": title, "files": paths})
            eligible = ctx.df.year.between(1985, 2026)
            method_notes = [FOOTNOTE,
                "词云总览使用 c-TF-IDF 权重；全部主题独立词云使用与模型相同分词规则的真实词项出现次数。",
                "MDS 为高维距离的近似二维投影；UMAP 图取原有五维结果的前两坐标；图上距离与面积不代表显著性。",
                "图 9 国家/地区与机构仅来自 WoS 全部 Addresses；同一记录在每个组内计一次，合作记录可属于多个组。机构名称未做实体消歧。",
                "WoS 学科字段为空；文献类型不是学科领域。CNKI/万方原始中图分类另有补充图，仅覆盖分类有值的中文子集，两个模型主题编号独立。",
                f"逐年趋势仅展示 1985–2026；纳入 {int(eligible.sum()):,} 条，排除 {int((~eligible).sum()):,} 条（缺失/范围外）。2026 年不完整。",
                "图 15 展示各时期或数据库的主题构成；没有纵向跟踪同一文献，不能解释为主题迁移或因果关系。",
                "轮廓系数在原始嵌入余弦空间抽样计算；NPMI 来自文档级共现；仅两次随机种子比较，不能视为置信区间。",
                "主模型中中文记录集中于一个混合语言主题；中文细分结论应结合原报告中的中文补充模型。",
                "质量诊断图用于揭示局限；图库的工程校验通过不等于主题结论获得统计学验证。"]
            manifest = {"schema_version": ops.SCHEMA_VERSION, "model": "joint_bge_m3", **checks,
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - start, 1),
                        "requested_figures": requested, "data_fingerprint": fingerprint, "input_sha256": inputs,
                        "code_sha256": code_hashes, "runtime": ops.runtime_versions(), "random_seed": 42,
                        "render": {"png_dpi": 300, "formats": ["png", "svg", "pdf"], "offline_interactive": True},
                        "charts": charts, "topic_details": details, "artifacts": artifacts, "extras": extras,
                        "interactive_contracts": INTERACTIVE_CONTRACTS, "method_notes": method_notes}
            if ctx._mds is not None:
                extras["mds_normalized_stress"] = ctx.mds_stress
            result = ops.validate_release(stage, manifest)
            ops.make_gallery(stage, manifest, CATALOG, EXTRA_INTERACTIVE)
            ops.write_json(stage / "figures_validation.json", {**result, "files": len(artifacts) + 2,
                                                               "preflight": checks, "data_fingerprint": fingerprint,
                                                               "publication": "validated before commit"})
            artifacts.extend(ops.artifact_info(stage, stage / name) for name in ("figure_gallery.html", "figures_validation.json"))
            result = ops.validate_release(stage, manifest)
            final_fingerprint, _ = ops.input_fingerprint(ROOT, destination)
            final_code = {name: ops.sha256(ROOT / name) for name in code_hashes}
            if final_fingerprint != fingerprint or final_code != code_hashes:
                raise ValueError("渲染期间分析数据或绘图代码改变；拒绝发布混合版本")
            ops.write_json(stage / "figures_manifest.json", manifest)
            files = [item["path"] for item in artifacts] + ["figures_manifest.json"]
            ops.publish(stage, destination, files)
            log("figures_complete", seconds=round(time.time() - start, 1),
                gallery=str(destination / "figure_gallery.html"), **result)
            ops.write_json(destination / "figure_build_status.json", {"status": "success", "at": manifest["generated_at"],
                                                                      "requested": requested, "files": len(artifacts)})
        except BaseException as error:
            ops.write_json(destination / "figure_build_status.json", {"status": "failed", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                                      "error": f"{type(error).__name__}: {error}", "requested": requested})
            raise
        finally:
            DIR, FIG, VIZ = destination, destination / "figures", destination / "visualizations"
            plt.close("all")
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
