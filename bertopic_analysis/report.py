"""Produce an offline Chinese report, interactive charts and exportable figures."""
from __future__ import annotations
import html
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import normalized_mutual_info_score

from prepare_and_embed import ROOT, OUT, write_json
from analyze import save_table

COLORS = ["#176b87", "#d17a45", "#599977", "#9078a8", "#bd6277", "#8c8a48", "#4e93ac", "#725847", "#746ec9", "#c4944e", "#3f8d8c", "#ab6d43"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False, "svg.fonttype": "none"})
for font_path in [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")]:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
        break


def read_table(directory, name):
    return pd.read_json(directory / f"{name}.json", dtype={"doi": str, "pmid": str, "wos_id": str})


def labels_for(directory, summary):
    labels = json.loads((directory / "topic_labels.json").read_text(encoding="utf-8"))
    manual = ROOT / "topic_names.json"
    if manual.exists():
        replacement = json.loads(manual.read_text(encoding="utf-8")).get(directory.name, {})
        labels.update({str(key): value for key, value in replacement.items()})
    return {int(key): value for key, value in labels.items()}


def write_interactive(figure, destination):
    figure.update_layout(template="plotly_white", font=dict(family="Microsoft YaHei, Arial", size=12), margin=dict(l=50, r=30, t=65, b=55), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    figure.write_html(destination, include_plotlyjs="directory", full_html=True, config={"responsive": True, "displaylogo": False, "toImageButtonOptions": {"format": "svg"}})


def save_figure(figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(directory / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def trend_table(assignments, labels):
    earlier = assignments.loc[assignments.year.between(2016, 2020)]
    recent = assignments.loc[assignments.year.between(2021, 2025)]
    rows = []
    for topic in sorted(set(assignments.topic) - {-1}):
        a = int(earlier.topic.eq(topic).sum())
        b = int(recent.topic.eq(topic).sum())
        share_a = a / max(1, len(earlier))
        share_b = b / max(1, len(recent))
        rows.append({"topic": int(topic), "label": labels[int(topic)], "records_2016_2020": a, "records_2021_2025": b, "share_2016_2020": share_a, "share_2021_2025": share_b, "change_percentage_points": 100 * (share_b - share_a), "share_ratio": share_b / share_a if share_a else None})
    return pd.DataFrame(rows).sort_values("change_percentage_points", ascending=False)


def charts(directory, assignments, summary, labels):
    model_title = {"joint_bge_m3": "四库联合 bge-m3", "chinese_bge_zh": "中文 bge-large-zh-v1.5", "chinese_bge_m3_comparison": "中文 bge-m3"}.get(directory.name, directory.name)
    interactive = directory / "visualizations"
    figures = directory / "figures"
    interactive.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    summary = summary.copy()
    summary["label"] = summary.topic.map(labels)
    summary["display_label"] = summary.apply(lambda row: f"T{int(row.topic)} · {row.label}", axis=1)
    top = summary.head(20).iloc[::-1]
    fig = px.bar(top, x="count", y="display_label", orientation="h", title="主要主题规模（保留重复记录）", labels={"count": "文献记录数", "display_label": "主题"}, color_discrete_sequence=[COLORS[0]])
    fig.update_layout(height=780)
    write_interactive(fig, interactive / "topic_sizes.html")
    static, ax = plt.subplots(figsize=(10, 7))
    short_labels = [f"T{int(row.topic)} {str(row.label)[:34]}" for row in top.itertuples()]
    ax.barh(short_labels, top["count"], color=COLORS[0])
    ax.set_xlabel("文献记录数（未去重）")
    ax.set_title(f"{model_title}：主要主题规模")
    save_figure(static, figures, "topic_sizes")

    largest = summary.head(12).topic.tolist()
    eligible = assignments.loc[assignments.year.notna() & assignments.year.le(2026)]
    counts = eligible.groupby(["year", "topic"]).size().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    timeline = go.Figure()
    static, ax = plt.subplots(figsize=(11, 5.5))
    for i, topic in enumerate(largest):
        label = f"T{topic} {labels[topic]}"
        timeline.add_trace(go.Scatter(x=shares.index.astype(int), y=shares.get(topic, 0), mode="lines", name=label, line=dict(color=COLORS[i % len(COLORS)])))
        ax.plot(shares.index, 100 * shares[topic], label=f"T{topic}", color=COLORS[i % len(COLORS)], lw=1.4)
    timeline.update_layout(title="主要主题的年度占比（2026 为不完整年份）", xaxis_title="发表年份", yaxis_title="占当年全部记录比例", yaxis_tickformat=".0%", height=620, legend=dict(orientation="h", y=-0.2))
    write_interactive(timeline, interactive / "topics_over_time.html")
    ax.set_xlim(max(1960, int(shares.index.min())), 2026)
    ax.set_xlabel("发表年份")
    ax.set_ylabel("当年记录占比（%）")
    ax.set_title(f"{model_title}：年度主题占比")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    save_figure(static, figures, "topics_over_time")

    source = read_table(directory, "topics_by_source")
    source_heat = source.loc[source.topic.isin(summary.head(20).topic)].pivot(index="topic", columns="source", values="share_within_source").fillna(0).reindex(summary.head(20).topic)
    fig = go.Figure(go.Heatmap(z=source_heat.values, x=source_heat.columns, y=[f"T{topic} {labels[topic]}" for topic in source_heat.index], colorscale="Blues", colorbar=dict(title="库内占比", tickformat=".0%"), hovertemplate="%{y}<br>%{x}<br>库内占比 %{z:.2%}<extra></extra>"))
    fig.update_layout(title="各数据库中的主题占比（分别以各库全部记录为分母）", height=850, yaxis_autorange="reversed")
    write_interactive(fig, interactive / "topics_by_source.html")
    static, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(100 * source_heat.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(source_heat.columns)), source_heat.columns)
    ax.set_yticks(range(len(source_heat)), [f"T{topic} {labels[topic][:28]}" for topic in source_heat.index])
    ax.set_title("各库主题占比（%）")
    static.colorbar(image, ax=ax, label="库内记录占比（%）", shrink=0.7)
    save_figure(static, figures, "topics_by_source")

    display = assignments.copy()
    display["plot_topic"] = display.topic.map(lambda topic: f"T{int(topic)} {labels[int(topic)]}" if topic in largest else ("离群文献" if topic == -1 else "其他主题"))
    display["title_preview"] = display.title.str.slice(0, 180)
    fig = px.scatter(display, x="umap_1", y="umap_2", color="plot_topic", hover_name="title_preview", hover_data={"record_id": True, "source": True, "year": True, "topic": True, "cluster_membership_strength": ":.3f", "umap_1": False, "umap_2": False, "plot_topic": False}, render_mode="webgl", opacity=0.65, title="全部记录的语义分布（五维 UMAP 的前两个坐标）", labels={"umap_1": "UMAP 坐标 1", "umap_2": "UMAP 坐标 2", "plot_topic": "主题"}, color_discrete_sequence=COLORS + ["#b6bcc3", "#dddddd"])
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=800)
    write_interactive(fig, interactive / "document_map.html")
    static, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(assignments.umap_1, assignments.umap_2, c="#cccccc", s=2, alpha=0.3, rasterized=True)
    for i, topic in enumerate(largest):
        subset = assignments.loc[assignments.topic.eq(topic)]
        ax.scatter(subset.umap_1, subset.umap_2, s=3, alpha=0.55, color=COLORS[i % len(COLORS)], label=f"T{topic}", rasterized=True)
    ax.set(xlabel="UMAP 坐标 1", ylabel="UMAP 坐标 2", title="全部记录的语义分布")
    ax.legend(ncol=4, markerscale=3, frameon=False)
    save_figure(static, figures, "document_map")
    return interactive, figures


def extra_charts(directory, summary, labels):
    interactive = directory / "visualizations"
    figures = directory / "figures"
    ids = json.loads((directory / "topic_ids.json").read_text(encoding="utf-8"))
    cosine = np.load(directory / "topic_similarity.npy")
    selected = summary.head(35).topic.astype(int).tolist()
    indices = [ids.index(topic) for topic in selected]
    fig = go.Figure(go.Heatmap(z=cosine[np.ix_(indices, indices)], x=[f"T{topic}" for topic in selected], y=[f"T{topic}" for topic in selected], colorscale="Blues", zmin=0, zmax=1, hovertemplate="%{x} / %{y}<br>主题向量余弦相似度 %{z:.3f}<extra></extra>"))
    fig.update_layout(title="主要主题的语义相似度（主题中心向量）", height=780)
    write_interactive(fig, interactive / "topic_similarity.html")
    if (directory / "hierarchy_linkage.npy").exists():
        tree = np.load(directory / "hierarchy_linkage.npy")
        static, ax = plt.subplots(figsize=(12, 6))
        dendrogram(tree, truncate_mode="lastp", p=min(45, len(ids)), show_leaf_counts=True, ax=ax)
        ax.set(title="主题层次关系：平均连接法、余弦距离", xlabel="主题树叶节点（括号表示合并节点数）", ylabel="余弦距离")
        save_figure(static, figures, "topic_hierarchy")
    keywords = read_table(directory, "topic_keywords")
    static, axes = plt.subplots(2, 4, figsize=(15, 8))
    for ax, topic in zip(axes.ravel(), summary.head(8).topic):
        subset = keywords.loc[keywords.topic.eq(topic)].head(8).iloc[::-1]
        ax.barh(subset.keyword, subset.ctfidf_weight, color=COLORS[int(topic) % len(COLORS)])
        ax.set_title(f"T{int(topic)}", fontsize=12)
        ax.tick_params(axis="y", labelsize=8)
    static.suptitle("主要主题关键词及 c-TF-IDF 权重")
    static.tight_layout()
    save_figure(static, figures, "topic_keywords")


def main():
    directory = OUT / "joint_bge_m3"
    assignments = read_table(directory, "document_topics")
    summary = read_table(directory, "topic_summary")
    metrics = json.loads((directory / "selected_model_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "data_manifest.json").read_text(encoding="utf-8"))
    stability = json.loads((directory / "stability.json").read_text(encoding="utf-8"))
    candidates = read_table(directory, "parameter_comparison")
    labels = labels_for(directory, summary)
    summary["label"] = summary.topic.map(labels)
    assignments["topic_label"] = assignments.topic.map(labels)
    source_topics = read_table(directory, "topics_by_source")
    language_topics = read_table(directory, "topics_by_language")
    annual_topics = read_table(directory, "topics_by_year")
    for table in (source_topics, language_topics, annual_topics):
        table["topic_label"] = table.topic.map(labels)
    similarity_table = read_table(directory, "topic_similarity")
    similarity_table["label_a"] = similarity_table.topic_a.map(labels)
    similarity_table["label_b"] = similarity_table.topic_b.map(labels)
    for name, table in {"document_topics": assignments, "topic_summary": summary, "topics_by_source": source_topics, "topics_by_language": language_topics, "topics_by_year": annual_topics}.items():
        save_table(table, directory, name)
    save_table(similarity_table, directory, "topic_similarity")
    write_json(directory / "topic_labels.json", {str(key): value for key, value in labels.items()})
    chinese_distribution = assignments.loc[assignments.language_of_text.eq("zh"), "topic"].value_counts()
    chinese_main_topic = int(chinese_distribution.index[0])
    chinese_main_count = int(chinese_distribution.iloc[0])
    chinese_main_share = chinese_main_count / manifest["chinese_texts"]
    trends = trend_table(assignments, labels)
    save_table(trends, directory, "recent_topic_changes")
    associations = {"topic_source_nmi": float(normalized_mutual_info_score(assignments.topic, assignments.source)), "topic_language_nmi": float(normalized_mutual_info_score(assignments.topic, assignments.language_of_text)), "meaning": "Descriptive association diagnostics; affected by imbalanced source/language frequencies and not a causal effect or correctness test."}
    write_json(directory / "source_language_associations.json", associations)
    # The joint report links to the release generated by make_figures.py.
    # Keep the separate Chinese-model figures for the embedding comparison.
    for name in ["chinese_bge_zh", "chinese_bge_m3_comparison"]:
        chinese_directory = OUT / name
        if (chinese_directory / "topic_summary.json").exists():
            subset_summary = read_table(chinese_directory, "topic_summary")
            subset_assignments = read_table(chinese_directory, "document_topics")
            subset_labels = labels_for(chinese_directory, subset_summary)
            subset_summary["label"] = subset_summary.topic.map(subset_labels)
            subset_assignments["topic_label"] = subset_assignments.topic.map(subset_labels)
            save_table(subset_summary, chinese_directory, "topic_summary")
            save_table(subset_assignments, chinese_directory, "document_topics")
            write_json(chinese_directory / "topic_labels.json", {str(key): value for key, value in subset_labels.items()})
            charts(chinese_directory, subset_assignments, subset_summary, subset_labels)
    representative = read_table(directory, "representative_documents")
    top = summary.head(10)
    growing = trends.loc[trends.records_2021_2025.ge(25)].head(5)
    declining = trends.loc[trends.records_2016_2020.ge(25)].tail(5).iloc[::-1]
    source_frame = pd.DataFrame(manifest["sources"])
    source_frame["share"] = source_frame.records / manifest["rows"]
    comparisons = json.loads((OUT / "chinese_model_comparison.json").read_text(encoding="utf-8")) if (OUT / "chinese_model_comparison.json").exists() else None
    api_m3 = json.loads((OUT / "joint_bge_m3_embedding_metadata.json").read_text(encoding="utf-8"))
    api_zh = json.loads((OUT / "chinese_bge_zh_embedding_metadata.json").read_text(encoding="utf-8"))
    hashes_unchanged = all(hashlib.sha256((Path(manifest["input_directory"]) / name).read_bytes()).hexdigest() == expected for name, expected in manifest["source_sha256"].items())
    if not hashes_unchanged:
        raise RuntimeError("An input file changed after preparation; rerun the pipeline before reporting")
    facts = {
        "records": len(assignments), "topics": metrics["topics"], "outliers": metrics["outliers"], "outlier_rate": metrics["outlier_rate"],
        "topic_names": {str(key): value for key, value in labels.items()},
        "largest_topics": top[["topic", "label", "count", "share_all_documents", "keywords"]].to_dict("records"),
        "recent_increases": growing.to_dict("records"), "recent_decreases": declining.to_dict("records"),
        "selected_model": metrics, "stability": stability, "associations": associations,
        "chinese_language_concentration": {"topic": chinese_main_topic, "count": chinese_main_count, "share_chinese_documents": chinese_main_share},
        "api_tokens": api_m3["api_tokens_this_run"] + api_zh["api_tokens_this_run"],
        "wos_records_retained": int(assignments.source.eq("WoS").sum()), "source_hashes_unchanged": hashes_unchanged,
    }
    write_json(OUT / "report_facts.json", facts)
    e = lambda value: html.escape(str(value))
    fmt = lambda value: f"{value:,.0f}"
    topic_rows = "".join(f'<tr><td>T{int(row.topic)}</td><td>{e(row.label)}</td><td>{fmt(row.count)}</td><td>{row.share_all_documents:.2%}</td><td>{e(row.keywords)}</td></tr>' for row in summary.itertuples())
    source_rows = "".join(f'<tr><td>{e(row.source)}</td><td>{fmt(row.records)}</td><td>{fmt(row.with_abstract)}</td><td>{fmt(row.missing_abstract)}</td><td>{row.share:.2%}</td></tr>' for row in source_frame.itertuples())
    candidate_rows = "".join(f'<tr><td>{row.min_cluster_size}</td><td>{row.topics}</td><td>{row.outlier_rate:.2%}</td><td>{row.semantic_silhouette_sample1500:.4f}</td><td>{row.mean_npmi_top10:.4f}</td><td>{row.topic_diversity_top10:.4f}</td><td>{row.selection_score:.4f}</td></tr>' for row in candidates.itertuples())
    change_rows = "".join(f'<tr><td>T{int(row.topic)} {e(row.label)}</td><td>{row.records_2016_2020}</td><td>{row.records_2021_2025}</td><td>{row.share_2016_2020:.2%}</td><td>{row.share_2021_2025:.2%}</td><td>{row.change_percentage_points:+.2f}</td></tr>' for row in growing.itertuples())
    top_descriptions = "".join(f'<li><b>T{int(row.topic)} {e(row.label)}</b>：{fmt(row.count)} 条记录，占全部输入的 {row.share_all_documents:.2%}。关键词：{e("、".join(row.keywords.split("; ")[:8]))}。</li>' for row in top.head(5).itertuples())
    figures_section = "".join(f'<details {"open" if i < 2 else ""}><summary>{e(title)}</summary><iframe loading="lazy" src="joint_bge_m3/visualizations/{filename}" title="{e(title)}"></iframe></details>' for i, (title, filename) in enumerate([
        ("主题规模", "fig12_topic_size_distribution.html"), ("全部文献语义分布", "fig05_document_map.html"), ("年度主题占比", "fig08_topic_trends.html"), ("各库主题分布", "fig09_by_source.html"), ("主题语义相似度", "fig10_topic_similarity.html")]))
    chinese_section = ""
    if comparisons:
        zh = comparisons["bge_zh"]
        m3 = comparisons["bge_m3"]
        chinese_section = f'''<section id="chinese"><h2>中文子集模型对照</h2><p>联合模型仍是全部四库共同拟合。以下只是在同一组 {comparisons["chinese_documents"]} 条中文文本上分别拟合两种模型，检验嵌入模型选择的影响；两个向量空间没有拼接。</p><table><tr><th>模型</th><th>主题数</th><th>离群率</th><th>语义轮廓系数</th><th>NPMI</th></tr><tr><td>bge-large-zh-v1.5</td><td>{zh["topics"]}</td><td>{zh["outlier_rate"]:.2%}</td><td>{zh["semantic_silhouette_sample1500"]:.4f}</td><td>{zh["mean_npmi_top10"]:.4f}</td></tr><tr><td>bge-m3</td><td>{m3["topics"]}</td><td>{m3["outlier_rate"]:.2%}</td><td>{m3["semantic_silhouette_sample1500"]:.4f}</td><td>{m3["mean_npmi_top10"]:.4f}</td></tr></table><p>两模型分组一致性：ARI={comparisons["ari_all"]:.4f}，NMI={comparisons["nmi_all"]:.4f}。主题编号不直接对应，不能据此把两个模型的同号主题视为相同。</p><p><a href="chinese_bge_zh/visualizations/topic_sizes.html">中文专用模型主题图</a> · <a href="chinese_bge_m3_comparison/visualizations/topic_sizes.html">中文子集多语言模型主题图</a> · <a href="chinese_model_assignments.csv">逐条对照结果</a></p></section>'''
    representative_data = {str(topic): json.loads(representative.loc[representative.topic.eq(topic)].to_json(orient="records", force_ascii=False)) for topic in summary.topic}
    browser_topics = json.loads(summary[["topic", "label", "count", "share_all_documents", "keywords", "npmi_top10"]].to_json(orient="records", force_ascii=False))
    browser_data = json.dumps({"topics": browser_topics, "representatives": representative_data}, ensure_ascii=False).replace("</", "<\\/")
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>四库联合 BERTopic 分析</title><style>
:root{{--ink:#173242;--muted:#61727e;--accent:#176b87;--paper:#f3f6f7;--line:#dbe3e7}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);font:15px/1.75 "Microsoft YaHei",Arial,sans-serif;color:var(--ink)}}header{{background:#153b4d;color:white;padding:44px max(5vw,24px)}}header h1{{margin:0 0 12px;font-size:32px}}header p{{margin:0;color:#d7e6ec}}main{{max-width:1220px;margin:24px auto;padding:0 20px 60px}}nav{{display:flex;flex-wrap:wrap;gap:18px;margin:24px 0}}a{{color:var(--accent)}}nav a{{text-decoration:none}}section{{background:white;padding:28px 32px;margin:22px 0;border:1px solid var(--line);border-radius:9px}}h2{{margin:0 0 16px;font-size:22px}}h3{{font-size:17px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card{{background:#eaf2f5;padding:17px;border-radius:6px}}.card strong{{font-size:29px;display:block}}.card span{{color:var(--muted)}}.note{{background:#fff5df;padding:16px 20px;border-left:4px solid #bd8a36}}.table-scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}}th{{background:#edf3f5;white-space:nowrap}}.topics-table td:last-child{{min-width:320px}}details{{margin:16px 0;border:1px solid var(--line);padding:12px;border-radius:6px}}summary{{font-weight:600;cursor:pointer}}iframe{{width:100%;height:850px;border:0}}input,select{{padding:10px 12px;border:1px solid #adbfc8;border-radius:5px;font:inherit}}.controls{{display:flex;gap:12px;flex-wrap:wrap;margin:15px 0}}.representative{{border-top:1px solid var(--line);padding:16px 0}}.representative h4{{margin:4px 0;font-size:15px}}.muted{{color:var(--muted)}}footer{{color:var(--muted);text-align:center}}@media(max-width:760px){{.cards{{grid-template-columns:1fr 1fr}}section{{padding:20px 16px}}header h1{{font-size:26px}}iframe{{height:700px}}}}</style></head><body>
<header><h1>四库联合 BERTopic 文献主题分析</h1><p>CNKI · 万方 · PubMed · Web of Science　｜　分析日期：2026-10-08</p><p>全部记录共同建模 · 未去重 · 硅基流动 BAAI/bge-m3</p></header><main>
<nav><a href="#overview">结果概览</a><a href="#data">数据</a><a href="#themes">主题</a><a href="#figures">交互图</a><a href="#trends">年度变化</a><a href="#quality">质量与稳定性</a><a href="#chinese">中文对照</a><a href="#downloads">下载</a></nav>
<section id="overview"><h2>结果概览</h2><div class="cards"><div class="card"><strong>{fmt(len(assignments))}</strong><span>输入文献记录</span></div><div class="card"><strong>{metrics["topics"]}</strong><span>非离群主题</span></div><div class="card"><strong>{metrics["outlier_rate"]:.1%}</strong><span>离群记录比例</span></div><div class="card"><strong>{fmt(manifest["chinese_texts"])}</strong><span>中文文本记录</span></div></div><p>同一篇文章在多个库出现、或在 WoS 内重复导出的记录均保留，每条记录独立参与聚类密度和频数统计。共 {metrics["outliers"]:,} 条记录保留为主题 -1，不强制归入已有主题。</p><ul>{top_descriptions}</ul><div class="note">主题数量、占比与年度变化描述的是这批未去重的输入记录。WoS 占原始记录约 89%，且检索范围包含其他谷物；当前结果覆盖用户提供的全部文献范围，不能直接称为严格筛选后的“水稻胚乳领域总体”。</div></section>
<section id="data"><h2>数据与处理口径</h2><div class="table-scroll"><table><tr><th>数据库</th><th>记录数</th><th>有摘要</th><th>缺摘要</th><th>记录占比</th></tr>{source_rows}</table></div><p>建模文本为标题＋摘要。{manifest["missing_abstracts"]} 条缺摘要记录使用标题并保留标记。仅清理 HTML 标记、明确版权尾注和多余空白；语义嵌入保留原文词序、数字、基因名称及科学符号。中英文分词仅用于主题关键词提取。</p><p>WoS 保留全部 20,092 条记录，对应 19,092 个唯一 WoS ID。跨库 DOI、同名标题和相同文本的重复情况以字段标记，没有删除。相同文本仅在 API 向量缓存阶段复用计算。</p><p>CNKI 的 .xls 实际是 HTML 表格；万方包含两层导出表头，均已按实际格式读取。年份缺失时使用 CNKI 发表时间补齐。原始 2027 年记录保留参与建模并标记复核，在截至 2026 年的趋势图中排除；2026 年尚未完整。</p></section>
<section id="themes"><h2>主题及代表文献</h2><p>主题标签为关键词概括，便于检索。代表文献按与主题中心的余弦相似度排序，同一文本在示例中只展示一次；这不会改变建模样本或主题频数。</p><div class="controls"><input id="topic-search" placeholder="搜索主题标签或关键词"><select id="topic-select"></select></div><div id="topic-detail"></div><details><summary>查看全部主题表</summary><div class="table-scroll"><table class="topics-table"><tr><th>主题</th><th>标签</th><th>记录数</th><th>全部记录占比</th><th>关键词</th></tr>{topic_rows}</table></div></details></section>
<section id="figures"><h2>交互图与可导出图形</h2><p><a href="joint_bge_m3/figure_gallery.html">打开完整主题图库：15 类主图、全部 42 个主题独立图、PNG / SVG / PDF 与自动校验清单</a></p><p>图表支持缩放、筛选与 SVG 导出。语义分布图是五维 UMAP 的前两个坐标，图上距离只作探索；它不是主题正确性的独立验证。</p>{figures_section}<p><a href="joint_bge_m3/figures/fig04_topic_hierarchy.svg">主题层次树 SVG</a> · <a href="joint_bge_m3/figures/fig02_topic_keywords.svg">主要主题关键词 SVG</a></p></section>
<section id="trends"><h2>近期主题占比变化</h2><p>比较两个完整五年窗口：2016–2020 与 2021–2025。分母分别为窗口内全部记录（包含离群记录），以下列出近五年不少于 25 条记录的占比增幅前列主题。此处是样本内的观察性变化，不是检索覆盖校正后的学科增长率。</p><div class="table-scroll"><table><tr><th>主题</th><th>2016–2020 记录</th><th>2021–2025 记录</th><th>前期占比</th><th>近期占比</th><th>变化，百分点</th></tr>{change_rows}</table></div><p>逐年频数与占比见下载数据；另提供 BERTopic 五年区间的关键词演化结果。所有年份使用同一联合模型的主题定义，避免逐年独立建模后误把同号主题进行比较。</p></section>
<section id="quality"><h2>参数、质量诊断与稳定性</h2><p>向量为 API 返回的 1,024 维向量并进行 L2 归一化。UMAP：邻居数 30、维数 5、余弦距离、min_dist=0、随机种子 42。HDBSCAN 比较三组最小簇规模，min_samples=10。c-TF-IDF 使用中英文分词和 1–2 gram，减弱高频词影响。</p><div class="table-scroll"><table><tr><th>最小簇规模</th><th>主题数</th><th>离群率</th><th>原始向量轮廓系数</th><th>平均 NPMI</th><th>关键词多样性</th><th>选择分数</th></tr>{candidate_rows}</table></div><p>选定最小簇规模：{metrics["min_cluster_size"]}。选择分数是覆盖率、语义轮廓系数、NPMI 和关键词多样性的启发式加权组合，用于这三组候选之间的比较，没有通用合格阈值。</p><p>NPMI 根据本语料中文档级的前十关键词共现计算（非滑动窗口）；不存在共现的词对记为 -1。语义轮廓系数使用原始嵌入空间的余弦距离，在已分配主题的记录中最多抽样 1,500 条。关键词多样性为前十关键词中不同词项的比例。</p><p>改用 UMAP 种子 7 的稳定性检查：全部记录 ARI={stability["ari_all_including_noise"]:.4f}，NMI={stability["nmi_all_including_noise"]:.4f}；两个种子下均非离群的 {stability["both_assigned_documents"]:,} 条记录 ARI={stability["ari_both_assigned"]:.4f}。两次主题编号无需相同，ARI/NMI 对编号置换不敏感。</p><p>主题与数据库的关联 NMI={associations["topic_source_nmi"]:.4f}；主题与文本语言的关联 NMI={associations["topic_language_nmi"]:.4f}。这两项只作描述性诊断。HDBSCAN membership strength 表示簇成员强度，不是主题正确性的概率。最近主题及余弦相似度仅作辅助建议，没有改变离群记录的原始归属。</p></section>
{chinese_section}
<section id="downloads"><h2>完整结果与复现</h2><ul><li><a href="merged_four_databases.csv">四库合并数据，全部记录</a></li><li><a href="joint_bge_m3/document_topics.csv">每条文献的主题、成员强度和原始元数据</a></li><li><a href="joint_bge_m3/topic_summary.csv">主题汇总</a> · <a href="joint_bge_m3/topic_keywords.csv">关键词权重</a> · <a href="joint_bge_m3/representative_documents.csv">代表文献</a></li><li><a href="joint_bge_m3/topics_by_year.csv">年度主题频数与占比</a> · <a href="joint_bge_m3/topics_by_source.csv">数据库主题分布</a> · <a href="joint_bge_m3/topics_by_language.csv">语言主题分布</a></li><li><a href="joint_bge_m3/parameter_comparison.csv">参数比较</a> · <a href="joint_bge_m3/seed_stability_assignments.csv">随机种子稳定性逐条结果</a> · <a href="joint_bge_m3/recent_topic_changes.csv">近期变化</a></li><li><a href="joint_bge_m3/topic_keywords_over_time.csv">五年区间关键词演化</a> · <a href="joint_bge_m3/topic_similarity.csv">主题相似度</a></li><li><a href="data_manifest.json">数据口径和源文件校验值</a> · <a href="runtime_versions.json">运行库版本</a> · <a href="../README.md">运行与方法说明</a></li></ul><p>完整本地模型、可移植模型、嵌入矩阵、词典和脚本保存在分析目录。API 密钥没有写入脚本、日志、模型或报告。本次使用远端嵌入 API，本地 CPU 完成降维聚类，没有使用本地 GPU。</p></section>
<section><h2>方法来源</h2><p><a href="https://maartengr.github.io/BERTopic/algorithm/algorithm.html">BERTopic 算法</a> · <a href="https://maartengr.github.io/BERTopic/faq.html">中文分词与稳定性说明</a> · <a href="https://api-docs.siliconflow.cn/docs/api/embeddings-post">硅基流动嵌入接口</a> · <a href="https://huggingface.co/BAAI/bge-m3">BGE-M3</a> · <a href="https://huggingface.co/BAAI/bge-large-zh-v1.5">BGE 中文模型</a></p></section><footer>四库联合分析 · 2026-10-08</footer></main>
<script>const data={browser_data};const select=document.getElementById('topic-select');const detail=document.getElementById('topic-detail');function esc(s){{return String(s??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));}}function draw(){{const t=data.topics.find(t=>String(t.topic)===select.value);if(!t){{detail.textContent='没有匹配主题';return;}}let text='<h3>T'+t.topic+' '+esc(t.label)+'</h3><p>'+t.count.toLocaleString()+' 条记录，占全部记录 '+(t.share_all_documents*100).toFixed(2)+'%。关键词：'+esc(t.keywords)+'</p>';for(const r of data.representatives[String(t.topic)]||[]){{text+='<div class="representative"><h4>'+r.rank+'. '+esc(r.title)+'</h4><p class="muted">'+esc(r.source)+' · '+esc(r.year)+' · '+esc(r.record_id)+' · 主题中心余弦 '+Number(r.centroid_cosine).toFixed(3)+'</p><p>'+esc(r.abstract||'原导出无摘要')+'</p>'+(r.doi?'<a href="https://doi.org/'+encodeURIComponent(r.doi)+'" target="_blank" rel="noopener">DOI 文献链接</a>':'')+'</div>';}}detail.innerHTML=text;}}function filter(){{const q=document.getElementById('topic-search').value.toLowerCase();const rows=data.topics.filter(t=>(t.label+' '+t.keywords+' '+t.topic).toLowerCase().includes(q));select.innerHTML=rows.map(t=>'<option value="'+t.topic+'">T'+t.topic+' '+esc(t.label)+' ('+t.count+')</option>').join('');draw();}}select.addEventListener('change',draw);document.getElementById('topic-search').addEventListener('input',filter);filter();</script></body></html>'''
    interpretation_section = f'''<section><h2>当前结果的解释边界</h2><p>联合模型的原始向量轮廓系数为 {metrics["semantic_silhouette_sample1500"]:.4f}，接近零，主题间仍有明显重叠。主模型有 {metrics["outliers"]:,} 条离群记录，应把主题和代表文献作为探索性分组审阅，不能把自动标签视为已经验证的分类。</p><p>随机种子由 42 改为 7 后，主题数从 {metrics["topics"]} 变为 {stability["alternative_topics"]}，离群率从 {metrics["outlier_rate"]:.2%} 变为 {stability["alternative_outlier_rate"]:.2%}。两次都已归类的记录一致性较高，但离群边界与主题数量存在敏感性。</p><p>{manifest["chinese_texts"]} 条中文文本中，{chinese_main_count} 条（{chinese_main_share:.1%}）集中在 T{chinese_main_topic}，说明联合模型存在明显语言分组倾向。该主题包含多个研究方向，不能据此声称中文研究只有一个实质主题。中文子集的单独拟合用于细化分析，编号与联合主题没有一一对应关系。</p><p>两个中文模型在相同最小簇规模 15、min_samples=3、15 邻居和种子 42 下比较。bge-m3 在最小簇规模 25 的候选中合并为两个粗主题，覆盖率较高使启发式分数偏高，因此中文模型对照固定相同参数，避免把聚类粒度差异当成嵌入模型效果。</p></section>'''
    page = page.replace('<section id="data">', interpretation_section + '<section id="data">', 1)
    if comparisons:
        chinese_summary = read_table(OUT / "chinese_bge_zh", "topic_summary")
        chinese_representatives = read_table(OUT / "chinese_bge_zh", "representative_documents")
        chinese_rows = "".join(f'<tr><td>T{int(row.topic)}</td><td>{e(row.label)}</td><td>{int(row.count)}</td><td>{e(row.keywords)}</td></tr>' for row in chinese_summary.itertuples())
        chinese_examples = "".join(f'<details><summary>T{int(row.topic)} {e(row.label)}：代表文献</summary>' + "".join(f'<div class="representative"><h4>{int(rep.rank)}. {e(rep.title)}</h4><p class="muted">{e(rep.source)} · {e(rep.year)} · {e(rep.record_id)}</p><p>{e(rep.abstract or "原导出无摘要")}</p></div>' for rep in chinese_representatives.loc[chinese_representatives.topic.eq(row.topic)].itertuples()) + '</details>' for row in chinese_summary.itertuples())
        chinese_detail = f'''<section><h2>中文专用模型的细分主题</h2><p>bge-large-zh-v1.5 将中文子集划为 {comparisons["bge_zh"]["topics"]} 个主题，保留 {comparisons["bge_zh"]["outliers"]} 条离群记录。部分原始导出含会议或技术资讯，同样按全部保留口径处理。</p><details><summary>中文主题汇总表</summary><div class="table-scroll"><table><tr><th>主题</th><th>标签</th><th>记录数</th><th>关键词</th></tr>{chinese_rows}</table></div></details>{chinese_examples}<p><a href="chinese_bge_zh/document_topics.csv">中文专用模型逐条结果</a> · <a href="chinese_bge_zh/topic_summary.csv">中文主题汇总</a> · <a href="chinese_bge_zh/parameter_comparison.csv">中文参数敏感性</a> · <a href="chinese_bge_m3_comparison/parameter_comparison.csv">中文 bge-m3 参数敏感性</a></p></section>'''
        page = page.replace('<section id="downloads">', chinese_detail + '<section id="downloads">', 1)
    (OUT / "analysis_report.html").write_text(page, encoding="utf-8")
    print(json.dumps({"stage": "report_complete", "report": str(OUT / "analysis_report.html"), "topics": metrics["topics"]}), flush=True)


if __name__ == "__main__":
    main()
