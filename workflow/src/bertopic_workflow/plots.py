"""Fifteen figure categories with explicit applicability and numeric evidence."""
from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
import numpy as np
import plotly.graph_objects as go

from .common import WorkflowError, read_json, write_json
from .modeling import Tokenizer

TITLES = ['主题总览', '主题关键词', '主题词云', '主题层级', '文档语义分布',
          '主题与关键词热力图', '主题词秩', '主题时间趋势', '主题分类对比', '主题相似度',
          '主题关系网络', '主题规模分布', '离群记录占比', '主题质量诊断', '主题构成桑基图']
PALETTE = ['#176b87', '#d17a45', '#599977', '#9078a8', '#bd6277', '#8c8a48', '#4e93ac', '#725847']


def color(topic):
    return '#aab4bd' if topic == -1 else PALETTE[int(topic) % len(PALETTE)]


def wrapped(value, width=24):
    return '\n'.join(textwrap.wrap(str(value), width=width, break_long_words=True))


def style_font(config, corpus):
    candidates = [config.report.font_path, 'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    font = next((str(p) for p in candidates if p and Path(p).is_file()), None)
    if font:
        font_manager.fontManager.addfont(font)
        family = font_manager.FontProperties(fname=font).get_name()
    else:
        family = next((f.name for f in font_manager.fontManager.ttflist
                       if any(n in f.name for n in ['Noto Sans CJK', 'WenQuanYi', 'Microsoft YaHei', 'SimHei'])), None)
        if not family:
            raise WorkflowError('FONT_MISSING', '请安装中文字体或通过 report.font_path 指定字体文件。')
    plt.rcParams.update({'font.family': family, 'svg.fonttype': 'none', 'svg.hashsalt': 'dsh-bertopic-v1',
                         'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return font, family


class Renderer:
    def __init__(self, config, ingest_dir, model_dir, destination):
        self.config, self.directory = config, Path(destination)
        self.model_dir = Path(model_dir)
        self.docs = read_json(self.model_dir / 'document_topics.json')
        self.summary = read_json(self.model_dir / 'topic_summary.json')
        self.keywords = read_json(self.model_dir / 'topic_keywords.json')
        self.metrics = read_json(self.model_dir / 'metrics.json')
        self.stability = read_json(self.model_dir / 'stability.json')
        self.candidates = read_json(self.model_dir / 'parameter_comparison.json')
        self.ids = read_json(self.model_dir / 'topic_ids.json')
        self.labels = {int(t): v for t, v in read_json(self.model_dir / 'topic_labels.json').items()}
        self.similarity = np.load(self.model_dir / 'topic_similarity.npy', allow_pickle=False)
        self.top = [r['topic'] for r in self.summary[:config.report.top_topics]]
        self.counts = {r['topic']: r['count'] for r in self.summary}
        self.catalog = []
        self.font, self.family = style_font(config, self.docs)
        self.tokenizer = Tokenizer(config.model.stopwords, config.model.domain_terms)
        self.ingest = read_json(Path(ingest_dir) / 'input_manifest.json')
        self.note = (f"保留记录 N={len(self.docs):,}；去重={config.import_options.deduplication}；"
                     f"主题 -1 为离群；截止年={config.report.year_end}；标签来自关键词，需结合文献解释。")
        (self.directory / 'figures').mkdir()
        (self.directory / 'interactive').mkdir()
        (self.directory / 'evidence').mkdir()

    def name(self, topic):
        return f'T{topic} {self.labels.get(topic, "")}'

    def words(self, topic, count=10):
        return [r for r in self.keywords if r['topic'] == topic][:count]

    def save(self, figure, name):
        files = []
        figure.text(.015, .012, self.note, fontsize=8, color='#647480', va='bottom')
        figure.tight_layout(rect=(0, getattr(figure, '_workflow_bottom', .055), 1, .95))
        for fmt in self.config.report.formats:
            relative = f'figures/{name}.{fmt}'
            metadata = {'Date': None} if fmt == 'svg' else {'CreationDate': None, 'ModDate': None} if fmt == 'pdf' else None
            figure.savefig(self.directory / relative, dpi=self.config.report.dpi, bbox_inches='tight',
                           facecolor='white', metadata=metadata)
            files.append(relative)
        plt.close(figure)
        return files

    def interactive(self, figure, name):
        figure.update_layout(template='plotly_white', font={'family': self.family, 'size': 12},
                             margin={'l': 100, 'r': 80, 't': 100, 'b': 100}, height=760)
        relative = f'interactive/{name}.html'
        figure.write_html(self.directory / relative, include_plotlyjs='directory', div_id=name,
                          config={'displaylogo': False, 'responsive': True})
        return relative

    def emit(self, index, figure=None, interactive=None, evidence=None, reason=None, suffix=''):
        name = f'fig{index:02d}{suffix}'
        files = self.save(figure, name) if figure is not None else []
        if interactive is not None:
            files.append(self.interactive(interactive, name))
        if evidence is not None:
            write_json(self.directory / 'evidence' / (name + '.json'), evidence)
            files.append(f'evidence/{name}.json')
        self.catalog.append({'number': index, 'title': TITLES[index - 1], 'variant': suffix,
                             'status': 'generated' if figure is not None or interactive is not None else 'not_applicable',
                             'reason': reason, 'files': files})

    def skip(self, index, reason):
        self.emit(index, reason=reason)

    def overview(self):
        if not self.ids:
            return self.skip(1, '没有非离群主题。')
        from sklearn.manifold import MDS
        from sklearn.decomposition import PCA
        stress = None
        if len(self.ids) == 1:
            coords, method = np.zeros((1, 2)), 'single_topic'
        elif len(self.ids) <= 150:
            distance = np.clip(1 - self.similarity, 0, 2)
            np.fill_diagonal(distance, 0)
            mds = MDS(metric='precomputed', init='random', random_state=self.config.model.seed,
                      n_init=4, normalized_stress=True)
            coords, method = mds.fit_transform(distance), 'MDS_centroid_cosine_distance'
            stress = float(mds.stress_)
        else:
            coords = PCA(n_components=2).fit_transform(np.load(self.model_dir / 'topic_centroids.npy', allow_pickle=False))
            method = 'PCA_centroids_large_topic_set'
        self.layout = coords
        figure, ax = plt.subplots(figsize=(13, 9))
        counts = np.asarray([self.counts[t] for t in self.ids])
        ax.scatter(coords[:, 0], coords[:, 1], s=1600 * counts / counts.max(),
                   c=[color(t) for t in self.ids], alpha=.7, edgecolors='white')
        for t in self.top:
            i = self.ids.index(t)
            ax.annotate(f'T{t}', coords[i], ha='center', va='center', fontsize=10)
        ax.set(xlabel='投影坐标 1', ylabel='投影坐标 2', title=f'主题总览：气泡面积与记录数成正比\n{method}；normalized stress={stress if stress is not None else "不适用"}')
        chart = go.Figure(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode='markers',
            marker={'size': counts, 'sizemode': 'area', 'sizeref': 2 * counts.max() / 55**2,
                    'color': [color(t) for t in self.ids]},
            text=[escape(self.name(t)) for t in self.ids], customdata=counts,
            hovertemplate='%{text}<br>记录数 %{customdata}<extra></extra>'))
        chart.update_layout(title='主题中心的探索性投影：距离失真见数据文件')
        self.emit(1, figure, chart, {'topic_ids': self.ids, 'coordinates': coords, 'counts': counts,
                                   'method': method, 'normalized_stress': stress, 'bubble_area': 'linear_count'})

    def keyword_bars(self):
        if not self.top:
            return self.skip(2, '没有非离群主题关键词。')
        cols = min(4, len(self.top))
        rows = (len(self.top) + cols - 1) // cols
        figure, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)
        traces = []
        for index, topic in enumerate(self.top):
            items = self.words(topic, 8)[::-1]
            ax = axes.ravel()[index]
            ax.barh([r['keyword'] for r in items], [r['ctfidf_weight'] for r in items], color=color(topic))
            ax.set_title(wrapped(self.name(topic), 32), fontsize=11)
            ax.set_xlabel('c-TF-IDF 权重')
        for ax in axes.ravel()[len(self.top):]:
            ax.set_visible(False)
        for index, topic in enumerate(self.ids):
            items = self.words(topic)[::-1]
            traces.append(go.Bar(y=[escape(r['keyword']) for r in items], x=[r['ctfidf_weight'] for r in items],
                                 orientation='h', visible=index == 0, name=f'T{topic}'))
        chart = go.Figure(traces)
        chart.update_layout(title='主题关键词：选择主题', xaxis_title='c-TF-IDF 权重',
            updatemenus=[{'buttons': [{'label': f'T{t}', 'method': 'update',
                                      'args': [{'visible': [i == j for i in range(len(self.ids))]},
                                               {'title': escape(self.name(t))}]} for j, t in enumerate(self.ids)]}])
        self.emit(2, figure, chart, {'static_topic_ids': self.top, 'interactive_topic_ids': self.ids})

    def clouds(self):
        if not self.top:
            return self.skip(3, '没有非离群主题关键词。')
        from wordcloud import WordCloud
        selected = self.top[:12]
        cols = min(3, len(selected))
        figure, axes = plt.subplots((len(selected) + cols - 1) // cols, cols,
                                   figsize=(6 * cols, 4 * ((len(selected) + cols - 1) // cols)), squeeze=False)
        for ax, topic in zip(axes.ravel(), selected, strict=False):
            frequencies = {r['keyword']: r['ctfidf_weight'] for r in self.words(topic, self.config.model.top_n_words)}
            if frequencies:
                cloud = WordCloud(font_path=self.font, background_color='white', width=900, height=500,
                                  random_state=self.config.model.seed, collocations=False).generate_from_frequencies(frequencies)
                ax.imshow(cloud, interpolation='bilinear')
            ax.set_title(wrapped(self.name(topic), 30), fontsize=11)
            ax.axis('off')
        for ax in axes.ravel()[len(selected):]:
            ax.axis('off')
        self.emit(3, figure, evidence={'topic_ids': selected, 'weights': 'c-TF-IDF; not raw frequency',
                                      'raster_layers': ['wordcloud']})

    def hierarchy(self):
        if len(self.ids) < 2:
            return self.skip(4, '主题层级至少需要两个非离群主题。')
        from scipy.cluster.hierarchy import dendrogram
        tree = np.load(self.model_dir / 'hierarchy.npy', allow_pickle=False)
        figure, ax = plt.subplots(figsize=(15, max(6, min(20, len(self.ids) * .3))))
        result = dendrogram(tree, orientation='right', ax=ax, labels=[f'T{t}' for t in self.ids],
                            leaf_font_size=10, truncate_mode='lastp' if len(self.ids) > 60 else None,
                            p=60, show_leaf_counts=True)
        ax.set(title='主题层级：平均连接法、主题中心余弦距离', xlabel='余弦距离')
        chart = go.Figure()
        for x, y in zip(result['dcoord'], result['icoord'], strict=True):
            chart.add_trace(go.Scatter(x=x, y=y, mode='lines', showlegend=False,
                                      line={'color': '#176b87'}, hoverinfo='skip'))
        chart.update_layout(title='主题层级；主题过多时聚合叶节点显示簇内主题数',
                            yaxis={'tickvals': np.arange(len(result['ivl'])) * 10 + 5, 'ticktext': result['ivl']})
        self.emit(4, figure, chart, {'topic_ids': self.ids, 'linkage': tree,
                                   'static_truncated': len(self.ids) > 60, 'leaf_labels': result['ivl']})

    def document_map(self):
        docs = self.docs
        limit = self.config.report.max_plot_documents
        rng = np.random.default_rng(self.config.model.seed)
        if len(docs) > limit:
            positions = np.sort(rng.choice(len(docs), limit, replace=False))
            docs = [docs[int(i)] for i in positions]
        figure, ax = plt.subplots(figsize=(14, 10))
        chart = go.Figure()
        groups = sorted(set(r['topic'] for r in docs))
        for t in groups:
            selected = [r for r in docs if r['topic'] == t]
            ax.scatter([r['plot_x'] for r in selected], [r['plot_y'] for r in selected],
                       s=4, alpha=.55, color=color(t), rasterized=True)
            chart.add_trace(go.Scattergl(x=[r['plot_x'] for r in selected], y=[r['plot_y'] for r in selected],
                mode='markers', name=escape(self.name(t)), marker={'size': 4, 'color': color(t)},
                customdata=[[r['record_id'], escape(r['title']), r['source'], r['year']] for r in selected],
                hovertemplate='%{customdata[1]}<br>%{customdata[0]}<br>%{customdata[2]} %{customdata[3]}<extra></extra>'))
        ax.set(title=f'文献分布：显示 {len(docs):,}/{len(self.docs):,} 条；降维结果的前两个坐标',
               xlabel='UMAP 坐标 1', ylabel='UMAP 坐标 2')
        chart.update_layout(title='文献语义分布；投影不能独立证明聚类正确')
        self.emit(5, figure, chart, {'displayed_record_ids': [r['record_id'] for r in docs],
                                   'total_records': len(self.docs), 'sampling': 'uniform_without_replacement' if len(docs) < len(self.docs) else 'none',
                                   'topic_ids': groups, 'projection': 'first_two_coordinates_of_clustering_UMAP',
                                   'raster_layers': ['static_scatter']})

    def keyword_heatmap(self):
        if not self.top:
            return self.skip(6, '没有非离群主题关键词。')
        words = list(dict.fromkeys(r['keyword'] for t in self.top[:15] for r in self.words(t, 5)))
        lookup = {(r['topic'], r['keyword']): r['ctfidf_weight'] for r in self.keywords}
        matrix = np.array([[lookup.get((t, w), 0) for w in words] for t in self.top[:15]])
        figure, ax = plt.subplots(figsize=(max(13, len(words) * .25), max(7, len(matrix) * .4)))
        image = ax.imshow(matrix, aspect='auto', cmap='Blues')
        ax.set_xticks(range(len(words)), words, rotation=65, ha='right', fontsize=9)
        ax.set_yticks(range(len(matrix)), [f'T{t}' for t in self.top[:15]])
        ax.set_title('主题与关键词：c-TF-IDF 权重；仅显示所列主题与词项')
        figure.colorbar(image, ax=ax, label='c-TF-IDF 权重', shrink=.7)
        all_words = list(dict.fromkeys(r['keyword'] for t in self.ids for r in self.words(t, 5)))
        full = [[lookup.get((t, w), 0) for w in all_words] for t in self.ids]
        chart = go.Figure(go.Heatmap(z=full, x=[escape(w) for w in all_words], y=[f'T{t}' for t in self.ids], colorscale='Blues'))
        chart.update_layout(title='全部主题的核心关键词矩阵', yaxis_autorange='reversed')
        self.emit(6, figure, chart, {'static_topics': self.top[:15], 'keywords': words, 'weights': matrix,
                                   'interactive_topics': self.ids, 'interactive_keywords': all_words})

    def ranks(self):
        if not self.top:
            return self.skip(7, '没有非离群主题关键词。')
        figure, axes = plt.subplots(1, 2, figsize=(18, max(7, len(self.top) * .35)))
        chart = go.Figure()
        ratios = []
        for t in self.ids:
            weights = [r['ctfidf_weight'] for r in self.words(t, self.config.model.top_n_words)]
            chart.add_trace(go.Scatter(x=list(range(1, len(weights) + 1)), y=weights, name=f'T{t}', mode='lines'))
            if t in self.top:
                axes[0].loglog(range(1, len(weights) + 1), weights, label=f'T{t}', color=color(t))
                ratios.append(weights[0] / weights[9] if len(weights) >= 10 and weights[9] > 0 else None)
        axes[0].set(title='关键词权重衰减（所列主题）', xlabel='关键词秩', ylabel='c-TF-IDF 权重')
        axes[0].legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, -.15), fontsize=9)
        axes[1].barh(range(len(self.top)), [v or 0 for v in ratios], color=[color(t) for t in self.top])
        axes[1].set_yticks(range(len(self.top)), [f'T{t}' for t in self.top], fontsize=11)
        axes[1].invert_yaxis()
        axes[1].set(title='首位与第 10 位权重比（不足 10 词标为缺失）', xlabel='权重比')
        for i, value in enumerate(ratios):
            if value is None:
                axes[1].annotate('缺失', (0, i), xytext=(5, 0), textcoords='offset points')
        chart.update_layout(title='全部主题词秩', xaxis_type='log', yaxis_type='log')
        self.emit(7, figure, chart, {'static_topics': self.top, 'first_to_tenth_weight_ratio': ratios})

    def eligible_years(self):
        options = self.config.report
        return [r for r in self.docs if r['year'] is not None and r['year'] <= options.year_end
                and (options.year_start is None or r['year'] >= options.year_start)]

    def trends(self):
        docs = self.eligible_years()
        if not docs or not self.top:
            return self.skip(8, '没有可用年份或非离群主题。')
        years = sorted(set(r['year'] for r in docs))
        plot_years = list(range(years[0], years[-1] + 1))
        missing_years = sorted(set(plot_years) - set(years))
        totals = Counter(r['year'] for r in docs)
        counts = Counter((r['year'], r['topic']) for r in docs)
        figure, axes = plt.subplots(2, 1, figsize=(17, 12), sharex=True)
        chart = go.Figure()
        shares = []
        for t in self.ids:
            values = [counts[y, t] / totals[y] for y in years]
            plot_values = [counts[y, t] / totals[y] if totals[y] else np.nan for y in plot_years]
            if t in self.top:
                axes[0].plot(plot_years, np.asarray(plot_values) * 100, '.-', label=f'T{t}', color=color(t))
                shares.append(values)
            chart.add_trace(go.Scatter(x=plot_years, y=plot_values, mode='lines+markers',
                                      connectgaps=False, name=f'T{t}'))
        remaining = [1 - sum(counts[y, t] / totals[y] for t in self.top) for y in years]
        bottom = np.zeros(len(years))
        for values, label, shade in zip(shares + [remaining], [f'T{t}' for t in self.top] + ['其他主题及离群'],
                                       [color(t) for t in self.top] + ['#c4cbd0'], strict=True):
            height = np.asarray(values) * 100
            axes[1].bar(years, height, bottom=bottom, width=.85, label=label, color=shade)
            bottom += height
        for ax in axes:
            ax.set_ylabel('当年全部有效记录占比（%）')
            ax.set_ylim(0, 100)
        axes[0].set_title(f'年度主题占比；无记录年份断线；不完整年份={self.config.report.incomplete_year or "未指定"}')
        axes[1].set(xlabel='发表年份', ylim=(0, 100), title='年度记录构成（包含其他主题及离群）')
        handles, labels = axes[1].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=min(6, len(labels)), bbox_to_anchor=(.5, .055), fontsize=9)
        figure.subplots_adjust(bottom=.2, hspace=.25)
        figure._workflow_bottom = .20
        chart.update_layout(title='全部主题的年度占比：分母包含离群记录', yaxis_tickformat='.0%')
        self.emit(8, figure, chart, {'years': years, 'annual_denominators': [totals[y] for y in years],
                                   'static_topics': self.top, 'shares': shares, 'remainder': remaining,
                                   'missing_years': missing_years, 'gap_policy': 'break_without_imputation',
                                   'composition_encoding': 'observed_annual_stacked_bars',
                                   'included': len(docs), 'excluded': len(self.docs) - len(docs)})

    def groups(self):
        fields = [('source', '数据库'), ('journal', '期刊'), ('institutions', '机构'),
                  ('countries', '国家/地区'), ('category', '原始分类'), ('language_of_text', '文本语言')]
        figure, axes = plt.subplots(2, 3, figsize=(23, max(11, len(self.top) * .7)))
        evidence, generated = {}, False
        for ax, (field, label) in zip(axes.ravel(), fields, strict=True):
            members = []
            for row in self.docs:
                values = {v.strip() for v in re.split(r'[;；|、]', str(row.get(field) or '')) if v.strip()}
                members.extend((row['record_id'], row['topic'], value) for value in values)
            group_counts = Counter(value for _, _, value in members)
            ordered = [g for g, _ in group_counts.most_common(8)]
            if not ordered or not self.top:
                ax.text(.5, .5, f'{label}：没有原始字段覆盖或有效主题', ha='center', wrap=True)
                ax.axis('off')
                evidence[field] = {'status': 'not_applicable', 'covered_records': len({i for i, _, _ in members})}
                continue
            generated = True
            counts = Counter((group, topic) for _, topic, group in members)
            matrix = [[counts[g, t] / group_counts[g] for g in ordered] for t in self.top]
            image = ax.imshow(matrix, aspect='auto', cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(len(ordered)), [wrapped(g, 13) for g in ordered], fontsize=8, rotation=40, ha='right')
            ax.set_yticks(range(len(self.top)), [f'T{t}' for t in self.top], fontsize=9)
            ax.set_title(label + '：组内占比（包含离群）')
            figure.colorbar(image, ax=ax, shrink=.6)
            evidence[field] = {'status': 'generated', 'covered_records': len({i for i, _, _ in members}),
                               'total_records': len(self.docs), 'groups': ordered,
                               'group_denominators': [group_counts[g] for g in ordered], 'topics': self.top, 'shares': matrix,
                               'counting': 'unique record/group; multi-membership full counting; no entity resolution'}
            chart = go.Figure(go.Heatmap(z=matrix, x=[escape(g) for g in ordered], y=[f'T{t}' for t in self.top], zmin=0, zmax=1, colorscale='Blues'))
            chart.update_layout(title=label + '：已覆盖原始字段的组内主题占比', yaxis_autorange='reversed')
            relative = self.interactive(chart, 'fig09_' + field)
            evidence[field]['interactive'] = relative
        if generated:
            self.emit(9, figure, evidence=evidence)
        else:
            plt.close(figure)
            self.skip(9, '没有可用分组元数据或非离群主题。')

    def similarities(self):
        if not self.ids:
            return self.skip(10, '没有非离群主题。')
        order = [self.ids.index(t) for t in self.top]
        matrix = self.similarity[np.ix_(order, order)]
        figure, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(order)), [f'T{t}' for t in self.top], rotation=90)
        ax.set_yticks(range(len(order)), [f'T{t}' for t in self.top])
        ax.set_title('主题中心余弦相似度（静态显示所列主题）')
        figure.colorbar(image, ax=ax, label='余弦相似度')
        chart = go.Figure(go.Heatmap(z=self.similarity, x=[f'T{t}' for t in self.ids], y=[f'T{t}' for t in self.ids], zmin=-1, zmax=1, colorscale='RdBu'))
        chart.update_layout(title='全部主题中心余弦相似度', yaxis_autorange='reversed')
        self.emit(10, figure, chart, {'topic_ids': self.ids, 'cosine_similarity': self.similarity, 'static_topics': self.top})

    def network(self):
        if len(self.ids) < 2:
            return self.skip(11, '关系网络至少需要两个非离群主题。')
        pairs = self.similarity[np.triu_indices(len(self.ids), 1)]
        threshold = max(self.config.report.network_min_cosine, float(np.quantile(pairs, self.config.report.network_quantile)))
        figure, ax = plt.subplots(figsize=(13, 10))
        chart, edges = go.Figure(), []
        for i in range(len(self.ids)):
            for j in range(i + 1, len(self.ids)):
                if self.similarity[i, j] >= threshold:
                    x, y = self.layout[[i, j], 0], self.layout[[i, j], 1]
                    ax.plot(x, y, color='#9cabb7', alpha=.6, zorder=0)
                    chart.add_trace(go.Scatter(x=x, y=y, mode='lines', line={'width': 1, 'color': '#adb8c0'}, showlegend=False, hoverinfo='skip'))
                    edges.append({'source': self.ids[i], 'target': self.ids[j], 'cosine': self.similarity[i, j]})
        counts = np.asarray([self.counts[t] for t in self.ids])
        ax.scatter(self.layout[:, 0], self.layout[:, 1], s=1300 * counts / counts.max(), c=[color(t) for t in self.ids], alpha=.85)
        for t in self.top:
            ax.annotate(f'T{t}', self.layout[self.ids.index(t)], ha='center', va='center')
        ax.set_title(f'主题关系：余弦相似度 ≥ {threshold:.6f}；{len(edges)} 条边\n节点面积与记录数成正比；边不表示因果')
        chart.add_trace(go.Scatter(x=self.layout[:, 0], y=self.layout[:, 1], mode='markers',
            marker={'size': counts, 'sizemode': 'area', 'sizeref': 2 * counts.max() / 45**2, 'color': [color(t) for t in self.ids]},
            text=[escape(self.name(t)) for t in self.ids], hovertemplate='%{text}<extra></extra>', showlegend=False))
        chart.update_layout(title=f'主题关系：精确阈值 {threshold:.6f}；节点面积与记录数成正比')
        self.emit(11, figure, chart, {'threshold': threshold, 'edges': edges, 'node_area': 'linear_count', 'layout_reused_from_fig01': True})

    def sizes(self):
        if not self.summary:
            return self.skip(12, '所有记录均为离群，没有非离群主题规模。')
        selected = self.summary[:min(80, len(self.summary))]
        counts = [r['count'] for r in selected]
        denominator = sum(r['count'] for r in self.summary)
        cumulative = np.cumsum(counts) / denominator
        figure, ax = plt.subplots(figsize=(max(13, len(selected) * .3), 8))
        ax.bar(range(len(counts)), counts, color=[color(r['topic']) for r in selected])
        ax.set_xticks(range(len(counts)), [f'T{r["topic"]}' for r in selected], rotation=90)
        ax.set(title='主题规模：静态最多显示 80 主题，交互版包含全部主题', ylabel='记录数（非离群）')
        twin = ax.twinx()
        twin.plot(range(len(counts)), cumulative * 100, color='#bd6277')
        twin.set(ylabel=f'累计占已归类记录比例（%；分母={denominator:,}）', ylim=(0, 105))
        all_counts = [r['count'] for r in self.summary]
        chart = go.Figure(go.Bar(x=[f'T{r["topic"]}' for r in self.summary], y=all_counts, name='记录数'))
        chart.add_trace(go.Scatter(x=[f'T{r["topic"]}' for r in self.summary], y=np.cumsum(all_counts) / denominator, yaxis='y2', name='已归类累计比例'))
        chart.update_layout(title=f'主题规模与累计比例（已归类分母={denominator:,}）', yaxis2={'overlaying': 'y', 'side': 'right', 'range': [0, 1.05], 'tickformat': '.0%'})
        self.emit(12, figure, chart, {'topics': [r['topic'] for r in self.summary], 'counts': all_counts,
                                    'cumulative_denominator': denominator, 'outliers_excluded_from_pareto': self.metrics['outliers']})

    def outliers(self):
        figure, axes = plt.subplots(2, 2, figsize=(15, 11))
        noise, total = self.metrics['outliers'], len(self.docs)
        axes[0, 0].pie([total - noise, noise], labels=['已归类', '离群'], colors=[PALETTE[0], '#aab4bd'], autopct='%1.1f%%', wedgeprops={'width': .45})
        axes[0, 0].set_title(f'离群记录：{noise:,}/{total:,}')
        sources = sorted(set(r['source'] for r in self.docs))
        rates = [sum(r['topic'] == -1 for r in self.docs if r['source'] == s) / sum(r['source'] == s for r in self.docs) for s in sources]
        axes[0, 1].bar(sources, np.array(rates) * 100, color=PALETTE[0])
        axes[0, 1].set(title='分库离群率', ylabel='库内全部有效记录占比（%）', ylim=(0, 100))
        years = sorted(set(r['year'] for r in self.eligible_years()))
        yearly = [sum(r['topic'] == -1 and r['year'] == y for r in self.docs) / sum(r['year'] == y for r in self.docs) for y in years]
        if years:
            dense_years = list(range(years[0], years[-1] + 1))
            lookup = dict(zip(years, yearly, strict=True))
            axes[1, 0].plot(dense_years, [lookup.get(y, np.nan) * 100 for y in dense_years], '.-')
        else:
            axes[1, 0].text(.5, .5, '没有可用年份', ha='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set(title='逐年离群率（仅配置时间范围；无记录年份断线）',
                       ylabel='当年全部有效记录占比（%）', xlabel='年份', ylim=(0, 100))
        for label, assigned in [('已归类', True), ('离群', False)]:
            values = [r['nearest_cosine'] for r in self.docs if (r['topic'] >= 0) == assigned and r['nearest_cosine'] is not None]
            if values:
                axes[1, 1].hist(values, bins=30, alpha=.55, density=True, label=label)
        axes[1, 1].set(title='最近主题中心的余弦相似度', xlabel='余弦相似度', ylabel='组内概率密度')
        if self.ids:
            axes[1, 1].legend()
        else:
            axes[1, 1].text(.5, .5, '无非离群主题中心', transform=axes[1, 1].transAxes, ha='center')
        chart = go.Figure(go.Bar(x=sources, y=rates))
        chart.update_layout(title='分库离群率', yaxis_tickformat='.0%', yaxis_range=[0, 1])
        self.emit(13, figure, chart, {'assigned': total - noise, 'outliers': noise, 'sources': sources, 'rates': rates,
                                    'nearest_similarity_metric': 'centroid_cosine_for_both_groups'})

    def quality(self):
        selected = self.summary[:min(60, len(self.summary))]
        height = max(12, min(28, len(selected) * .4))
        figure, axes = plt.subplots(2, 3, figsize=(24, height))
        for ax, field, label in [(axes[0, 0], 'npmi', '文档共现 NPMI'), (axes[1, 1], 'centroid_cohesion', '主题中心内聚力'),
                                 (axes[1, 2], 'keyword_exclusivity', '前十关键词独享占比')]:
            values = [r[field] for r in selected]
            valid = [(i, v) for i, v in enumerate(values) if v is not None]
            ax.barh([i for i, _ in valid], [v for _, v in valid], color=[color(selected[i]['topic']) for i, _ in valid])
            ax.set_yticks(range(len(selected)), [f'T{r["topic"]}' for r in selected], fontsize=10)
            ax.invert_yaxis()
            mean = np.mean([v for _, v in valid]) if valid else None
            ax.set_title(label + (f'；所列主题均值={mean:.3f}' if mean is not None else '；不可计算'))
        silhouettes = read_json(self.model_dir / 'silhouette_samples.json')
        if silhouettes:
            axes[0, 1].hist([r['silhouette'] for r in silhouettes], bins=30, color=PALETTE[0])
        else:
            axes[0, 1].text(.5, .5, '可用样本不足，轮廓系数不可计算', ha='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title(f'原始向量余弦轮廓系数；样本={len(silhouettes)}\n均值={self.metrics["silhouette_cosine"]}')
        axes[0, 1].set_xlabel('轮廓系数（不含离群；接近零表示重叠）')
        axes[0, 2].plot([r['min_cluster_size'] for r in self.candidates], [r['outlier_rate'] for r in self.candidates], 'o-', label='离群率')
        axes[0, 2].set(title='候选参数比较', xlabel='最小簇规模', ylabel='离群率')
        if self.stability:
            axes[1, 0].bar([str(r['seed']) for r in self.stability], [r['ari_all'] for r in self.stability], color=PALETTE[0])
        else:
            axes[1, 0].text(.5, .5, '本次未配置种子比较', ha='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set(title='种子比较：全部记录 ARI\n有限种子比较，不是置信区间', xlabel='比较种子', ylabel='ARI（包括离群）')
        chart = go.Figure(go.Bar(x=[f'T{r["topic"]}' for r in self.summary], y=[r['npmi'] for r in self.summary]))
        chart.update_layout(title='全部主题 NPMI；算法诊断不等于结论正确性认证')
        self.emit(14, figure, chart, {'metrics': self.metrics, 'stability': self.stability, 'static_topics': [r['topic'] for r in selected],
                                    'npmi_by_topic': {r['topic']: r['npmi'] for r in self.summary},
                                    'silhouette_sample_records': len(silhouettes)})

    def sankey(self):
        docs = self.eligible_years()
        by_database = not docs
        if by_database:
            docs = self.docs
        width = self.config.report.time_bin_years
        bins = sorted(set((r['year'] // width) * width for r in docs)) if not by_database else []
        left = sorted(set(r['source'] for r in docs)) if by_database else [f'{b}–{min(b + width - 1, self.config.report.year_end)}' for b in bins]
        right = [f'T{t}' for t in self.top] + (['其他主题'] if set(self.ids) - set(self.top) else []) + (['离群'] if any(r['topic'] == -1 for r in docs) else [])
        flows = Counter()
        for row in docs:
            start = (row['year'] // width) * width if not by_database else None
            l = row['source'] if by_database else f'{start}–{min(start + width - 1, self.config.report.year_end)}'
            r = f'T{row["topic"]}' if row['topic'] in self.top else '离群' if row['topic'] == -1 else '其他主题'
            flows[l, r] += 1
        right = [r for r in right if any(b == r for a, b in flows)]
        nodes = left + right
        chart = go.Figure(go.Sankey(node={'label': nodes}, link={'source': [nodes.index(a) for a, b in flows],
                           'target': [nodes.index(b) for a, b in flows], 'value': list(flows.values())}))
        chart.update_layout(title=('数据库' if by_database else '时期') + '到主题的记录构成；不表示主题迁移、分裂或合并')
        figure, ax = plt.subplots(figsize=(18, max(10, min(25, max(len(left), len(right)) * .55))))
        # Independent node scales on each side keep all ribbons inside their nodes.
        def layout(labels, side):
            totals = {label: sum(count for (a, b), count in flows.items() if (a if side == 0 else b) == label) for label in labels}
            gap = min(.018, .3 / max(1, len(labels)))
            scale = (1 - gap * max(0, len(labels) - 1)) / len(docs)
            y, result = 1., {}
            for label in labels:
                h = totals[label] * scale
                result[label] = [y - h, h, 0., scale]
                ax.barh(y - h / 2, .025, height=h, left=0 if side == 0 else .975, color='#718b9d')
                ax.text(-.015 if side == 0 else 1.015, y - h / 2, f'{label} ({totals[label]:,})',
                        ha='right' if side == 0 else 'left', va='center', fontsize=9)
                y -= h + gap
            return result
        lnodes, rnodes = layout(left, 0), layout(right, 1)
        for (a, b), count in flows.items():
            ly, _, loffset, lscale = lnodes[a]
            ry, _, roffset, rscale = rnodes[b]
            y0, y1 = ly + loffset, ry + roffset
            h0, h1 = count * lscale, count * rscale
            vertices = [(0.025, y0), (.4, y0), (.6, y1), (.975, y1), (.975, y1 + h1),
                        (.6, y1 + h1), (.4, y0 + h0), (.025, y0 + h0), (.025, y0)]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
                     MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(vertices, codes), facecolor=PALETTE[right.index(b) % len(PALETTE)], alpha=.4, edgecolor='none'))
            lnodes[a][2] += h0
            rnodes[b][2] += h1
        ax.set(xlim=(-.25, 1.45), ylim=(-.02, 1.02), title=('数据库' if by_database else '时期') + '到主题的构成（流量=记录数；不表示迁移）')
        ax.axis('off')
        self.emit(15, figure, chart, {'flows': [{'period': a, 'topic_group': b, 'count': n} for (a, b), n in flows.items()],
                                    'included_records': len(docs), 'excluded_records': len(self.docs) - len(docs),
                                    'dimension': 'database' if by_database else 'period',
                                    'interpretation': 'cross-sectional composition; not longitudinal theme migration'})

    def details(self, check_cancel):
        from wordcloud import WordCloud
        selected = self.summary[:self.config.report.topic_details]
        entries = []
        for row in selected:
            check_cancel()
            t = row['topic']
            items = self.words(t, 12)[::-1]
            figure, ax = plt.subplots(figsize=(10, 6))
            ax.barh([r['keyword'] for r in items], [r['ctfidf_weight'] for r in items], color=color(t))
            ax.set(title=wrapped(self.name(t), 50), xlabel='c-TF-IDF 权重')
            files = self.save(figure, f'topic_{t:04d}_keywords')
            frequencies = Counter(token for r in self.docs if r['topic'] == t for token in self.tokenizer(r['text']))
            if frequencies:
                cloud = WordCloud(font_path=self.font, width=1000, height=600, background_color='white',
                                  random_state=self.config.model.seed, collocations=False).generate_from_frequencies(frequencies)
                figure, ax = plt.subplots(figsize=(11, 7))
                ax.imshow(cloud)
                ax.set_title(f'T{t}：真实分词出现次数词云')
                ax.axis('off')
                files.extend(self.save(figure, f'topic_{t:04d}_word_frequency'))
                write_json(self.directory / 'evidence' / f'topic_{t:04d}_word_frequency.json', frequencies.most_common(200))
            entries.append({'topic': t, 'files': files})
        return entries

    def gallery(self, details):
        cards = []
        for item in self.catalog:
            links = ' · '.join(f'<a href="{escape(p, quote=True)}">{escape(Path(p).suffix.lstrip("."))}</a>' for p in item['files'])
            image = next((p for p in item['files'] if p.endswith('.png')), None)
            cards.append(f'<article><h2>{item["number"]}. {escape(item["title"])}</h2>'
                         + (f'<a href="{image}"><img loading="lazy" src="{image}" alt="{escape(item["title"])}"></a>' if image else '')
                         + f'<p>{escape(item["reason"] or "")}</p><p>{links}</p></article>')
        rows = ''.join('<tr><td>T' + str(item['topic']) + '</td><td>' + escape(self.labels[item['topic']]) + '</td><td>'
                       + ' · '.join(f'<a href="{p}">{escape(Path(p).name)}</a>' for p in item['files']) + '</td></tr>' for item in details)
        group_links = ''.join(f'<li><a href="interactive/{p.name}">{escape(p.stem)}</a></li>'
                              for p in sorted((self.directory / 'interactive').glob('fig09_*.html')))
        html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(self.config.title)}</title><style>body{{font:16px system-ui;max-width:1440px;margin:auto;padding:24px;color:#173242;background:#f4f7f9}}article,section{{background:white;padding:22px;margin:20px 0;border-radius:12px}}img{{max-width:100%;height:auto}}a{{color:#176b87}}td{{padding:10px;border-bottom:1px solid #ddd}}table{{width:100%}}.scroll{{overflow:auto}}</style>
<h1>{escape(self.config.title)}</h1><p>{len(self.docs):,} 条有效记录 · {len(self.ids)} 个非离群主题 · {self.metrics['outliers']:,} 条离群</p>
<p>{escape(self.note)}</p><section><h2>数据口径与诊断</h2><p>轮廓系数={self.metrics['silhouette_cosine']}；NPMI={self.metrics['npmi_mean']}。指标用于诊断；自动关键词标签未经人工评审。</p><p>缺失年份不进入时间图。国家与机构使用原始字段或保守的地址解析，逐条记录 metadata_origin；不识别的国家留空。分类仅使用原始导出值，未用模型补造；多组记录在每组各计一次，机构未实体消歧。</p><p>词云总览采用 c-TF-IDF 权重，独立词频词云采用真实分词出现次数；词云和文档散点的 SVG/PDF 可以包含栅格层。</p><p>主题模型的安全导出使用 safetensors；不包含原 UMAP/HDBSCAN 推断器，载入后的中心相似度分配与原密度聚类不同。</p><p>图的数值证据见各 JSON 链接；不可计算的图说明原因。</p></section>
{''.join(cards)}<section><h2>分组交互图</h2><ul>{group_links}</ul></section><section><h2>独立主题图</h2><p>显示 {len(details)}/{len(self.ids)} 个主题，可通过 topic_details 调整。</p><div class="scroll"><table>{rows}</table></div></section></html>'''
        (self.directory / 'report.html').write_text(html, encoding='utf-8')


def render(config, ingest_dir, model_dir, destination, check_cancel=lambda: None, progress=lambda **kwargs: None):
    renderer = Renderer(config, ingest_dir, model_dir, destination)
    methods = [renderer.overview, renderer.keyword_bars, renderer.clouds, renderer.hierarchy,
               renderer.document_map, renderer.keyword_heatmap, renderer.ranks, renderer.trends,
               renderer.groups, renderer.similarities, renderer.network, renderer.sizes,
               renderer.outliers, renderer.quality, renderer.sankey]
    for index, method in enumerate(methods, 1):
        check_cancel()
        method()
        progress(completed=index, total=15, chart=index)
    details = renderer.details(check_cancel)
    renderer.gallery(details)
    write_json(Path(destination) / 'figures_catalog.json', {'charts': renderer.catalog, 'topic_details': details,
                'font_family': renderer.family, 'dpi': config.report.dpi,
                'generated': sum(r['status'] == 'generated' for r in renderer.catalog),
                'not_applicable': sum(r['status'] == 'not_applicable' for r in renderer.catalog)})
    return {'charts': 15, 'generated': sum(r['status'] == 'generated' for r in renderer.catalog),
            'topic_details': len(details), 'report': 'report.html'}
