"""Dashboard of the undersampling experiment. Launch from the undersampling folder:

    streamlit run dashboard/app.py
    (or ./dashboard/run_dashboard.sh)

Reads results/<name>/ (summary.json and the CSVs written by main.py). On a cloud VM, open it through
an SSH tunnel: ssh -L 8501:localhost:8501 <vm>, then browse http://localhost:8501.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / 'results'

# Categorical slots 1-2 of the validated reference palette (light / dark mode).
COLORS = {'baseline': '#2a78d6', 'undersampling': '#eb6834'}
TIME_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']   # slots 1, 2, 3, 7: adjacent-pair safe
GRID = 'rgba(128,128,128,0.18)'
LABELS = {'baseline': 'Baseline', 'undersampling': 'Undersampling'}

st.set_page_config(page_title='Undersampling experiment', layout='wide')

# Full-width charts and tables: Streamlit renamed this option in 1.46 ('use_container_width' -> 'width').
_ST_VERSION = tuple(int(x) for x in st.__version__.split('.')[:2])
LOG_R_TICKS = {'tickvals': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
               'ticktext': ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1']}
FULL = {'width': 'stretch'} if _ST_VERSION >= (1, 46) else {'use_container_width': True}


# ---------- data ----------
def experiments():
    return sorted(p.name for p in RESULTS.iterdir() if (p / 'summary.json').exists()) if RESULTS.exists() else []


@st.cache_data(show_spinner=False)
def load(name, mtime):
    folder = RESULTS / name
    read = lambda f: pd.read_csv(folder / f) if (folder / f).exists() else pd.DataFrame()  # noqa: E731
    return (json.loads((folder / 'summary.json').read_text()), read('runs_top.csv'), read('runs_curve.csv'),
            read('trials_baseline.csv'), read('trials_undersampling.csv'))


def millions(x):
    return f'{x / 1e6:,.2f}M'


def style(fig, title, x_title, y_title, height=360):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation='h', y=1.02, x=1, xanchor='right', yanchor='bottom', title=None),
                      hovermode='closest', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, showline=False)
    return fig


# ---------- page ----------
names = experiments()
if not names:
    st.error(f'No results found in {RESULTS}. Run main.py first.')
    st.stop()

with st.sidebar:
    st.header('Experiment')
    name = st.selectbox('Results folder', names, index=names.index('full') if 'full' in names else 0)
    st.caption('Rerun `python main.py --stages analyze` to refresh a summary.')

summary, runs, curve, trials_b, trials_u = load(name, (RESULTS / name / 'summary.json').stat().st_mtime)
cfg = summary['config']

st.title('Undersampling for churn prediction')
st.markdown('> **If each tree samples only a fraction r of the CONTINUA, the model reaches about the same AUC and '
            'profit as the baseline trained on all the data, in clearly less time. But below some r the time stops '
            'going down, so making r smaller after that point only risks the result.**')
st.caption(f"Experiment **{name}** · train {cfg['train_months'][0]}–{cfg['train_months'][-1]} · validation "
           f"{cfg['valid_month']} · test {cfg['test_month']} · clients {cfg['sample_clients']:.0%} · "
           f"bagging switch {'on' if cfg['bagging_baseline'] else 'off'} · {cfg['n_trials']} trials · "
           f"top {cfg['top_k']} × {cfg['n_seeds']} seeds")

# ---------- verdict ----------
verdict = summary.get('verdict', {})
if verdict:
    st.subheader('Verdict')
    labels = {'profit_about_the_same': 'Profit about the same (≥ −δ)', 'time_clearly_less': 'Time clearly less',
              'time_flattens': 'Time flattens', 'useful_zone': 'Useful zone exists',
              'hypothesis_confirmed': 'Hypothesis confirmed'}
    cols = st.columns(len(verdict))
    for col, (key, value) in zip(cols, verdict.items()):
        col.metric(labels.get(key, key), '✅ Yes' if value else '❌ No')

    tiles = st.columns(4)
    if 'searches' in summary:
        p = summary['searches']['profit_band']
        tiles[0].metric('Band profit: mean difference', millions(p['mean_diff']),
                        f"95% interval [{millions(p['ci_low'])}, {millions(p['ci_high'])}]", delta_color='off')
        tiles[1].metric('Wilcoxon p-value (band profit)', f"{p['wilcoxon_p']:.3g}",
                        f"{p['n_pairs']} paired runs", delta_color='off')
        t = summary['time']
        tiles[2].metric('Total time reduction', f"{t['reduction']:.1%}",
                        f"required {cfg['time_reduction']:.0%} · search only {t['search_only_reduction']:.1%}",
                        delta_color='off')
    if 'curve' in summary:
        c = summary['curve']
        zone = f"r ∈ [{c['useful_zone_range'][0]:g}, {c['useful_zone_range'][1]:g}]" if c['useful_zone'] else 'none'
        tiles[3].metric('Useful zone', zone, f"r_g = {c['r_g']} · r_t = {c['r_t']}", delta_color='off')

tab_cmp, tab_curve, tab_time, tab_trials, tab_info = st.tabs(
    ['Baseline vs undersampling', 'r curve', 'Time', 'Optuna trials', 'Run info'])

# ---------- comparison ----------
with tab_cmp:
    if runs.empty:
        st.info('No seed runs yet (stage "seeds").')
    else:
        metric = st.radio('Metric (test month)', ['profit_band', 'profit_peak', 'auc'], horizontal=True,
                          format_func={'profit_band': 'Band profit (5–9%)', 'profit_peak': 'Highest point',
                                       'auc': 'AUC'}.get)
        left, right = st.columns(2)
        fig = go.Figure()
        for search in ['baseline', 'undersampling']:
            d = runs[runs['search'] == search]
            fig.add_trace(go.Box(x=d['rank'].map(lambda k: f'#{k}'), y=d[metric], name=LABELS[search],
                                 marker=dict(color=COLORS[search], size=8, line=dict(width=1, color='white')),
                                 line=dict(width=2), boxpoints='all', jitter=0.35, pointpos=0,
                                 hovertemplate='%{x} · %{y:,.4~f}<extra>' + LABELS[search] + '</extra>'))
        fig.update_layout(boxmode='group')
        left.plotly_chart(style(fig, 'Distribution over seeds, by configuration rank', 'Configuration rank',
                                metric), **FULL)

        b = runs[runs['search'] == 'baseline'].set_index(['rank', 'seed'])[metric]
        u = runs[runs['search'] == 'undersampling'].set_index(['rank', 'seed'])[metric]
        idx = b.index.intersection(u.index)
        diffs = (u.loc[idx] - b.loc[idx]).reset_index(name='diff')
        comp = summary['searches'][metric]
        fig = go.Figure()
        fig.add_vrect(x0=-comp['margin'], x1=comp['margin'], fillcolor='rgba(128,128,128,0.12)', line_width=0,
                      annotation_text='± δ', annotation_position='top left')
        fig.add_vline(x=0, line=dict(color='rgba(128,128,128,0.6)', width=1))
        fig.add_trace(go.Scatter(x=diffs['diff'], y=diffs['rank'].map(lambda k: f'#{k}'), mode='markers',
                                 marker=dict(color=COLORS['undersampling'], size=9, opacity=0.75,
                                             line=dict(width=2, color='white')),
                                 customdata=diffs['seed'], name='Seed pair',
                                 hovertemplate='rank %{y} · seed %{customdata}<br>diff %{x:,.4~f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=[comp['ci_low'], comp['ci_high']], y=['mean', 'mean'], mode='lines',
                                 line=dict(color='#0b0b0b', width=2), name='95% interval',
                                 hovertemplate='%{x:,.4~f}<extra>95% interval</extra>'))
        fig.add_trace(go.Scatter(x=[comp['mean_diff']], y=['mean'], mode='markers', name='Mean difference',
                                 marker=dict(color='#0b0b0b', size=11, symbol='diamond'),
                                 hovertemplate='%{x:,.4~f}<extra>Mean difference</extra>'))
        right.plotly_chart(style(fig, 'Paired differences (undersampling − baseline)', f'Difference in {metric}',
                                 'Rank'), **FULL)

        st.markdown('**Per configuration**')
        st.dataframe(pd.DataFrame(summary['per_config']), **FULL, hide_index=True)

# ---------- r curve ----------
with tab_curve:
    if curve.empty:
        st.info('No curve runs yet (stage "curve").')
    else:
        c = summary['curve']
        table = pd.DataFrame(c['table']).sort_values('r')
        agg = curve.groupby('r').agg(p_lo=('profit_band', 'min'), p_hi=('profit_band', 'max'),
                                     t_lo=('train_seconds', 'min'), t_hi=('train_seconds', 'max')).reset_index()
        table = table.merge(agg, on='r')
        left, right = st.columns(2)
        for col, (y, lo, hi, title, y_title, color) in zip([left, right], [
                ('profit_band_mean', 'p_lo', 'p_hi', 'Band profit on test vs r', 'Band profit', COLORS['baseline']),
                ('train_seconds_mean', 't_lo', 't_hi', 'Training time vs r', 'Seconds per training',
                 COLORS['undersampling'])]):
            fig = go.Figure()
            if c['useful_zone']:
                fig.add_vrect(x0=c['useful_zone_range'][0], x1=c['useful_zone_range'][1],
                              fillcolor='rgba(27,175,122,0.12)', line_width=0, annotation_text='useful zone',
                              annotation_position='top left')
            fig.add_trace(go.Scatter(x=pd.concat([table['r'], table['r'][::-1]]),
                                     y=pd.concat([table[hi], table[lo][::-1]]), fill='toself',
                                     fillcolor='rgba(128,128,128,0.15)', line=dict(width=0), hoverinfo='skip',
                                     name='Seed range (min–max)'))
            fig.add_trace(go.Scatter(x=table['r'], y=table[y], mode='lines+markers', name='Mean over seeds',
                                     line=dict(color=color, width=2), marker=dict(size=8, line=dict(width=2, color='white')),
                                     hovertemplate='r = %{x:g}<br>%{y:,.4~f}<extra></extra>'))
            fig.update_xaxes(type='log', autorange='reversed', **LOG_R_TICKS)
            col.plotly_chart(style(fig, title, 'r (share of CONTINUA per tree, log scale)', y_title),
                             **FULL)
        st.caption('x axis runs from r = 1 (the baseline) towards smaller r. '
                   f"r_g = {c['r_g']} (profit holds down to here) · r_t = {c['r_t']} (time flattens here).")
        flat = table[table['auc_mean'] <= 0.5 + 1e-9]['r'].tolist()
        if flat:
            st.warning(f"At r = {', '.join(f'{r:g}' for r in flat)} the model could not make a single split "
                       '(too few rows per tree for min_data_in_leaf): every client gets the same score, AUC is 0.5, '
                       'and the profit shown is that of an arbitrary ordering.')
        st.dataframe(pd.DataFrame(c['table']), **FULL, hide_index=True)

# ---------- time ----------
with tab_time:
    if 'time' not in summary:
        st.info('No time comparison yet.')
    else:
        t = summary['time']
        parts = [('prep_seconds', 'Data preparation'), ('search_seconds', 'Optuna search'),
                 ('final_train_seconds', 'Final model'), ('predict_seconds', 'Prediction')]
        fig = go.Figure()
        for (key, label), color in zip(parts, TIME_COLORS):
            fig.add_trace(go.Bar(y=[LABELS[s] for s in ['baseline', 'undersampling']],
                                 x=[t[s][key] / 60 for s in ['baseline', 'undersampling']], name=label,
                                 orientation='h', marker=dict(color=color, line=dict(width=2, color='white')),
                                 hovertemplate='%{y} · ' + label + ': %{x:,.1f} min<extra></extra>'))
        fig.update_layout(barmode='stack')
        st.plotly_chart(style(fig, f"Total time (reduction {t['reduction']:.1%})", 'Minutes', None, height=280),
                        **FULL)
        m = st.columns(3)
        m[0].metric('Total time reduction', f"{t['reduction']:.1%}", f"required {cfg['time_reduction']:.0%}",
                    delta_color='off')
        m[1].metric('Search-only reduction', f"{t['search_only_reduction']:.1%}")
        m[2].metric('Time per tree reduction', f"{t['per_tree_reduction']:.1%}",
                    f"{t['baseline']['seconds_per_tree']:.4f} s → {t['undersampling']['seconds_per_tree']:.4f} s",
                    delta_color='off')
        st.caption('Total time depends on how many trees each configuration grows (learning rate and early stopping). '
                   'Time per tree removes that effect and isolates what r saves.')
        trial_time = pd.concat([trials_b.assign(search='baseline'), trials_u.assign(search='undersampling')])
        if not trial_time.empty:
            left, right = st.columns(2)
            fig = go.Figure()
            for search in ['baseline', 'undersampling']:
                d = trial_time[trial_time['search'] == search]
                fig.add_trace(go.Scatter(x=d['trial'], y=d['trial_seconds'], mode='markers', name=LABELS[search],
                                         marker=dict(color=COLORS[search], size=8, line=dict(width=2, color='white')),
                                         customdata=np.stack([d['r'], d['trees_built']], axis=1),
                                         hovertemplate='trial %{x}: %{y:,.1f} s<br>r = %{customdata[0]:.3f} · '
                                                       '%{customdata[1]} trees<extra>' + LABELS[search] + '</extra>'))
            left.plotly_chart(style(fig, 'Time per trial', 'Trial', 'Seconds'), **FULL)
            fig = go.Figure()
            for search in ['baseline', 'undersampling']:
                d = trial_time[trial_time['search'] == search]
                fig.add_trace(go.Scatter(x=d['r'], y=d['seconds_per_tree'], mode='markers', name=LABELS[search],
                                         marker=dict(color=COLORS[search], size=8, line=dict(width=2, color='white')),
                                         customdata=d['trial'],
                                         hovertemplate='trial %{customdata}: r = %{x:.3f}<br>%{y:.4f} s per tree'
                                                       '<extra>' + LABELS[search] + '</extra>'))
            fig.update_xaxes(type='log', **LOG_R_TICKS)
            right.plotly_chart(style(fig, 'Time per tree vs r (every trial)', 'r (log scale)', 'Seconds per tree'),
                               **FULL)

# ---------- trials ----------
with tab_trials:
    trials = pd.concat([trials_b.assign(search='baseline'), trials_u.assign(search='undersampling')])
    if trials.empty:
        st.info('No trials yet (stage "search").')
    else:
        left, right = st.columns(2)
        fig = go.Figure()
        for search in ['baseline', 'undersampling']:
            d = trials[trials['search'] == search].sort_values('trial')
            fig.add_trace(go.Scatter(x=d['trial'], y=d['valid_auc_score'], mode='markers', name=LABELS[search],
                                     marker=dict(color=COLORS[search], size=8, opacity=0.6,
                                                 line=dict(width=2, color='white')),
                                     hovertemplate='trial %{x}: AUC %{y:.5f}<extra>' + LABELS[search] + '</extra>'))
            fig.add_trace(go.Scatter(x=d['trial'], y=d['valid_auc_score'].cummax(), mode='lines',
                                     line=dict(color=COLORS[search], width=2), name=f'{LABELS[search]} best so far',
                                     hoverinfo='skip'))
        left.plotly_chart(style(fig, 'Validation AUC by trial', 'Trial', 'Validation AUC'), **FULL)
        d = trials[trials['search'] == 'undersampling']
        if not d.empty:
            fig = go.Figure(go.Scatter(x=d['r'], y=d['valid_auc_score'], mode='markers', name='Undersampling',
                                       marker=dict(color=COLORS['undersampling'], size=9,
                                                   line=dict(width=2, color='white')),
                                       customdata=d['trial'],
                                       hovertemplate='trial %{customdata}: r = %{x:.3f}<br>AUC %{y:.5f}<extra></extra>'))
            fig.update_xaxes(type='log', **LOG_R_TICKS)
            right.plotly_chart(style(fig, 'Undersampling search: r tried vs validation AUC', 'r (log scale)',
                                     'Validation AUC'), **FULL)
        st.dataframe(trials, **FULL, hide_index=True)

# ---------- info ----------
with tab_info:
    env = summary.get('environment', {})
    timings = summary.get('timings', {})
    st.markdown(f"**Machine:** {env.get('platform')} · {env.get('cpu_count')} cores · {env.get('ram_gb')} GB RAM  \n"
                f"**Peak memory of the run:** {timings.get('peak_memory_gb', float('nan')):.2f} GB  \n"
                f"**Features:** {timings.get('n_features')} columns")
    with st.expander('Configuration'):
        st.json(cfg)
    with st.expander('Environment'):
        st.json(env)
