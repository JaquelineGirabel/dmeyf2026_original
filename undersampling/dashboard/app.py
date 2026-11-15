"""Tablero del experimento de submuestreo (undersampling). Se lanza desde la carpeta undersampling:

    ./dashboard/run_dashboard.sh        (o: streamlit run dashboard/app.py)

Solo lee results/: results/<base>/<modo>_<r_apply>/summary.json, results/<base>/runs.csv y
results/<base>/trials/*.csv. No necesita el dataset ni los modelos. Los colores de fondo están en
.streamlit/config.toml. La variable UNDERSAMPLING_RESULTS permite apuntar a otra carpeta de resultados.

Regla de color: el modelo base es siempre azul marino; el submuestreo toma el color de acento de cada sección
(todas las parejas validadas para daltonismo y visión normal sobre fondo blanco).
"""
import html
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

RESULTS = Path(os.environ.get('UNDERSAMPLING_RESULTS', Path(__file__).resolve().parents[1] / 'results'))

# ---------- paleta ----------
NAVY, NAVY_DARK = '#1f3a5f', '#14213d'          # modelo base · encabezado
INK, INK_2, MUTED = '#1b2430', '#4a5466', '#6b7587'
GRID, AXIS, BORDER = '#eef0f3', '#cfd4db', '#e1e4e9'
GOLD = '#e0b04a'
ACCENT = {'eleccion': '#eb6834', 'ganancia': '#1baf7a', 'auc': '#6250d6', 'tiempo': '#d4911c', 'busquedas': '#2a78d6',
          'monitoreo': '#e87ba4'}
MON_DARK = '#b8457a'                                # texto y bordes sobre el acento de monitoreo
TIME_PARTS = ['#a3b4c7', '#7890ab', '#4d6886', '#1f3c5c']    # preparación → búsqueda → entrenamiento → predicción
GOOD, BAD, MID = '#1e6b52', '#b42318', '#a15c07'
BAND = 'rgba(107,117,135,0.13)'
ZONE = 'rgba(30,107,82,0.10)'


def rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})'


st.set_page_config(page_title='Submuestreo · modelo de bajas', page_icon='📉', layout='wide')
_ST_VERSION = tuple(int(x) for x in st.__version__.split('.')[:2])
FULL = {'width': 'stretch'} if _ST_VERSION >= (1, 46) else {'use_container_width': True}

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.4rem; max-width: 1280px;}}
  header[data-testid="stHeader"] {{background: transparent;}}
  .top {{background: {NAVY_DARK}; color: #fff; border-radius: 8px; padding: 26px 32px 22px; margin-bottom: 18px;}}
  .kicker {{font-size: 11.5px; letter-spacing: .16em; text-transform: uppercase; font-weight: 650; color: #8fa0bd;}}
  .top h1 {{color: #fff; font-size: 30px; font-weight: 700; letter-spacing: -.01em; margin: 6px 0 6px; padding: 0;}}
  .top .sub {{color: #c9d2e0; font-size: 14.5px; margin: 0; max-width: 980px; line-height: 1.55;}}
  .top .sub b {{color: #fff;}}
  .stTabs [data-baseweb="tab-list"] {{gap: 4px; border-bottom: 1px solid #d9dde3;}}
  .stTabs [data-baseweb="tab"] {{font-size: 14px; color: {INK_2}; padding: 8px 14px;}}
  .stTabs [aria-selected="true"] {{color: {NAVY_DARK}; font-weight: 650;}}
  .stTabs [data-baseweb="tab-highlight"] {{background-color: {GOLD}; height: 3px;}}
  .page-h {{margin: 6px 0 18px; padding-bottom: 12px; border-bottom: 1px solid #d9dde3;}}
  .page-h .kicker {{color: {MUTED};}}
  .page-h h2 {{font-size: 24px; margin: 4px 0 0; color: {NAVY_DARK}; letter-spacing: -.01em; padding: 0;}}
  .page-h p {{margin: 6px 0 0; color: {INK_2}; font-size: 15px; max-width: 940px; line-height: 1.55;}}
  .part {{margin: 34px 0 4px; padding: 2px 0 2px 12px; border-left: 4px solid var(--c);}}
  .part h3 {{font-size: 20px; margin: 0; color: {NAVY_DARK}; padding: 0; letter-spacing: -.01em;}}
  .part p {{margin: 4px 0 0; color: {INK_2}; font-size: 14.5px; max-width: 940px; line-height: 1.55;}}
  .sec {{display: flex; align-items: center; gap: 10px; font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
        color: {MUTED}; font-weight: 700; margin: 26px 0 10px; padding-bottom: 6px; border-bottom: 1px solid {BORDER};}}
  .sec .dot {{width: 10px; height: 10px; border-radius: 2px; display: inline-block;}}
  .sec-text {{color: {INK_2}; font-size: 14.5px; margin: -2px 0 12px; max-width: 940px; line-height: 1.55;}}
  div[data-testid="stPlotlyChart"] {{background: #fff; border: 1px solid {BORDER}; border-radius: 6px; padding: 8px 10px;}}
  .kpis {{display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: 14px; margin: 4px 0 6px;}}
  .kpi {{background: #fff; border: 1px solid {BORDER}; border-top: 3px solid var(--c); border-radius: 6px; padding: 14px 18px;}}
  .kpi .l {{font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; font-weight: 700;}}
  .kpi .v {{font-size: 28px; font-weight: 700; color: {NAVY_DARK}; margin: 4px 0 2px; font-variant-numeric: tabular-nums;}}
  .kpi .s {{font-size: 13px; color: {INK_2}; line-height: 1.45;}}
  .ok {{color: {GOOD}; font-weight: 700;}} .ko {{color: {BAD}; font-weight: 700;}} .na {{color: {MUTED}; font-weight: 700;}}
  .concl {{margin: 10px 0 4px; padding: 10px 14px; border-radius: 4px; border-left: 4px solid; font-size: 14px;
          font-weight: 600; color: {NAVY_DARK}; line-height: 1.5;}}
  .concl.good {{background: #eef7f2; border-color: {GOOD};}} .concl.bad {{background: #fdf3f2; border-color: {BAD};}}
  .concl.mid {{background: #fffbf2; border-color: {MID};}} .concl.info {{background: #f7f9fc; border-color: {NAVY};}}
  .cap {{color: {MUTED}; font-size: 12.8px; margin: 6px 2px 0; line-height: 1.5;}}
  .card {{background: #fff; border: 1px solid {BORDER}; border-radius: 6px; padding: 18px 22px; margin: 0 0 14px;}}
  .card p, .card li {{font-size: 14.5px; color: #2a3343; line-height: 1.55;}}
  .card h3 {{font-size: 17px; margin: 0 0 8px; color: {NAVY_DARK}; padding: 0;}}
  .checks {{display: grid; gap: 0;}}
  .check {{display: grid; grid-template-columns: 110px 1fr; gap: 10px; padding: 9px 0; border-bottom: 1px solid {GRID};}}
  .check:last-child {{border-bottom: none;}}
  .check .t {{color: {NAVY_DARK}; font-weight: 600; font-size: 14.3px;}}
  .check .w {{color: {MUTED}; font-size: 13px;}}
  .verdict {{display: flex; align-items: center; gap: 14px; border-radius: 6px; padding: 14px 20px; margin: 0 0 14px;
            font-size: 17px; font-weight: 700; border: 1px solid;}}
  .verdict.good {{background: #eef7f2; border-color: #b7dcc8; color: {GOOD};}}
  .verdict.bad {{background: #fdf3f2; border-color: #f0c4be; color: {BAD};}}
  .verdict.na {{background: #fff; border-color: {BORDER}; color: {INK_2};}}
  .verdict .pill {{font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #fff; border-radius: 10px;
                  padding: 3px 10px;}}
  table.t {{width: 100%; border-collapse: collapse; font-size: 13.5px; background: #fff; border: 1px solid {BORDER};
           border-radius: 6px; overflow: hidden; font-variant-numeric: tabular-nums;}}
  table.t thead th {{font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; font-weight: 700;
                    text-align: right; padding: 9px 12px; border-bottom: 1px solid #cfd4db; background: #fafbfc;}}
  table.t td {{padding: 8px 12px; border-bottom: 1px solid {GRID}; color: #2a3343; text-align: right;}}
  table.t th:first-child, table.t td:first-child {{text-align: left;}}
  table.t.txt th, table.t.txt td {{text-align: left;}}
  table.t.txt td:first-child {{width: 32%; color: {INK_2};}}
  table.t tr.hl td {{background: #fff8ec; font-weight: 650; color: {NAVY_DARK};}}
  .alerts {{background: #fff; border: 1px solid {BORDER}; border-radius: 6px; overflow: hidden; margin-bottom: 6px;}}
  .alert {{display: grid; grid-template-columns: 150px 90px 1fr; gap: 14px; padding: 11px 18px; border-left: 4px solid var(--c);
          border-bottom: 1px solid {GRID};}}
  .alert:last-child {{border-bottom: none;}}
  .alert .lv {{font-weight: 700; font-size: 13.5px;}}
  .alert .ar {{font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; font-weight: 700; padding-top: 2px;}}
  .alert .t {{color: {NAVY_DARK}; font-weight: 650; font-size: 14.3px;}}
  .alert .w {{color: {INK_2}; font-size: 13.3px; line-height: 1.5; margin-top: 2px;}}
  .scale {{display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; margin: 2px 0 16px; border: 1px solid {BORDER};
          border-radius: 6px; overflow: hidden; background: #fff;}}
  .scale div {{padding: 12px 16px; font-size: 13.8px; color: {INK_2}; border-top: 6px solid var(--c);}}
  .scale div:first-child {{border-top-color: #e3e7ee;}}
  .scale b {{display: block; color: {NAVY_DARK}; font-size: 15px; margin-bottom: 2px;}}
  .scale .sw2 {{display: none;}}
  .legend {{display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: {INK_2}; margin: 2px 0 10px;}}
  .sw {{display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 6px; vertical-align: -1px;}}
</style>
""", unsafe_allow_html=True)


# ---------- formato en español ----------
def num(x, d=1):
    """1234567.891 -> '1.234.567,9'"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    return f'{x:,.{d}f}'.replace(',', '§').replace('.', ',').replace('§', '.')


def pesos(x, d=1):
    return f'$ {num(x / 1e6, d)} M'


def pct(x, d=1, sign=False):
    s = num(abs(x) * 100, d) + ' %'
    return ('−' if x < 0 else '+' if sign and x > 0 else '') + s


def signed(x, d=1):
    return ('−' if x < 0 else '+') + num(abs(x), d)


def r_txt(r):
    return '—' if r is None else num(r, 3).rstrip('0').rstrip(',')


def dur(seconds):
    if seconds >= 3600:
        return f'{num(seconds / 3600, 1)} h'
    if seconds >= 60:
        return f'{num(seconds / 60, 1)} min'
    return f'{num(seconds, 1)} s'


MONTHS = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre',
          'noviembre', 'diciembre']


def month(yyyymm):
    return f'{MONTHS[yyyymm % 100 - 1]} de {yyyymm // 100}'


def bold(text):
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)


BASES = {'full': 'Datos completos', 'sample010': 'Muestra del 10 % de los clientes'}
MODES = {'grid': 'r de una lista fija', 'optuna': 'r elegido por Optuna'}
MODES_LONG = {'grid': 'r de una lista fija: una búsqueda de hiperparámetros por cada valor',
              'optuna': 'r elegido por Optuna, junto con los demás hiperparámetros'}
APPLY = {'per_tree': 'r aplicado en cada árbol', 'before': 'r aplicado una vez, antes de entrenar'}
R_TICKS = dict(tickvals=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
               ticktext=['0,01', '0,02', '0,05', '0,1', '0,2', '0,5', '1'])


def experiment_label(key):
    base, exp = key.split('/')
    if base == '_old_design':
        return f'Diseño anterior · {BASES.get(exp, exp)}'
    mode, apply = exp.split('_', 1)
    return f'{BASES.get(base, base)} · {MODES.get(mode, mode)} · {APPLY.get(apply, apply)}'


def study_label(study):
    if study == 'baseline':
        return 'Modelo base (r = 1)'
    apply = 'en cada árbol' if 'per_tree' in study else 'antes de entrenar'
    if study.startswith('grid_'):
        return f"r = {r_txt(float(study.split('_r')[-1]))} · aplicado {apply}"
    return f'r elegido por Optuna · aplicado {apply}'


# ---------- piezas de la página ----------
def page_head(kicker, title, text):
    st.markdown(f'<div class="page-h"><div class="kicker">{kicker}</div><h2>{title}</h2><p>{bold(text)}</p></div>',
                unsafe_allow_html=True)


def block(title, color, text=None):
    """Bloque grande dentro de una pestaña (agrupa varias secciones)."""
    st.markdown(f'<div class="part" style="--c:{color}"><h3>{title}</h3>' + (f'<p>{bold(text)}</p>' if text else '')
                + '</div>', unsafe_allow_html=True)


def section(title, color=None, text=None):
    dot = f'<span class="dot" style="background:{color}"></span>' if color else ''
    st.markdown(f'<div class="sec">{dot}{title}</div>' + (f'<div class="sec-text">{bold(text)}</div>' if text else ''),
                unsafe_allow_html=True)


def kpis(items, color):
    cells = ''.join(f'<div class="kpi" style="--c:{c or color}"><div class="l">{l}</div><div class="v">{v}</div>'
                    f'<div class="s">{s}</div></div>' for l, v, s, c in items)
    st.markdown(f'<div class="kpis" style="--n:{len(items)}">{cells}</div>', unsafe_allow_html=True)


def concl(kind, text):
    st.markdown(f'<div class="concl {kind}">{bold(text)}</div>', unsafe_allow_html=True)


def caption(text):
    st.markdown(f'<div class="cap"><b>Lectura:</b> {bold(text)}</div>', unsafe_allow_html=True)


def legend(items):
    st.markdown('<div class="legend">' + ''.join(f'<span><span class="sw" style="background:{c}"></span>{l}</span>'
                                                 for l, c in items) + '</div>', unsafe_allow_html=True)


def status(ok, yes='Cumple', no='No cumple'):
    if ok is None:
        return '<span class="na">— Sin dato</span>'
    return f'<span class="ok">✔ {yes}</span>' if ok else f'<span class="ko">✖ {no}</span>'


def table(df, highlight=None, text=False):
    """text=True: tabla de texto (Dato / Valor), todo alineado a la izquierda."""
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns)
    rows = ''
    for i, (_, row) in enumerate(df.iterrows()):
        cls = ' class="hl"' if highlight is not None and highlight[i] else ''
        rows += f'<tr{cls}>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>'
    st.markdown(f'<table class="t{" txt" if text else ""}"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True)


# ---------- gráficos ----------
def style(fig, title, x_title, y_title, height=380, legend_on=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=NAVY_DARK), x=0.01, xanchor='left'),
        height=height, margin=dict(l=12, r=28, t=54, b=12), separators=',.',
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', color=INK_2, size=12),
        showlegend=legend_on, legend=dict(orientation='h', y=-max(0.2, 80 / height), x=0, xanchor='left', yanchor='top',
                                          title=None,
                                          font=dict(color=INK_2, size=12)),
        hoverlabel=dict(bgcolor='#fff', bordercolor=BORDER, font=dict(color=INK, size=12)),
        hovermode='closest', plot_bgcolor='#fff', paper_bgcolor='#fff')
    fig.update_xaxes(title=x_title, gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2, size=12))
    fig.update_yaxes(title=y_title, gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2, size=12))
    return fig


def r_axis(fig, short=False):
    title = 'r (escala logarítmica)' if short else 'r: fracción de CONTINUA que usa cada árbol (escala logarítmica)'
    fig.update_xaxes(type='log', autorange=False, range=[np.log10(1.35), np.log10(0.0075)], **R_TICKS,
                     title=dict(text=title, font=dict(color=INK_2, size=12)))
    return fig


def mark_r(fig, r, text, color, top=False, dash='dot', width=1.2):
    """Vertical line at r with its label next to it. (On log axes Plotly places annotations in log10 units.)"""
    fig.add_vline(x=r, line=dict(color=color, width=width, dash=dash))
    fig.add_annotation(x=np.log10(r), y=0.98 if top else 0.02, xref='x', yref='paper', text=text, showarrow=False,
                       xanchor='left', xshift=5, yanchor='top' if top else 'bottom', font=dict(color=INK_2, size=11),
                       bgcolor='rgba(255,255,255,0.85)')


def show(fig):
    st.plotly_chart(fig, **FULL, config={'displaylogo': False, 'locale': 'es',
                                         'modeBarButtonsToRemove': ['lasso2d', 'select2d']})


# ---------- datos ----------
def experiments():
    if not RESULTS.exists():
        return []
    return sorted(f'{s.parent.parent.name}/{s.parent.name}' for s in RESULTS.glob('*/*/summary.json')
                  if not s.parent.parent.name.startswith('_'))


@st.cache_data(show_spinner=False)
def load(key, mtime):
    base, exp = key.split('/')
    folder = RESULTS / base
    summary = json.loads((folder / exp / 'summary.json').read_text())
    runs = pd.read_csv(folder / 'runs.csv') if (folder / 'runs.csv').exists() else pd.DataFrame()
    trials = {p.stem: pd.read_csv(p) for p in sorted((folder / 'trials').glob('*.csv'))}
    return summary, runs, trials


@st.cache_data(show_spinner=False)
def load_monitoring(base, mtime):
    return json.loads((RESULTS / base / 'monitoring' / 'monitoring.json').read_text())


def short_month(yyyymm):
    return f"{MONTHS[yyyymm % 100 - 1][:3]} {yyyymm // 100}"


def runs_of(runs, study, trial, seeds):
    d = runs[(runs['study'] == study) & (runs['trial'] == trial)]
    return d[d['seed'].isin(seeds)]


def header(sub):
    st.markdown(f'<div class="top"><div class="kicker">Modelo de bajas · experimento de submuestreo</div>'
                f'<h1>¿Se puede entrenar con menos datos sin perder ganancia?</h1><p class="sub">{bold(sub)}</p></div>',
                unsafe_allow_html=True)


HYPOTHESIS = ('**Hipótesis:** si cada árbol se entrena con solo una fracción de los clientes que permanecen (CONTINUA), '
              'el modelo alcanza una ganancia y un AUC equivalentes a los del modelo base, entrenado con la totalidad de '
              'los datos, en un tiempo considerablemente menor. Sin embargo, por debajo de cierta proporción el tiempo '
              'deja de disminuir, por lo que reducirla más allá de ese punto solo pone en riesgo el resultado.')


# ======================= diseño anterior (solo lectura) =======================
def old_design_view(folder, key):
    s = json.loads((folder / 'summary.json').read_text())
    cfg = s['config']
    header(HYPOTHESIS)
    page_head('Diseño anterior', 'Resultados del diseño anterior',
              'En este diseño, Optuna buscaba r entre 0,01 y 1 junto con los demás hiperparámetros, y se comparaba contra '
              'la búsqueda del modelo base. La curva de r usaba la mejor configuración del modelo base **sin '
              'reajustarla** para cada r, lo que perjudica a los valores pequeños de r. Se conserva solo como referencia.')
    sr, tm, v = s.get('searches', {}), s.get('time', {}), s.get('verdict', {})
    if sr:
        p, a = sr['profit_band'], sr['auc']
        section('Resultado principal', NAVY)
        kpis([('Valor de r elegido por Optuna', r_txt(s.get('undersampling_best_r')), 'la mejor prueba de la búsqueda', ACCENT['eleccion']),
              ('Ganancia en el mes de prueba', pct(p['mean_diff'] / p['baseline_mean'], 1, sign=True),
               f"{pesos(p['undersampling_mean'])} contra {pesos(p['baseline_mean'])}<br>"
               + status(p['non_inferior'], 'Dentro de la tolerancia', 'Fuera de la tolerancia'), ACCENT['ganancia']),
              ('AUC', signed(a['mean_diff'], 4), f"{num(a['undersampling_mean'], 4)} contra {num(a['baseline_mean'], 4)}",
               ACCENT['auc']),
              ('Tiempo total', pct(-tm['reduction'], 0, sign=True),
               f"{dur(tm['undersampling']['total_seconds'])} contra {dur(tm['baseline']['total_seconds'])}<br>"
               + status(tm['meets_threshold'], f"Reducción de al menos {pct(cfg['time_reduction'], 0)}", 'Reducción inferior a la exigida'),
               ACCENT['tiempo'])], NAVY)
        section('Comparación en el mes de prueba', NAVY,
                f"Las 3 mejores configuraciones de cada búsqueda, con {cfg['n_seeds']} semillas cada una, comparadas de a "
                'pares (misma posición y misma semilla).')
        names = {'profit_band': 'Ganancia (franja 5–9 %)', 'profit_peak': 'Ganancia máxima', 'auc': 'AUC'}
        rows = []
        for m in ['profit_band', 'profit_peak', 'auc']:
            c = sr[m]
            f = (lambda x: num(x, 4)) if m == 'auc' else pesos
            rows.append([names[m], f(c['baseline_mean']), f(c['undersampling_mean']),
                         (signed(c['mean_diff'], 4) if m == 'auc' else pesos(c['mean_diff'], 2)),
                         f"[{f(c['ci_low'])}; {f(c['ci_high'])}]", num(c['wilcoxon_p'], 4),
                         status(c['non_inferior'], 'Sí', 'No')])
        table(pd.DataFrame(rows, columns=['Medida', 'Modelo base', 'Submuestreo', 'Diferencia', 'Intervalo del 95 %',
                                          'Wilcoxon (p)', 'Dentro de la tolerancia']))
    curve = pd.read_csv(folder / 'runs_curve.csv') if (folder / 'runs_curve.csv').exists() else pd.DataFrame()
    if not curve.empty:
        section('Curva de r con la configuración del modelo base fija', NAVY)
        agg = curve.groupby('r').agg(p=('profit_band', 'mean'), lo=('profit_band', 'min'), hi=('profit_band', 'max'),
                                     t=('train_seconds', 'mean')).reset_index()
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.concat([agg['r'], agg['r'][::-1]]), y=pd.concat([agg['hi'], agg['lo'][::-1]]),
                                     fill='toself', fillcolor=rgba(ACCENT['ganancia'], 0.15), line=dict(width=0),
                                     hoverinfo='skip', name='Rango entre semillas'))
            fig.add_trace(go.Scatter(x=agg['r'], y=agg['p'], mode='lines+markers', name='Promedio',
                                     line=dict(color=ACCENT['ganancia'], width=2.5),
                                     marker=dict(size=8, line=dict(width=2, color='#fff')),
                                     hovertemplate='r = %{x}: $ %{y:,.0f}<extra></extra>'))
            show(r_axis(style(fig, 'Ganancia según r', '', 'Ganancia (franja 5–9 %)'), short=True))
        with right:
            fig = go.Figure(go.Scatter(x=agg['r'], y=agg['t'], mode='lines+markers', name='Promedio',
                                       line=dict(color=ACCENT['tiempo'], width=2.5),
                                       marker=dict(size=8, line=dict(width=2, color='#fff')),
                                       hovertemplate='r = %{x}: %{y:,.1f} s<extra></extra>'))
            show(r_axis(style(fig, 'Tiempo de entrenamiento según r', '', 'Segundos', legend_on=False), short=True))
        caption('la ganancia disminuye desde r = 0,6 porque los hiperparámetros del modelo base no se reajustaron para cada r; '
                'por eso el diseño actual hace una búsqueda por cada valor de r.')
    if v:
        section('Veredicto de ese diseño', NAVY)
        labels = {'profit_about_the_same': 'Misma ganancia que el modelo base', 'time_clearly_less': 'Tiempo considerablemente menor',
                  'time_flattens': 'El tiempo deja de disminuir', 'useful_zone': 'Existe una zona útil',
                  'hypothesis_confirmed': 'Hipótesis confirmada'}
        st.markdown('<div class="card"><div class="checks">' + ''.join(
            f'<div class="check"><div>{status(val, "Sí", "No")}</div><div class="t">{labels.get(k, k)}</div></div>'
            for k, val in v.items()) + '</div></div>', unsafe_allow_html=True)


# ======================= página =======================
keys = experiments()
old_keys = sorted(f'_old_design/{p.parent.name}' for p in RESULTS.glob('_old_design/*/summary.json'))
with st.sidebar:
    st.markdown('#### Resultados')
    options = keys + old_keys
    default = next((i for i, k in enumerate(keys) if k.startswith('full/')), 0)
    key = st.selectbox('Experimento', options, index=default, format_func=experiment_label) if options else None
    st.caption('Cada corrida genera su resumen al finalizar.')

if key is None:
    header(HYPOTHESIS)
    st.info(f'Todavía no hay resultados en `{RESULTS}`. Corré `main.py` primero.')
    st.stop()
if key.startswith('_old_design/'):
    old_design_view(RESULTS / key, key)
    st.stop()

summary, runs, trials = load(key, (RESULTS / key / 'summary.json').stat().st_mtime)
cfg = summary['config']
if summary.get('choice') and 'valid_profit' not in summary['choice']['table'][0]:
    header(HYPOTHESIS)
    st.warning('Este resultado se calculó con la medida de ganancia anterior (promedio de la franja 5–9 %). Hay que '
               'recalcularlo con la medida actual: correr las etapas "validate", "test", "final" y "analyze" (guardan las '
               'curvas de ganancia sin cambiar los entrenamientos ya hechos).')
    st.stop()
seeds = cfg['derived_seeds']['final']
choice, test, curve = summary.get('choice', {}), summary.get('test'), summary.get('curve')
verdict = summary.get('verdict', {})
winner_r = choice.get('winner_r')
delta = cfg['delta']
grid_mode = summary['r_mode'] == 'grid'
VALID, TEST = month(cfg['valid_month']), month(cfg['test_month'])

header(HYPOTHESIS)

TAB_NAMES = ['Resumen', 'Monitoreo', 'Búsquedas de Optuna', 'Elección de la proporción de undersampling', 'Score', 'Tiempo',
             'Metodología y ejecución']
T_SUMMARY, T_MON, T_SEARCH, T_CHOICE, T_METRICS, T_TIME, T_RUN = range(len(TAB_NAMES))
# ?seccion=Tiempo in the URL opens that section directly (handy to share a link to one section).
_wanted = st.query_params.get('seccion')
try:
    tabs = st.tabs(TAB_NAMES, default=_wanted if _wanted in TAB_NAMES else None)
except TypeError:                                    # Streamlit versions without 'default'
    tabs = st.tabs(TAB_NAMES)

# ======================= Resumen =======================
with tabs[T_SUMMARY]:
    page_head('Resumen', 'Principales resultados',
              'La **proporción de undersampling (r)** es la fracción de los CONTINUA que se usa para entrenar. Su valor se '
              f'elige con el mes de **{VALID}** y se confirma con el de **{TEST}**, que no interviene en la elección. A continuación se presentan '
              'el veredicto y las tres métricas que lo determinan: ganancia, AUC y tiempo.')
    confirmed = verdict.get('hypothesis_confirmed')
    if confirmed is None:
        st.markdown('<div class="verdict na">Resultado incompleto: falta alguna etapa del experimento.</div>',
                    unsafe_allow_html=True)
    else:
        kind, pill, text = (('good', GOOD, 'La hipótesis se confirma') if confirmed else
                            ('bad', BAD, 'La hipótesis no se confirma'))
        st.markdown(f'<div class="verdict {kind}"><span class="pill" style="background:{pill}">Veredicto</span>{text}</div>',
                    unsafe_allow_html=True)

    section('Métricas determinantes', NAVY)
    items = [('r elegido', r_txt(winner_r) if winner_r is not None else 'ninguno',
              f'cada árbol usa el {pct(winner_r, 1)} de los CONTINUA' if winner_r is not None
              else 'ningún valor de r conservó la ganancia en validación', ACCENT['eleccion'])]
    if test:
        p, a, t = test['metrics']['profit'], test['metrics']['auc'], test['time']
        items += [
            (f'Ganancia en {TEST}', signed(p['mean_diff'] / 1e6, 2) + '\u00a0M',
             f"diferencia promedio de {p['n_pairs']} comparaciones pareadas con el modelo base "
             f"({pct(p['mean_diff'] / p['reference_mean'], 1, sign=True)})<br>"
             + status(p['non_inferior'], f'Dentro de la tolerancia ({pct(delta, 0)})', 'Fuera de la tolerancia'),
             ACCENT['ganancia']),
            ('AUC', signed(a['mean_diff'], 4),
             f"diferencia promedio de {a['n_pairs']} comparaciones pareadas con el modelo base<br>"
             + status(a['non_inferior'], f'Dentro de la tolerancia ({pct(delta, 0)})', 'Fuera de la tolerancia'),
             ACCENT['auc']),
            ('Tiempo total', pct(-t['reduction'], 0, sign=True),
             f"{dur(t['winner']['total_seconds'])} contra {dur(t['baseline']['total_seconds'])} del modelo base<br>"
             + status(t['meets_threshold'], 'Reducción de al menos ' + pct(cfg['time_reduction'], 0), 'Reducción inferior a la exigida'),
             ACCENT['tiempo'])]
    else:
        why = 'sin un valor de r elegido, no corresponde la confirmación en el mes de prueba'
        items += [(f'Ganancia en {TEST}', '—', why, ACCENT['ganancia']), ('AUC', '—', why, ACCENT['auc']),
                  ('Tiempo total', '—', why, ACCENT['tiempo'])]
    kpis(items, NAVY)

    left, right = st.columns([1.1, 1], gap='medium')
    with left:
        section('Criterios de la hipótesis y su cumplimiento', NAVY)
        rows = [('r_passed_validation', 'Algún valor de r conserva la ganancia en validación',
                 f'Se elige con el mes de {VALID}, sin utilizar el mes de prueba.'),
                ('about_the_same_auc_and_profit', 'Ganancia y AUC equivalentes a los del modelo base',
                 f'En {TEST}, ninguno disminuye más de {pct(delta, 0)}.'),
                ('clearly_less_time', f"Al menos {pct(cfg['time_reduction'], 0)} de reducción del tiempo",
                 'Preparación de datos + búsqueda + modelo final + predicción.'),
                ('time_flattens', 'El tiempo deja de disminuir a partir de cierto valor de r',
                 f"Reducir r por debajo de ese valor ahorra menos del {pct(cfg['flat_share'], 0)} del tiempo del modelo base."),
                ('useful_zone', 'Existe una zona útil',
                 'Un intervalo de valores de r en el que el tiempo ya no disminuye y la ganancia se mantiene.')]
        def row_status(k):
            if winner_r is None and k in ('about_the_same_auc_and_profit', 'clearly_less_time'):
                return '<span class="na">— No aplica</span>'
            return status(verdict.get(k), 'Sí', 'No')
        st.markdown('<div class="card"><div class="checks">' + ''.join(
            f'<div class="check"><div>{row_status(k)}</div>'
            f'<div><div class="t">{label}</div><div class="w">{why}</div></div></div>' for k, label, why in rows)
            + '</div></div>', unsafe_allow_html=True)
    with right:
        section('Síntesis', NAVY)
        lines = []
        if winner_r is None:
            lines.append('Ningún valor de r mantuvo la ganancia del modelo base en el mes de validación; por lo tanto, no hay un valor de '
                         'r para recomendar.')
        else:
            lines.append(f'Entre los valores de r que conservan la ganancia, el más rápido fue **r = {r_txt(winner_r)}**: '
                         f'cada árbol usa el {pct(winner_r, 1)} de los clientes que continúan.')
        if test:
            lines.append(f"En {TEST}, la ganancia cambia **{pct(p['mean_diff'] / p['reference_mean'], 1, sign=True)}** y el "
                         f"AUC **{signed(a['mean_diff'], 4)}** respecto del modelo base, y el proceso completo tarda "
                         f"**{pct(abs(t['reduction']), 0)} {'menos' if t['reduction'] >= 0 else 'más'}**.")
        if curve:
            if curve['useful_zone']:
                lo, hi = curve['useful_zone_range']
                lines.append(f'La **zona útil** va de r = {r_txt(lo)} a r = {r_txt(hi)}: en ese intervalo el tiempo prácticamente no disminuye y '
                             'la ganancia se mantiene.')
            else:
                hold = curve.get('holding_r', [])
                held = ('en el mes de prueba ningún r mantiene a la vez la ganancia y el AUC dentro de la tolerancia'
                        if not hold else
                        f"solo conservan la ganancia y el AUC los valores r = {', '.join(r_txt(r) for r in sorted(hold, reverse=True))}")
                lines.append(f"No se observa una zona útil: {held}, y el tiempo se estabiliza a partir de "
                             f"r = {r_txt(curve['r_flat'])}.")
        exp = summary.get('exploration')
        if exp:
            lines.append(f"La evaluación de todos los valores de r demandó {dur(exp['total_seconds'])}. Es un costo único: una "
                         'vez elegido r, no se repite en cada entrenamiento.')
        st.markdown('<div class="card">' + ''.join(f'<p>{bold(l)}</p>' for l in lines) + '</div>', unsafe_allow_html=True)

    section('Guía de lectura', NAVY)
    st.markdown(f"""<div class="card"><ol>
<li><b>Monitoreo:</b> alertas sobre los datos y la ejecución, tasa de bajas por mes, ingeniería de variables, variables constantes, drift e importancia de las variables.</li>
<li><b>Búsquedas de Optuna:</b> espacio de búsqueda de cada hiperparámetro y desarrollo de las búsquedas del modelo base y de cada valor de r.</li>
<li><b>Elección de la proporción de undersampling:</b> definición de r y regla de elección: con el mes de {VALID} se evalúa cada
valor y se elige el más rápido entre los que no pierden ganancia frente al modelo base.</li>
<li><b>Score:</b> ganancia y AUC; comparación del r elegido con el modelo base en el mes de prueba ({TEST}) y evolución de
cada métrica según r.</li>
<li><b>Tiempo:</b> duración de cada modelo, su composición por etapa y su evolución según r.</li>
<li><b>Metodología y ejecución:</b> datos, meses y parámetros utilizados, e información técnica para
reproducir el experimento.</li>
</ol><p style="margin-bottom:0">En todos los gráficos, el <b style="color:{NAVY}">azul marino</b> es el modelo base; el
color de cada sección corresponde al modelo con submuestreo.</p></div>""", unsafe_allow_html=True)

    section('Glosario', NAVY)
    st.markdown(f"""<div class="card"><ul>
<li><b>CONTINUA:</b> clientes que permanecen activos dos meses después; constituyen la gran mayoría (unos 93 por cada baja).</li>
<li><b>BAJA+2:</b> clientes que se dan de baja dentro de dos meses; son el objetivo de la predicción.</li>
<li><b>Submuestreo (undersampling):</b> entrenar con solo una parte de los CONTINUA, con el fin de reducir el tiempo de entrenamiento.</li>
<li><b>r (proporción de undersampling):</b> fracción de CONTINUA que se utiliza. r = 1 corresponde al modelo base, con
todos; r = 0,1 utiliza el 10 %.</li>
<li><b>Modelo base:</b> modelo de referencia de la materia, entrenado con todos los datos (r = 1).</li>
<li><b>Ganancia de un modelo:</b> ganancia de contactar a su porcentaje de clientes con mayor probabilidad de
baja. El porcentaje es donde su curva de ganancia, promediada sobre sus semillas, es máxima en el mes de validación. Cada BAJA+2 contactado suma $ 1.072.500; cada otro cliente contactado resta $ 27.500.</li>
<li><b>AUC:</b> capacidad del modelo para ordenar a los clientes según su probabilidad de baja (1 corresponde a un orden
perfecto; 0,5, al azar).</li>
<li><b>Semilla:</b> valor que fija el componente aleatorio de un entrenamiento. Repetir con varias semillas permite
estimar qué parte del resultado se debe al azar.</li>
<li><b>Variable constante:</b> variable que en un mes toma el mismo valor para todos los clientes, por una falla en la
carga del dato (en la materia, «variable rota»).</li>
<li><b>PSI:</b> índice de estabilidad poblacional; mide cuánto cambió la distribución de una variable respecto de los
meses de entrenamiento.</li>
<li><b>Tolerancia ({pct(delta, 0)}):</b> pérdida máxima admitida para considerar que el resultado es equivalente.</li>
<li><b>Intervalo de confianza del 95 %:</b> intervalo que contiene la diferencia real con un 95 % de confianza.</li>
<li><b>Zona útil:</b> valores de r en los que el tiempo ya no disminuye y la ganancia se mantiene.</li>
</ul></div>""", unsafe_allow_html=True)

# ======================= Monitoreo =======================
VAR_DESC = {
    'mrentabilidad': 'Rentabilidad del cliente en el mes', 'mrentabilidad_annual': 'Rentabilidad de los últimos 12 meses',
    'mcomisiones': 'Comisiones cobradas al cliente', 'mcomisiones_otras': 'Otras comisiones (monto)',
    'ccomisiones_otras': 'Otras comisiones (cantidad)', 'mactivos_margen': 'Margen por productos de activo (préstamos)',
    'mpasivos_margen': 'Margen por productos de pasivo (depósitos)', 'mcaja_ahorro': 'Saldo en caja de ahorro',
    'mcuenta_corriente': 'Saldo en cuenta corriente', 'mcuentas_saldo': 'Saldo total de las cuentas',
    'Visa_status': 'Estado de la tarjeta Visa', 'Master_status': 'Estado de la tarjeta Mastercard',
    'ctrx_quarter': 'Movimientos en los últimos tres meses', 'Visa_Finiciomora': 'Inicio de la mora en Visa',
    'Master_Finiciomora': 'Inicio de la mora en Mastercard', 'mtarjeta_visa_consumo': 'Consumos con Visa',
    'Visa_mpagospesos': 'Pagos de la Visa en pesos', 'Visa_msaldopesos': 'Saldo de la Visa en pesos',
    'cdescubierto_preacordado': 'Descubierto acordado', 'mprestamos_personales': 'Deuda en préstamos personales',
    'cpayroll_trx': 'Acreditaciones de sueldo', 'mpayroll': 'Sueldo acreditado', 'cproductos': 'Cantidad de productos',
    'ctarjeta_visa_transacciones': 'Transacciones con Visa', 'chomebanking_transacciones': 'Transacciones por homebanking',
    'ccajeros_propios_descuentos': 'Descuentos en cajeros propios (cantidad)',
    'mcajeros_propios_descuentos': 'Descuentos en cajeros propios (monto)',
    'ctarjeta_master_descuentos': 'Descuentos con Mastercard (cantidad)',
    'mtarjeta_master_descuentos': 'Descuentos con Mastercard (monto)',
    'ctarjeta_visa_descuentos': 'Descuentos con Visa (cantidad)', 'mtarjeta_visa_descuentos': 'Descuentos con Visa (monto)',
    'ctarjeta_visa_debitos_automaticos': 'Débitos automáticos en Visa (cantidad)',
    'mttarjeta_visa_debitos_automaticos': 'Débitos automáticos en Visa (monto)',
    'Visa_fultimo_cierre': 'Fecha del último cierre de la Visa', 'Master_fultimo_cierre': 'Fecha del último cierre de la Mastercard',
    'ctransferencias_recibidas': 'Transferencias recibidas (cantidad)',
    'mtransferencias_recibidas': 'Transferencias recibidas (monto)',
    'Visa_mfinanciacion_limite': 'Límite de financiación de la Visa',
    'Master_mfinanciacion_limite': 'Límite de financiación de la Mastercard',
    'Visa_mlimitecompra': 'Límite de compra de la Visa', 'Master_mlimitecompra': 'Límite de compra de la Mastercard',
    'cliente_edad': 'Edad del cliente', 'cliente_antiguedad': 'Antigüedad del cliente',
}
HP_INFO = {
    'num_leaves': ('Hojas por árbol', 'Complejidad máxima de cada árbol: más hojas permiten reglas más específicas.'),
    'learning_rate': ('Tasa de aprendizaje', 'Peso de la corrección que aporta cada árbol; un valor menor requiere más árboles.'),
    'min_data_in_leaf': ('Mínimo de clientes por hoja', 'Evita reglas basadas en muy pocos clientes.'),
    'feature_fraction': ('Fracción de variables', 'Proporción de variables disponibles para cada árbol.'),
    'bagging_fraction': ('Fracción de clientes por árbol', 'Proporción de clientes que utiliza cada árbol. Con submuestreo, '
                         'a los CONTINUA se les aplica además la proporción r.'),
}


HP_SHORT = {'num_leaves': 'Hojas por árbol', 'learning_rate': 'Tasa de aprendizaje',
            'min_data_in_leaf': 'Mín. de clientes por hoja', 'feature_fraction': 'Fracción de variables',
            'bagging_fraction': 'Fracción de clientes'}


def log_ticks(lo, hi):
    """Pocas marcas legibles en un eje logarítmico: los bordes y los 1 y 3 por potencia de 10 entre ellos."""
    vals = [lo, hi]
    for k in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1):
        for m in (1, 3):
            v = m * 10 ** k
            if lo * 2 < v < hi / 2:
                vals.append(v)
    vals = sorted(set(vals))
    return dict(tickvals=vals, ticktext=[hp_txt(v) for v in vals])


def hp_txt(v):
    return num(v, 0) if float(v).is_integer() else num(v, 3).rstrip('0').rstrip(',')


def var_cell(var):
    desc = VAR_DESC.get(var)
    return f'<code>{var}</code>' + (f'<br><span style="color:{MUTED};font-size:12.5px">{desc}</span>' if desc else '')


KIND = {'all_zero': 'todos los valores en cero', 'all_empty': 'todos los valores vacíos', 'constant': 'valor único'}
LEVELS = {'crit': ('✖', 'Crítica', BAD), 'warn': ('▲', 'Advertencia', MID), 'info': ('ℹ', 'Información', NAVY),
          'ok': ('✔', 'Sin observaciones', GOOD)}
FEATURE_TYPES = [
    ('original', 'Variables originales', 'Columnas del dataset, sin el número de cliente ni el mes.'),
    ('lag1', 'Rezago de 1 mes (lag 1)', 'Valor del cliente exactamente un mes antes; vacío si el cliente no figuraba ese mes.'),
    ('lag2', 'Rezago de 2 meses (lag 2)', 'Valor del cliente exactamente dos meses antes.'),
    ('delta1', 'Diferencia con 1 mes antes (delta 1)', 'Valor del mes menos su rezago de 1 mes.'),
    ('delta2', 'Diferencia con 2 meses antes (delta 2)', 'Valor del mes menos su rezago de 2 meses.'),
]


def short_study(study):
    return 'modelo base' if study == 'baseline' else (
        f"r = {r_txt(float(study.split('_r')[-1]))}" if study.startswith('grid_') else 'búsqueda con r en Optuna')


def trial_states(base):
    """Estado de las pruebas de Optuna por búsqueda, leído (solo lectura) de optuna.db."""
    db = RESULTS / base / 'optuna.db'
    if not db.exists():
        return {}
    import sqlite3
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    try:
        rows = con.execute('SELECT s.study_name, t.state, COUNT(*) FROM trials t JOIN studies s USING (study_id) '
                           'GROUP BY 1, 2').fetchall()
    finally:
        con.close()
    out = {}
    for study_name, state, n in rows:
        out.setdefault(study_name, {})[state] = n
    return out


def alert_list(items):
    order = {'crit': 0, 'warn': 1, 'info': 2, 'ok': 3}
    html_rows = ''
    for level, area, title, detail in sorted(items, key=lambda a: (order[a[0]], a[1])):
        icon, label, c = LEVELS[level]
        html_rows += (f'<div class="alert" style="--c:{c}"><div class="lv" style="color:{c}">{icon} {label}</div>'
                      f'<div class="ar">{area}</div><div><div class="t">{bold(title)}</div>'
                      f'<div class="w">{bold(detail)}</div></div></div>')
    st.markdown(f'<div class="alerts">{html_rows}</div>', unsafe_allow_html=True)


def psi_scale(lo, hi):
    st.markdown(f"""<div class="scale">
<div style="--c:#ffffff"><span class="sw2"></span><b>PSI &lt; {num(lo, 2)}</b> Distribución estable</div>
<div style="--c:#f2b3cb"><span class="sw2"></span><b>{num(lo, 2)} – {num(hi, 2)}</b> Cambio moderado</div>
<div style="--c:#8f2f5c"><span class="sw2"></span><b>PSI &gt; {num(hi, 2)}</b> Cambio significativo</div>
</div>""", unsafe_allow_html=True)


with tabs[T_MON]:
    color = ACCENT['monitoreo']
    page_head('Monitoreo', 'Estado de los datos, de la ejecución y del modelo',
              'Alertas sobre la calidad de los datos y la ejecución del experimento, seguidas de la tasa de bajas por mes, '
              'la ingeniería de variables, las variables constantes, los cambios de distribución (drift) y la '
              'importancia de las variables.')
    mon_file = RESULTS / summary['base'] / 'monitoring' / 'monitoring.json'
    mon = load_monitoring(summary['base'], mon_file.stat().st_mtime) if mon_file.exists() else None

    # ---------------- alertas
    alerts = []
    # ejecución
    missing = [name for name, part_ in [('elección de r', choice), ('confirmación en el mes de prueba', test),
                                        ('curva de r', curve)] if not part_]
    if missing:
        alerts.append(('crit', 'Ejecución', 'Experimento incompleto',
                       f"Faltan las etapas: {', '.join(missing)}. El veredicto no puede emitirse."))
    exp_studies = (['baseline'] + [r_['study'] for r_ in choice.get('table', []) if r_['r'] < 1]) if choice else list(trials)
    states = trial_states(summary['base'])
    failed = {s_: {k: v for k, v in states.get(s_, {}).items() if k != 'COMPLETE'} for s_ in exp_studies}
    failed = {s_: v for s_, v in failed.items() if v}
    short = [s_ for s_ in exp_studies if s_ in trials and len(trials[s_]) < cfg['n_trials'] and grid_mode]
    if failed:
        alerts.append(('warn', 'Ejecución', 'Pruebas de Optuna no completadas',
                       '; '.join(f"{short_study(s_)}: " + ', '.join(f'{n} {k.lower()}' for k, n in v.items())
                                 for s_, v in failed.items())
                       + '. Las pruebas fallidas o interrumpidas se reintentan al reanudar la corrida.'))
    if short:
        alerts.append(('warn', 'Ejecución', 'Búsquedas con menos pruebas que las previstas',
                       ', '.join(f"{short_study(s_)} ({len(trials[s_])} de {cfg['n_trials']})" for s_ in short) + '.'))
    if choice:
        base_search = next(r_['search_seconds'] for r_ in choice['table'] if r_['r'] >= 1)
        slow = [r_ for r_ in choice['table'] if r_['r'] < 1 and r_['search_seconds'] > base_search]
        if slow and grid_mode:
            alerts.append(('warn', 'Ejecución', 'Búsquedas más lentas que la del modelo base',
                           ', '.join(f"r = {r_txt(r_['r'])} ({dur(r_['search_seconds'])})" for r_ in slow)
                           + f" contra {dur(base_search)} del modelo base. Con menos datos por árbol se esperaría lo "
                           'contrario: la búsqueda exploró configuraciones con árboles más grandes, lo que agrega ruido '
                           'a la comparación de tiempos.'))
    capped = {s_: int((d_['trees_built'] >= cfg['max_rounds']).sum()) for s_, d_ in trials.items()
              if s_ in exp_studies and 'trees_built' in d_}
    capped = {s_: n for s_, n in capped.items() if n}
    if capped:
        alerts.append(('info', 'Ejecución', f"Pruebas que alcanzaron el máximo de {num(cfg['max_rounds'], 0)} árboles",
                       ', '.join(f'{short_study(s_)}: {n}' for s_, n in capped.items())
                       + '. En esas pruebas la parada temprana no llegó a actuar.'))
    space_ = cfg['search_space']
    edges = {}
    for s_ in exp_studies:
        if s_ not in trials or trials[s_].empty:
            continue
        b_ = trials[s_].loc[trials[s_]['valid_auc_score'].idxmax()]
        for k_, (lo_, hi_, _, log_) in space_.items():
            if k_ not in b_ or b_[k_] != b_[k_]:
                continue
            pos = ((np.log(b_[k_]) - np.log(lo_)) / (np.log(hi_) - np.log(lo_)) if log_ else (b_[k_] - lo_) / (hi_ - lo_))
            if pos < 0.03 or pos > 0.97:
                edges.setdefault(f"{HP_SHORT.get(k_, k_).lower()} ({'borde inferior' if pos < 0.5 else 'borde superior'})",
                                 []).append(short_study(s_))
    if edges:
        alerts.append(('info', 'Ejecución', 'Mejores configuraciones en el borde del rango de búsqueda',
                       '; '.join(f"{k_}: {', '.join(v)}" for k_, v in edges.items())
                       + '. El rango podría resultar insuficiente para ese hiperparámetro (ver Búsquedas de Optuna).'))
    if not any(a_[1] == 'Ejecución' for a_ in alerts):
        alerts.append(('ok', 'Ejecución', 'Ejecución completa y sin fallas',
                       'Todas las etapas finalizaron y todas las pruebas de Optuna se completaron.'))

    if mon:
        valid_m, test_m = mon['valid_month'], mon['test_month']
        broken = pd.DataFrame(mon['broken'])
        psi = pd.DataFrame(mon['psi']).T                       # variables x meses (str)
        lo_psi, hi_psi = mon['psi_levels']
        imp = mon.get('importance', {})
        imp_base = pd.Series(imp.get('baseline', {}).get('by_variable', {}))
        imp_rank = {v: i + 1 for i, v in enumerate(imp_base.index)}
        exp_name = key.split('/')[1]
        imp_win = pd.Series(imp.get(exp_name, {}).get('by_variable', {}))
        cls = pd.DataFrame(mon['classes'])

        def const_in(m):
            return set(broken.loc[broken['month'] == m, 'variable']) if not broken.empty else set()

        def drifted(m):
            col = psi[str(m)].drop(labels=list(const_in(m)), errors='ignore')
            return col[col > hi_psi], col[(col > lo_psi) & (col <= hi_psi)]

        for m, label_m, lvl in [(test_m, f'el mes de prueba ({TEST})', 'warn'), (valid_m, f'el mes de validación ({VALID})', 'warn')]:
            c_ = sorted(const_in(m), key=lambda v: imp_rank.get(v, 999))
            top_ = [v for v in c_ if imp_rank.get(v, 999) <= 20]
            if c_:
                alerts.append(('crit' if top_ else lvl, 'Datos', f'{len(c_)} variables constantes en {label_m}',
                               ', '.join(f'<code>{v}</code>' for v in c_[:6]) + (', entre otras. ' if len(c_) > 6 else '. ')
                               + (('Una de ellas figura' if len(top_) == 1 else f'{len(top_)} de ellas figuran')
                                  + ' entre las 20 más importantes del modelo base. ' if top_
                                  else 'Ninguna figura entre las 20 más importantes del modelo base. ')
                               + 'Afectan por igual al modelo base y al modelo con submuestreo.'))
        train_c = sorted({b_['month'] for b_ in mon['broken'] if b_['month'] in mon['train_months']})
        if train_c:
            alerts.append(('info', 'Datos', 'Variables constantes en meses de entrenamiento',
                           '; '.join(f"{month(m)}: {len(const_in(m))}" for m in train_c) + '.'))
        for m, label_m in [(test_m, TEST), (valid_m, VALID)]:
            big, _ = drifted(m)
            top_ = [v for v in big.index if imp_rank.get(v, 999) <= 20]
            if top_:
                alerts.append(('warn', 'Drift', f'Cambio significativo en variables importantes ({label_m})',
                               ', '.join(f"<code>{v}</code> (PSI {num(big[v], 2)})" for v in top_)
                               + f'. Están entre las 20 más importantes del modelo base.'))
            elif len(big):
                alerts.append(('info', 'Drift', f'{len(big)} variables con cambio significativo en {label_m}',
                               'Ninguna figura entre las 20 más importantes del modelo base.'))
        tr = cls[cls['month'].isin(mon['train_months'])]
        tr_share = (tr['baja1'] + tr['baja2']).sum() / tr['clients'].sum()
        te_share = float(cls.loc[cls['month'] == test_m, 'churn_share'].iloc[0])
        if abs(te_share / tr_share - 1) > 0.2:
            alerts.append(('warn', 'Datos', 'La tasa de bajas del mes de prueba difiere de la de entrenamiento',
                           f'{pct(te_share, 2)} en {TEST} contra {pct(tr_share, 2)} en entrenamiento '
                           f'({pct(te_share / tr_share - 1, 0, sign=True)}).'))
        pred = mon.get('predictions')
        if pred:
            for k_, lab_ in [('valid', VALID), ('test', TEST)]:
                ratio = pred[k_]['mean_score'] / pred[k_]['positive_share']
                if abs(ratio - 1) > 0.2:
                    alerts.append(('warn', 'Modelo', f'Calibración desviada en {lab_}',
                                   f"Probabilidad promedio predicha {pct(pred[k_]['mean_score'], 2)} contra una tasa real "
                                   f"de {pct(pred[k_]['positive_share'], 2)}."))
            drop = pred['valid']['auc'] - pred['test']['auc']
            if drop > 0.02:
                alerts.append(('warn', 'Modelo', 'Caída del AUC entre validación y prueba',
                               f"{num(pred['valid']['auc'], 4)} en {VALID} contra {num(pred['test']['auc'], 4)} en "
                               f"{TEST} (modelo base, un entrenamiento)."))
        for area in ['Datos', 'Drift', 'Modelo']:
            if not any(a_[1] == area for a_ in alerts):
                alerts.append(('ok', area, 'Sin observaciones', 'Ningún indicador superó los umbrales de alerta.'))
    if summary['base'] != 'full':
        alerts.append(('info', 'Datos', 'Resultados sobre una muestra del 10 % de los clientes',
                       'Con pocos BAJA+2 por mes, un solo caso modifica la ganancia en $ 1,07 M; las conclusiones '
                       'deben confirmarse con los datos completos.'))

    section('Alertas', color, 'Revisión automática de la ejecución, de la calidad de los datos, del drift y del modelo. '
            'Se ordenan por gravedad.')
    alert_list(alerts)

    if not mon:
        flag = '' if summary['base'] == 'full' else f" --sample-clients {cfg['sample_clients']:g}"
        st.info('Las métricas de datos, drift e importancia aún no están disponibles para estos datos. Se calculan con '
                f'`python monitor.py{flag}`, una vez finalizada la corrida.')
    else:
        role = {m: 'entrenamiento' for m in mon['train_months']} | {valid_m: 'validación', test_m: 'prueba'}
        t_row = cls[cls['month'] == test_m].iloc[0]
        big_t, mid_t = drifted(test_m)
        sample_note = '' if summary['base'] == 'full' else ' (muestra)'
        section('Indicadores principales', color)
        kpis([(f'Clientes en {TEST}', num(t_row['clients'], 0), f"{num(t_row['baja2'], 0)} BAJA+2 y "
               f"{num(t_row['baja1'], 0)} BAJA+1{sample_note}", color),
              (f'Tasa de bajas en {TEST}', pct(t_row['churn_share'], 2),
               f"BAJA+1 y BAJA+2 sobre el total de clientes; en entrenamiento, {pct(tr_share, 2)}", color),
              (f'Variables constantes en {TEST}', num(len(const_in(test_m)), 0),
               f"de {mon['n_raw_variables']} variables originales", color),
              (f'Cambio significativo en {TEST}', num(len(big_t), 0),
               f"variables con PSI mayor a {num(hi_psi, 2)}, sin contar las constantes; {len(mid_t)} con cambio "
               'moderado', color)], color)

        # ---- ingeniería de variables
        ft = mon.get('feature_types')
        if ft:
            by_type = imp.get('baseline', {}).get('by_type', {})
            section('Ingeniería de variables', color,
                    f"A partir de las {num(ft.get('original', 0), 0)} variables originales se agregaron rezagos y "
                    'diferencias de uno y dos meses, calculados por calendario: el rezago de un mes es el valor del mismo '
                    'cliente en el mes calendario anterior, no la fila anterior.')
            rows = [[name, num(ft.get(t_, 0), 0), desc, (pct(by_type[t_], 1) if t_ in by_type else '—')]
                    for t_, name, desc in FEATURE_TYPES if t_ in ft]
            added = sum(n for t_, n in ft.items() if t_ != 'original')
            rows.append(['<b>Total</b>', f"<b>{num(sum(ft.values()), 0)}</b>",
                         f"{num(added, 0)} variables agregadas", '<b>100 %</b>' if by_type else '—'])
            table(pd.DataFrame(rows, columns=['Tipo', 'Cantidad', 'Definición', 'Participación en la importancia']),
                  text=True)
            conv = mon.get('text_columns_converted') or []
            if by_type:
                concl('info', f"Las {num(added, 0)} variables agregadas aportan el "
                              f"{pct(1 - by_type.get('original', 0), 0)} de la importancia del modelo base; las "
                              f"variables originales, el {pct(by_type.get('original', 0), 0)}.")
            caption('la participación en la importancia corresponde al modelo base (ganancia de información, un '
                    'entrenamiento de su mejor configuración).'
                    + (f" Las columnas de texto {', '.join(f'<code>{c}</code>' for c in conv)} se convirtieron a número "
                       '(los valores no numéricos quedan vacíos).' if conv else ''))

        # ---- clientes y bajas
        section('Clientes y tasa de bajas por mes', color,
                'Cantidad de clientes por mes y proporción que se da de baja. BAJA+1 corresponde a los clientes que se dan '
                'de baja al mes siguiente; BAJA+2, a los que se dan de baja dentro de dos meses, que son el objetivo del '
                'modelo.')
        xlab = [f"{MONTHS[m % 100 - 1][:3]}<br>{m // 100}" for m in cls['month']]

        def role_bands(fig):
            """Bandas de fondo con el papel de cada mes (entrenamiento, validación, prueba)."""
            groups = []
            for i, m in enumerate(cls['month']):
                r_ = role.get(m, '')
                if groups and groups[-1][0] == r_:
                    groups[-1][2] = i
                else:
                    groups.append([r_, i, i])
            for j, (r_, a, b) in enumerate(groups):
                fig.add_vrect(x0=a - 0.5, x1=b + 0.5, fillcolor='#f5f7fa' if j % 2 == 0 else '#ffffff', line_width=0,
                              layer='below')
                fig.add_annotation(x=(a + b) / 2, y=1.0, xref='x', yref='paper', text=r_, showarrow=False,
                                   yanchor='bottom', font=dict(color=MUTED, size=11))
            return fig
        left, right = st.columns(2, gap='small')
        with left:
            fig = go.Figure(go.Bar(x=xlab, y=cls['clients'], marker=dict(color=color, cornerradius=4, line=dict(width=0)),
                                   name='Clientes', hovertemplate='%{y:,.0f} clientes<extra></extra>'))
            show(role_bands(style(fig, 'Clientes por mes', '', 'Clientes', legend_on=False)))
        with right:
            fig = go.Figure()
            for col_, name, c in [('baja2_share', 'BAJA+2', MON_DARK), ('baja1_share', 'BAJA+1', color)]:
                fig.add_trace(go.Bar(x=xlab, y=cls[col_] * 100, name=name, marker=dict(color=c, line=dict(width=2, color='#fff')),
                                     customdata=cls[col_.replace('_share', '')],
                                     hovertemplate=name + ': %{y:.2f} % (%{customdata:,.0f} clientes)<extra></extra>'))
            fig.add_trace(go.Scatter(x=xlab, y=cls['churn_share'] * 100, mode='text', showlegend=False, hoverinfo='skip',
                                     text=[pct(v, 2) for v in cls['churn_share']], textposition='top center',
                                     textfont=dict(color=INK_2, size=11)))
            fig.update_layout(barmode='stack')
            fig.update_yaxes(range=[0, float(cls['churn_share'].max()) * 100 * 1.25])
            show(role_bands(style(fig, 'Tasa de bajas por mes', '', '% de los clientes')))
        gap = [m for m in range(min(mon['months']), max(mon['months']) + 1) if m % 100 <= 12 and m not in mon['months']]
        caption(f"el modelo se entrena con {month(mon['train_months'][0])} a {month(mon['train_months'][-1])}; r se elige "
                f"con {VALID} y se confirma con {TEST}."
                + (f" {' y '.join(MONTHS[m % 100 - 1] for m in gap).capitalize()} se excluyen para respetar la distancia "
                   'de dos meses que requiere la etiqueta BAJA+2.' if gap else ''))

        # ---- variables constantes
        section('Variables con valor constante en el mes', color,
                'Una variable se considera **constante** en un mes cuando todos los clientes presentan el mismo valor '
                '(por lo general, cero) a pesar de que en los demás meses toma valores variados. Indica una falla en la '
                'carga del dato; en la materia se las conoce como «variables rotas». La revisión incluye los meses que '
                'no están en el archivo (julio y septiembre), a través de los rezagos.')
        if broken.empty:
            concl('good', 'No se detectaron variables constantes en ningún mes.')
        else:
            bt = sorted(const_in(test_m))
            top_hit = [v for v in bt if imp_rank.get(v, 999) <= 20]
            concl('bad' if bt else 'good',
                  (f"En {TEST}, el mes de prueba, **{len(bt)} variables** son constantes. Afectan por igual al modelo base "
                   'y al modelo con submuestreo, por lo que la comparación entre ambos no se ve sesgada, aunque pueden '
                   'reducir el resultado de los dos.' if bt else f'{TEST} no presenta variables constantes.')
                  + ((' Una de ellas figura' if len(top_hit) == 1 else f' {len(top_hit)} de ellas figuran')
                     + ' entre las 20 más importantes del modelo base.' if top_hit else
                     ' Ninguna figura entre las 20 variables más importantes del modelo base.' if bt else ''))
            months_all = sorted(set(broken['month']) | set(mon['months']))
            rows = []
            for var in sorted(broken['variable'].unique(), key=lambda v: (imp_rank.get(v, 999), v)):
                cells = []
                for m in months_all:
                    hit = broken[(broken['variable'] == var) & (broken['month'] == m)]
                    cells.append(f'<span class="ko" title="{KIND[hit.iloc[0]["kind"]]}">✖</span>' if len(hit) else
                                 f'<span style="color:{AXIS}">·</span>')
                rows.append([var_cell(var), (f'{imp_rank[var]}.º' if var in imp_rank else 'sin uso')] + cells)
            head = [f"{short_month(m)}{' · ' + role[m] if m in role and m not in mon['train_months'] else ''}"
                    for m in months_all]
            kinds = sorted({KIND[k] for k in broken['kind']})
            table(pd.DataFrame(rows, columns=['Variable', 'Puesto en importancia'] + head))
            caption(f"cada fila es una variable que resultó constante en al menos un mes; ✖ indica el mes afectado "
                    f"({' o '.join(kinds)}). El puesto en importancia corresponde al modelo base (1.º = la de mayor "
                    'importancia). Las descripciones son aproximadas, inferidas del nombre de cada variable.')

        # ---- drift
        section('Cambios de distribución (drift)', color,
                'El **índice de estabilidad poblacional (PSI)** compara la distribución de cada variable en un mes con su '
                'distribución en los meses de entrenamiento. Se interpreta según tres niveles:')
        psi_scale(lo_psi, hi_psi)
        big_v, mid_v = drifted(valid_m)
        kpis([(f'Cambio significativo en {VALID}', num(len(big_v), 0),
               f'{len(mid_v)} con cambio moderado (sin contar las constantes)', color),
              (f'Cambio significativo en {TEST}', num(len(big_t), 0),
               f'{len(mid_t)} con cambio moderado (sin contar las constantes)', color)], color)
        order_v = psi[[str(valid_m), str(test_m)]].max(axis=1).sort_values(ascending=False).head(20).index
        hm = psi.loc[order_v, [str(m) for m in mon['months']]]
        text = [[('constante' if v in const_in(int(m)) else num(hm.loc[v, m], 2)) for m in hm.columns] for v in hm.index]
        fig = go.Figure(go.Heatmap(
            z=np.minimum(hm.to_numpy(), 1.0), x=[f"{short_month(int(m))}<br>{role.get(int(m), '')}" for m in hm.columns],
            y=list(hm.index), text=text, texttemplate='%{text}', textfont=dict(size=11),
            colorscale=[[0, '#ffffff'], [lo_psi, '#fbe3ec'], [hi_psi, '#f2b3cb'], [0.6, '#e87ba4'], [1, '#8f2f5c']],
            zmin=0, zmax=1, xgap=2, ygap=2, colorbar=dict(title='PSI', tickvals=[0, lo_psi, hi_psi, 0.5, 1],
                                                          ticktext=['0', num(lo_psi, 2), num(hi_psi, 2), '0,5', '1 o más']),
            customdata=[[VAR_DESC.get(v, v)] * len(hm.columns) for v in hm.index],
            hovertemplate='%{y} · %{x}<br>%{customdata}<br>PSI: %{text}<extra></extra>'))
        fig.update_yaxes(autorange='reversed', tickfont=dict(family='ui-monospace, Menlo, monospace', size=11))
        show(style(fig, f'Las 20 variables con mayor cambio en {VALID} o {TEST} (PSI por mes)', '', '',
                   height=120 + 26 * len(hm), legend_on=False))
        caption('los meses de entrenamiento también tienen PSI porque cada uno se compara con los seis meses en conjunto. '
                'Las variables de fecha (como el último cierre de la tarjeta) cambian todos los meses por su naturaleza, y '
                'en las variables expresadas en pesos parte del cambio responde a la inflación. «constante» indica los '
                'meses en que la variable no tiene información.')

        # ---- importancia
        if not imp_base.empty:
            section('Importancia de las variables', color,
                    'Proporción de la ganancia de información total que aporta cada variable, incluidos sus rezagos y '
                    'diferencias. Se obtiene de un entrenamiento de la mejor configuración de cada modelo, con la misma '
                    'semilla.')
            top = imp_base.head(15)
            names = list(top.index)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=names, x=top.values * 100, orientation='h', name='Modelo base',
                                 marker=dict(color=NAVY, line=dict(width=2, color='#fff')),
                                 hovertemplate='%{y}: %{x:.1f} %<extra>Modelo base</extra>'))
            if not imp_win.empty:
                wr_ = imp[exp_name].get('r')
                fig.add_trace(go.Bar(y=names, x=[imp_win.get(v, 0) * 100 for v in names], orientation='h',
                                     name=f'r elegido ({r_txt(wr_)})', marker=dict(color=color, line=dict(width=2, color='#fff')),
                                     hovertemplate='%{y}: %{x:.1f} %<extra>r elegido</extra>'))
            fig.update_layout(barmode='group', bargap=0.25)
            fig.update_yaxes(autorange='reversed', tickfont=dict(family='ui-monospace, Menlo, monospace', size=11))
            left, right = st.columns([1.1, 1], gap='small')
            with left:
                show(style(fig, 'Las 15 variables más importantes del modelo base', '% de la importancia total', '',
                           height=140 + 34 * len(names)))
            with right:
                bt = const_in(test_m)
                rows = [[var_cell(v), pct(imp_base[v], 1),
                         (pct(imp_win.get(v, 0), 1) if not imp_win.empty else '—'),
                         ('<span class="ko">✖ constante</span>' if v in bt else
                          num(psi.loc[v, str(test_m)], 2) if v in psi.index else '—')] for v in names]
                table(pd.DataFrame(rows, columns=['Variable', 'Modelo base', 'r elegido', f'PSI en {short_month(test_m)}']))
            share3 = imp_base.head(3).sum()
            concl('info', f"Las tres variables más importantes ({', '.join(imp_base.index[:3])}) concentran el "
                          f"{pct(share3, 0)} de la importancia del modelo base"
                          + (', y el modelo con submuestreo se apoya en las mismas variables.' if not imp_win.empty and
                             set(imp_win.index[:3]) == set(imp_base.index[:3]) else '.'))

        # ---- calibración
        if pred:
            section('Calibración y desempeño del modelo base', color,
                    'Si el modelo está bien calibrado, la probabilidad promedio predicha debería aproximarse a la tasa '
                    'real de bajas. El AUC indica si el modelo conserva su capacidad de ordenar a los clientes.')
            rows = [[f"{month(pred[k]['month'])} ({'validación' if k == 'valid' else 'prueba'})",
                     pct(pred[k]['positive_share'], 2), pct(pred[k]['mean_score'], 2),
                     num(pred[k]['auc'], 4)] for k in ['valid', 'test']]
            table(pd.DataFrame(rows, columns=['Mes', 'Tasa real de bajas', 'Probabilidad promedio predicha', 'AUC']))
            caption('un único entrenamiento del modelo base con su mejor configuración. La tasa real incluye BAJA+1 y '
                    'BAJA+2, que es lo que el modelo aprende a predecir.')


# ======================= Elección de la proporción de undersampling =======================
with tabs[T_CHOICE]:
    color = ACCENT['eleccion']
    page_head('Elección de la proporción de undersampling', 'Definición y criterio de elección',
              f'Se define la proporción de undersampling y se presentan la regla con la que se eligió en {VALID} y el desempeño de '
              f'cada valor en {TEST}, el mes de prueba.')
    r_sorted = sorted(cfg['r_list'], reverse=True)
    how_sample = ('cada árbol selecciona al azar su propia muestra' if summary['r_apply'] == 'per_tree' else
                  'una única muestra, seleccionada antes de entrenar, para todos los árboles')
    how_tested = (f"{len(r_sorted)} valores, de {r_txt(r_sorted[0])} a {r_txt(r_sorted[-1])}, cada uno con su propia búsqueda "
                  f"de hiperparámetros y {cfg['n_seeds_valid']} semillas" if grid_mode else
                  f"{len(r_sorted)} valores, de {r_txt(r_sorted[0])} a {r_txt(r_sorted[-1])}, elegidos por Optuna junto con los "
                  f"demás hiperparámetros; {cfg['n_seeds_valid']} semillas por valor")
    st.markdown(f"""<div class="card"><h3>Qué es la proporción de undersampling (r)</h3><ul style="margin-bottom:0">
<li><b>Definición:</b> fracción de los CONTINUA que utiliza cada árbol. Los clientes que se dan de baja se utilizan
siempre en su totalidad.</li>
<li><b>Ejemplos:</b> r = 1 utiliza todos los CONTINUA (modelo base); r = 0,1, el 10 % (unos 9 por cada baja, en lugar
de 93).</li>
<li><b>Muestreo:</b> {how_sample}.</li>
<li><b>Valores evaluados:</b> {how_tested}.</li></ul></div>""", unsafe_allow_html=True)
    if not choice:
        st.info('Todavía no hay entrenamientos de validación (etapa "validate").')
    else:
        table_c = pd.DataFrame(choice['table'])
        CONF = pct(choice.get('confidence', 0.95), 1).replace(',0\u00a0', '\u00a0')
        st.markdown(f"""<div class="card"><h3>Regla de elección (fijada antes de conocer los resultados)</h3><ol style="margin-bottom:0">
<li><b>Cumple la regla</b> el valor de r cuya pérdida de ganancia frente al modelo base no supera la tolerancia
({pct(delta, 0)}), con un {CONF} de confianza.</li>
<li>Entre los que la cumplen se elige el <b>más rápido</b>. Si la diferencia de tiempo es menor al
{pct(cfg['near_fastest'], 0)} del tiempo del modelo base, se considera un empate y se elige el <b>r mayor</b>, que
utiliza más datos con un ahorro similar.</li></ol></div>""", unsafe_allow_html=True)

        und = table_c[table_c['r'] < 1].reset_index(drop=True)
        margin = float(table_c['margin'].iloc[0])
        table_c['state'] = np.where(table_c['passes'], 'cumple', 'no cumple')
        und['state'] = table_c.loc[table_c['r'] < 1, 'state'].to_numpy()
        n_pass = int(und['passes'].sum())
        if winner_r is not None:
            w = table_c[table_c['winner']].iloc[0]
            fastest = table_c[table_c['passes']].sort_values('choice_seconds').iloc[0]
            gap_s = w['choice_seconds'] - fastest['choice_seconds']
            tie = (f" El más rápido fue r = {r_txt(fastest['r'])} ({dur(fastest['choice_seconds'])}); r = {r_txt(w['r'])} "
                   f"tardó {dur(w['choice_seconds'])}. La diferencia ({dur(gap_s)}) es menor al {pct(cfg['near_fastest'], 0)} "
                   f"del tiempo del modelo base ({dur(cfg['near_fastest'] * table_c.loc[table_c['r'] == 1, 'choice_seconds'].iloc[0])}), "
                   f"por lo que se considera un empate y se elige el r mayor, que utiliza más datos con un ahorro similar."
                   if fastest['r'] != w['r'] else '')
            concl('good', f"Cumplen la regla {n_pass} de {len(und)} valores de r. Se eligió **r = {r_txt(winner_r)}**.{tie}")
        else:
            concl('bad', 'Ningún valor de r conservó la ganancia del modelo base: no hay un valor para recomendar.')

        section('Ganancia y tiempo de cada valor de r', color,
                'Cada fila es un valor de r. A la izquierda, la diferencia de ganancia respecto del modelo base; a la '
                'derecha, el tiempo utilizado.')
        legend([('Cumple la regla', color), ('No cumple', '#b9c2ce'), ('Elegido (★)', color)])
        order = [r_txt(r) for r in und['r']]
        state_color = [('#b9c2ce' if st_ == 'no cumple' else color) for st_ in und['state']]
        left, right = st.columns([1.25, 1], gap='small')
        with left:
            fig = go.Figure()
            fig.add_vrect(x0=-margin / 1e6, x1=float(max(und['ci_high'].max(), margin)) / 1e6 * 1.08,
                          fillcolor=rgba(GOOD, 0.05), line_width=0)
            fig.add_vline(x=-margin / 1e6, line=dict(color=INK_2, width=1, dash='dot'),
                          annotation_text=f'tolerancia (−{pct(delta, 0)})', annotation_position='top left',
                          annotation_font=dict(color=INK_2, size=11))
            fig.add_vline(x=0, line=dict(color=AXIS, width=1))
            fig.add_trace(go.Scatter(
                x=und['diff'] / 1e6, y=order, mode='markers', showlegend=False,
                marker=dict(color=[c if st_ == 'cumple' else '#ffffff' for st_, c in zip(und['state'], state_color)],
                            symbol=['star' if w else 'circle' for w in und['winner']],
                            size=[17 if w else 10 for w in und['winner']],
                            line=dict(width=2, color=state_color)),
                error_x=dict(type='data', symmetric=False, array=(und['ci_high'] - und['diff']) / 1e6,
                             arrayminus=(und['diff'] - und['ci_low']) / 1e6, color='#9aa6b5', thickness=1.6, width=0),
                customdata=np.stack([und['ci_low'] / 1e6, und['ci_high'] / 1e6, und['n_trials_with_r'],
                                     und['state']], axis=1),
                hovertemplate='r = %{y}<br>diferencia: $ %{x:,.2f} M<br>intervalo: [%{customdata[0]:,.2f}; '
                              '%{customdata[1]:,.2f}] M<br>%{customdata[3]}<extra></extra>'))
            fig.update_yaxes(type='category', categoryorder='array', categoryarray=order[::-1], title='r')
            show(style(fig, 'Diferencia de ganancia frente al modelo base', f'Millones de $ (con intervalo del {CONF})', 'r',
                       height=90 + 30 * len(order), legend_on=False))
        with right:
            fig = go.Figure(go.Bar(
                x=und['choice_seconds'] / 60, y=order, orientation='h',
                marker=dict(color=[color if w else '#c3ccd8' for w in und['winner']], cornerradius=4,
                            line=dict(width=0)),
                customdata=und['choice_seconds'].map(dur),
                hovertemplate='r = %{y}: %{customdata}<extra></extra>'))
            base_min = float(table_c.loc[table_c['r'] == 1, 'choice_seconds'].iloc[0]) / 60
            fig.add_vline(x=base_min, line=dict(color=NAVY, width=1.5),
                          annotation_text=f'modelo base: {dur(base_min * 60)}', annotation_position='top right',
                          annotation_font=dict(color=NAVY, size=11))
            fig.update_yaxes(type='category', categoryorder='array', categoryarray=order[::-1], showticklabels=False)
            show(style(fig, 'Tiempo utilizado' + (' (búsqueda + entrenamiento)' if grid_mode else ' (entrenamiento)'),
                       'Minutos', '', height=90 + 30 * len(order), legend_on=False))
        caption('un valor de r cumple la regla si todo su intervalo (la línea horizontal) queda a la derecha de la tolerancia. Entre los que '
                'la cumplen, se elige el de menor tiempo, resaltado en ambos gráficos.')

        section('Detalle por valor de r', color,
                f'Las columnas de elección son de **{VALID}**. La última columna muestra el resultado de cada r en **{TEST}**, '
                'el mes de prueba, incluido solo como referencia: **no interviene en la elección**. El veredicto se determina en ese mes.')
        test_diff = ({round(row['r'], 6): row['profit_diff'] for row in curve['table']} if curve else {})
        show_df = pd.DataFrame({
            'r': table_c['r'].map(r_txt),
            'Pruebas de Optuna': table_c['n_trials_with_r'],
            '% contactado': table_c['list_share'].map(lambda v: pct(v, 1)),
            'Ganancia promedio': table_c['valid_profit'].map(pesos),
            f'Diferencia en {VALID}': [('—' if r == 1 else pesos(v, 2)) for r, v in zip(table_c['r'], table_c['diff'])],
            f'Intervalo del {CONF}': [('—' if r == 1 else f'[{num(lo / 1e6, 2)}; {num(hi / 1e6, 2)}] M')
                                   for r, lo, hi in zip(table_c['r'], table_c['ci_low'], table_c['ci_high'])],
            '¿Cumple?': [('referencia' if r == 1 else status(ok, 'Sí', 'No')) for r, ok in zip(table_c['r'], table_c['passes'])],
            'Tiempo': table_c['choice_seconds'].map(dur),
            f'Diferencia en {TEST} (no se usó)': [('—' if r == 1 or round(r, 6) not in test_diff else
                                                 f'<span style="color:{MUTED}">{pesos(test_diff[round(r, 6)], 2)}</span>')
                                                for r in table_c['r']],
        })
        table(show_df, highlight=list(table_c['winner']))

        if curve:
            ct = pd.DataFrame(curve['table'])                     # r from 1 down
            base = ct[ct['r'] == 1].iloc[0]
            p_margin, a_margin = delta * abs(base['profit_mean']), delta * abs(base['auc_mean'])
            p_ok = ct['profit_ci_low'] > -p_margin
            a_ok = ct['auc_ci_low'] > -a_margin
            saving = 1 - ct['time_seconds'] / base['time_seconds']
            t_ok = saving >= cfg['time_reduction']
            all_ok = p_ok & a_ok & t_ok
            section(f'Resultado de cada valor de r en el mes de prueba ({TEST})', color,
                    f"Cada r con su **mejor configuración** (una sola), entrenada con {cfg['n_seeds']} semillas y comparada con el modelo "
                    f'base semilla por semilla. Hay una columna por cada criterio de la hipótesis: la **ganancia** y el '
                    f"**AUC** no pueden disminuir más que la tolerancia ({pct(delta, 0)}), y el **tiempo** debe reducirse al menos "
                    f"un {pct(cfg['time_reduction'], 0)}. La última columna indica si un valor de r cumple los tres criterios. Este mes no participó "
                    'de la elección.')
            n_all = int((all_ok & (ct['r'] < 1)).sum())
            if winner_r is not None:
                w = ct[(ct['r'] - winner_r).abs() < 1e-9].iloc[0]
                i = w.name
                fails = [name for name, ok in [('la ganancia', p_ok[i]), ('el AUC', a_ok[i]), ('el tiempo', t_ok[i])] if not ok]
                verdict_w = ('cumple los tres criterios' if not fails else
                             'no cumple por ' + ' y '.join(fails) if len(fails) <= 2 else 'no cumple ningún criterio')
                concl('good' if not fails else 'bad',
                      f"En {TEST}, el r elegido ({r_txt(winner_r)}) {verdict_w}: ganancia "
                      f"{pct(w['profit_diff'] / base['profit_mean'], 1, sign=True)}, AUC {signed(w['auc_diff'], 4)}, "
                      f"tiempo {pct(saving[i], 0)} {'menos' if saving[i] >= 0 else 'más'}. En total, {n_all} de "
                      f"{len(ct) - 1} valores de r cumplen los tres criterios.")
            test_df = pd.DataFrame({
                'r': ct['r'].map(r_txt),
                '% contactado': ct['list_share'].map(lambda v: pct(v, 1)),
                'Ganancia promedio': ct['profit_mean'].map(pesos),
                'Diferencia de ganancia': [('—' if r == 1 else f"{pesos(d, 2)} ({pct(d / base['profit_mean'], 1, sign=True)})")
                                           for r, d in zip(ct['r'], ct['profit_diff'])],
                '¿Ganancia en tolerancia?': [('referencia' if r == 1 else status(ok, 'Sí', 'No')) for r, ok in zip(ct['r'], p_ok)],
                'AUC': ct['auc_mean'].map(lambda v: num(v, 4)),
                'Diferencia de AUC': [('—' if r == 1 else signed(d, 4)) for r, d in zip(ct['r'], ct['auc_diff'])],
                '¿AUC en tolerancia?': [('referencia' if r == 1 else status(ok, 'Sí', 'No')) for r, ok in zip(ct['r'], a_ok)],
                'Tiempo': ct['time_seconds'].map(dur),
                'Cambio de tiempo': [('—' if r == 1 else f"{pct(v, 0)} {'menos' if v >= 0 else 'más'}")
                                     for r, v in zip(ct['r'], saving)],
                f"¿Ahorra {pct(cfg['time_reduction'], 0)} o más?": [('referencia' if r == 1 else status(ok, 'Sí', 'No'))
                                                                   for r, ok in zip(ct['r'], t_ok)],
                '¿Cumple los tres?': [('referencia' if r == 1 else status(bool(ok), 'Sí', 'No'))
                                      for r, ok in zip(ct['r'], all_ok)],
            })
            table(test_df, highlight=[winner_r is not None and abs(r - winner_r) < 1e-9 for r in ct['r']])
            st.markdown(f'<div class="cap">El tiempo es el de cada r ({"su búsqueda + un entrenamiento" if grid_mode else "un entrenamiento"}), '
                        'el mismo que se usó para elegir. Esta tabla usa la mejor configuración de cada r; el veredicto usa las '
                        f"{test['top_k'] if test else cfg['top_k']} mejores del r elegido (pestañas Score y Tiempo), por eso "
                        'sus cifras difieren levemente.</div>', unsafe_allow_html=True)

            # ---- validación contra prueba, por valor de r
            vt = und[['r', 'diff', 'passes', 'winner']].copy()
            test_by_r = {round(r_, 6): d_ for r_, d_ in zip(ct['r'], ct['profit_diff'])}
            vt['test_diff'] = [test_by_r.get(round(r_, 6), np.nan) for r_ in vt['r']]
            vt = vt.dropna()
            section(f'Diferencia de ganancia en {VALID} y en {TEST}, por valor de r', color,
                    f'Cada punto es un valor de r con su mejor configuración: su diferencia de ganancia frente al modelo '
                    f'base en {VALID} (eje horizontal, {cfg["n_seeds_valid"]} semillas) y en {TEST} (eje vertical, '
                    f'{cfg["n_seeds"]} semillas). Los puntos por debajo de la diagonal obtuvieron en {TEST} un resultado '
                    f'peor que en {VALID}.')
            lo_ = float(min(vt['diff'].min(), vt['test_diff'].min(), 0)) / 1e6 - 0.3
            hi_ = float(max(vt['diff'].max(), vt['test_diff'].max(), 0)) / 1e6 + 0.3
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode='lines', hoverinfo='skip', showlegend=False,
                                     line=dict(color=INK_2, width=1, dash='dash')))
            fig.add_annotation(x=lo_, y=lo_, text='igual resultado en ambos meses', showarrow=False, xanchor='left',
                               yanchor='bottom', xshift=28, yshift=2, font=dict(color=INK_2, size=11))
            for ok_, name, c in [(False, 'No cumplió la regla', '#b9c2ce'), (True, 'Cumplió la regla', color)]:
                d_ = vt[vt['passes'] == ok_]
                fig.add_trace(go.Scatter(
                    x=d_['diff'] / 1e6, y=d_['test_diff'] / 1e6, mode='markers+text', name=name,
                    text=[('★ ' if w_ else '') + r_txt(r_) for r_, w_ in zip(d_['r'], d_['winner'])],
                    textposition='middle right', textfont=dict(size=11, color=INK if ok_ else MUTED),
                    marker=dict(size=[15 if w_ else 10 for w_ in d_['winner']], color=c if ok_ else '#ffffff',
                                line=dict(width=2, color=c)),
                    customdata=np.stack([d_['r'].map(r_txt)], axis=1),
                    hovertemplate='r = %{customdata[0]}<br>' + VALID + ': $ %{x:,.2f} M<br>' + TEST
                                  + ': $ %{y:,.2f} M<extra></extra>'))
            fig.add_vline(x=0, line=dict(color=AXIS, width=1))
            fig.add_hline(y=0, line=dict(color=AXIS, width=1))
            fig.update_xaxes(range=[lo_, hi_])
            fig.update_yaxes(range=[lo_, hi_])
            left_s, _ = st.columns([1.4, 1])
            with left_s:
                show(style(fig, f'Diferencia de ganancia frente al modelo base: {VALID} contra {TEST}',
                           f'Diferencia en {VALID} (millones de $)', f'Diferencia en {TEST} (millones de $)', height=540))
            caption('los valores que cumplieron la regla en validación están en color; la estrella marca el elegido. La '
                    'distancia vertical a la diagonal indica cuánto cambió el resultado entre ambos meses.')


# ======================= Ganancia y AUC (misma estructura, distinto color) =======================
def metric_view(metric, label, color, money):
    fmt = pesos if money else (lambda v: num(v, 4))
    if not test and not curve:
        st.info('Todavía no hay resultados en el mes de prueba (etapa "test").')
        return
    if test:
        m = test['metrics'][metric]
        wr = r_txt(test['winner_r'])
        decisive = metric in ('profit', 'auc')
        if any('values' not in row for row in test['per_config']):
            st.info('Este resultado se calculó con la versión anterior del análisis: correr la etapa "analyze" otra vez.')
            return
        sides = {}
        for side in ['baseline', 'winner']:
            frames = [pd.DataFrame({'seed': row['seeds'], f'test_{metric}': row['values'][metric], 'rank': row['rank']})
                      for row in test['per_config'] if row['side'] == side]
            sides[side] = pd.concat(frames) if frames else pd.DataFrame(columns=['seed', f'test_{metric}', 'rank'])
        k = test['top_k']

        section(f'r elegido contra el modelo base · {TEST}', color,
                f"Se comparan los **{k} mejores modelos** de cada lado (cada modelo es una configuración, y sus "
                f"**{cfg['n_seeds']} semillas** forman su distribución). La comparación es pareada: el 1.er modelo del r "
                'elegido contra el 1.º del modelo base con la misma semilla, el 2.º contra el 2.º, y así sucesivamente. Cada '
                'par mide **el efecto de aplicar r**; el promedio de esas diferencias determina el resultado.')
        rel = m['mean_diff'] / m['reference_mean']
        diff_txt = (signed(m['mean_diff'] / 1e6, 2) + '\u00a0M') if money else signed(m['mean_diff'], 4)
        ci_txt = (f"[{signed(m['ci_low'] / 1e6, 2)}; {signed(m['ci_high'] / 1e6, 2)}] M" if money
                  else f"[{signed(m['ci_low'], 4)}; {signed(m['ci_high'], 4)}]")
        left_k, right_k = st.columns([1, 2], gap='small')
        with left_k:
            kpis([('Efecto de aplicar r (determinante)' if decisive else 'Efecto de aplicar r', diff_txt,
                   f"promedio de {m['n_pairs']} comparaciones pareadas"
                   + (f" ({pct(rel, 1, sign=True)} de la ganancia del modelo base)" if money else '')
                   + f'<br>Intervalo del 95 %: {ci_txt}<br>'
                   + (status(m['non_inferior'], f'Dentro de la tolerancia ({pct(delta, 0)})', 'Fuera de la tolerancia')
                      if decisive else '<span class="na">Informativa: no interviene en el veredicto</span>'), color)], color)
        with right_k:
            rows = []
            for rank in range(1, k + 1):
                b = sides['baseline'][sides['baseline']['rank'] == rank].set_index('seed')[f'test_{metric}']
                w = sides['winner'][sides['winner']['rank'] == rank].set_index('seed')[f'test_{metric}']
                common = b.index.intersection(w.index)
                d = float((w.loc[common] - b.loc[common]).mean()) if len(common) else float('nan')
                rows.append([f'{rank}.ª mejor', fmt(b.mean()), fmt(w.mean()),
                             (pesos(d, 2) + f" ({pct(d / b.mean(), 1, sign=True)})") if money else signed(d, 4)])
            table(pd.DataFrame(rows, columns=['Configuración', f"Modelo base ({cfg['n_seeds']} semillas)",
                                              f"r elegido ({cfg['n_seeds']} semillas)", 'Efecto de aplicar r']))
            st.markdown('<div class="cap">Cada fila compara una configuración con su par, cada una promediando sus '
                        f"{cfg['n_seeds']} semillas.</div>", unsafe_allow_html=True)
        if decisive:
            concl('good' if m['non_inferior'] else 'bad',
                  f"Con r = {wr}, {'el AUC' if metric == 'auc' else 'la ' + label.lower()} {'se mantiene dentro de la tolerancia' if m['non_inferior'] else 'disminuye más que la tolerancia'}"
                  f": la diferencia promedio es de {diff_txt} frente al modelo base.")

        section('Distribución de los resultados', color,
                f"Cada configuración entrenada con {cfg['n_seeds']} semillas, lo que permite dimensionar la variabilidad atribuible al azar.")
        left, right = st.columns(2, gap='small')
        with left:
            fig = go.Figure()
            for side, c, name in [('baseline', NAVY, 'Modelo base'), ('winner', color, f'r elegido ({wr})')]:
                d = sides[side]
                fig.add_trace(go.Violin(
                    x=d['rank'].map(lambda r_: f'{r_}.ª mejor'), y=d[f'test_{metric}'], name=name,
                    line=dict(color=c, width=1.5), offsetgroup=side, scalegroup=side,
                    fillcolor=rgba(c, 0.12), points='all', pointpos=0, jitter=0.35,
                    marker=dict(color=c, size=6, opacity=0.8, line=dict(width=1, color='#fff')),
                    box=dict(visible=True, width=0.12, line=dict(color=c, width=1.5), fillcolor='#fff'),
                    meanline=dict(visible=True, color=c), spanmode='hard',
                    customdata=np.stack([d['rank'], d['seed']], axis=1),
                    hovertemplate=('%{customdata[0]}.ª mejor configuración<br>'
                                   + ('$ %{y:,.0f}' if money else '%{y:.4f}') + '<extra>' + name + '</extra>')))
            fig.update_layout(violinmode='group')
            show(style(fig, f'{label} de cada entrenamiento, por configuración', 'Configuración', label))
            caption('cada punto es un entrenamiento (una semilla); la forma indica dónde se concentran los valores y la línea, el '
                    'promedio de esa configuración.')
        with right:
            b = sides['baseline'].set_index(['rank', 'seed'])[f'test_{metric}']
            w = sides['winner'].set_index(['rank', 'seed'])[f'test_{metric}']
            common = b.index.intersection(w.index)
            diff = (w.loc[common] - b.loc[common]).to_numpy()
            fig = go.Figure()
            fig.add_vrect(x0=-m['margin'], x1=max(float(diff.max()) if len(diff) else 0, m['margin']) * 1.15 + 1e-12,
                          fillcolor=rgba(GOOD, 0.05), line_width=0)
            fig.add_trace(go.Histogram(x=diff, nbinsx=24, marker=dict(color=rgba(color, 0.55), line=dict(color='#fff', width=1)),
                                       name='Pares', hovertemplate='%{y} pares<extra></extra>'))
            fig.add_vline(x=0, line=dict(color=AXIS, width=1))
            fig.add_vline(x=-m['margin'], line=dict(color=INK_2, width=1, dash='dot'),
                          annotation_text=f'tolerancia (−{pct(delta, 0)})', annotation_position='top left',
                          annotation_font=dict(color=INK_2, size=11))
            fig.add_vline(x=m['mean_diff'], line=dict(color=color, width=2.5),
                          annotation_text='promedio', annotation_position='bottom right',
                          annotation_font=dict(color=INK, size=11))
            fig.add_vrect(x0=m['ci_low'], x1=m['ci_high'], fillcolor=rgba(color, 0.18), line_width=0)
            fig.update_layout(bargap=0.05)
            show(style(fig, 'Diferencias pareadas (r elegido − modelo base)',
                       'Diferencia' + (' en $' if money else ' de AUC'), 'Cantidad de pares', legend_on=False))
            caption('cada par es la misma configuración y la misma semilla en los dos modelos. La franja de color es el '
                    'intervalo del 95 % del promedio: si queda a la derecha de la tolerancia, el resultado se considera equivalente.')

    if curve:
        tab = pd.DataFrame(curve['table']).sort_values('r')
        section(f'{label} según r · {TEST}', color,
                'Cada punto es la mejor configuración encontrada para ese r, entrenada con '
                f"{cfg['n_seeds']} semillas. De izquierda a derecha, cada árbol usa menos CONTINUA.")
        base_mean = float(tab.loc[tab['r'] == 1, f'{metric}_mean'].iloc[0])
        fig = go.Figure()
        fig.add_hrect(y0=base_mean * (1 - delta), y1=base_mean * (1 + delta), fillcolor=BAND, line_width=0)
        fig.add_hline(y=base_mean, line=dict(color=NAVY, width=1.5),
                      annotation_text=f'modelo base: {fmt(base_mean)}', annotation_position='bottom left',
                      annotation_font=dict(color=NAVY, size=11))
        if curve['useful_zone']:
            lo, hi = curve['useful_zone_range']
            fig.add_vrect(x0=lo, x1=hi, fillcolor=ZONE, line_width=0)
            fig.add_annotation(x=np.log10(hi), y=0.98, xref='x', yref='paper', text='zona útil', showarrow=False,
                               xanchor='left', yanchor='top', xshift=4, font=dict(color=GOOD, size=12))
        fig.add_trace(go.Scatter(x=pd.concat([tab['r'], tab['r'][::-1]]),
                                 y=pd.concat([tab[f'{metric}_max'], tab[f'{metric}_min'][::-1]]), fill='toself',
                                 fillcolor=rgba(color, 0.15), line=dict(width=0), hoverinfo='skip',
                                 name='Rango entre semillas'))
        fig.add_trace(go.Scatter(x=tab['r'], y=tab[f'{metric}_mean'], mode='lines+markers', name='Promedio de las semillas',
                                 line=dict(color=color, width=2.5), marker=dict(size=8, color=color, line=dict(width=2, color='#fff')),
                                 customdata=np.stack([tab['r'].map(r_txt), tab['n_trials_with_r']], axis=1),
                                 hovertemplate='r = %{customdata[0]}: ' + ('$ %{y:,.0f}' if money else '%{y:.4f}')
                                               + '<br>%{customdata[1]} pruebas de Optuna con este r<extra></extra>'))
        if winner_r is not None:
            mark_r(fig, winner_r, f'r elegido ({r_txt(winner_r)})', color)
        show(r_axis(style(fig, f'{label} según r (la franja gris es ±{pct(delta, 0)} del modelo base)', '', label, height=420)))
        hold = curve.get('holding_r', [])
        held = ('ningún r mantiene a la vez la ganancia y el AUC dentro de la tolerancia' if not hold else
                f"mantienen la ganancia y el AUC r = {', '.join(r_txt(r) for r in sorted(hold, reverse=True))}")
        caption(f"la línea azul marino es el modelo base; {held}"
                + (f". La zona útil va de r = {r_txt(curve['useful_zone_range'][0])} a "
                   f"r = {r_txt(curve['useful_zone_range'][1])}." if curve['useful_zone'] else '.'))


with tabs[T_METRICS]:
    page_head('Score', 'Ganancia y AUC, con y sin submuestreo',
              f'Las dos métricas que determinan si el resultado es equivalente, evaluadas en {TEST}, el mes de prueba. Cada una '
              f'debe mantenerse dentro de la tolerancia ({pct(delta, 0)}) frente al modelo base.')
    block('Ganancia', ACCENT['ganancia'])
    st.markdown(f"""<div class="card"><h3>Cómo se calcula la ganancia de un modelo</h3>
<p>Un <b>modelo</b> es una configuración de hiperparámetros; sus semillas forman su distribución. Contactar a un
<b>BAJA+2</b> suma <b>$ 1.072.500</b>; contactar a cualquier otro cliente (CONTINUA o BAJA+1) resta <b>$ 27.500</b>.</p><ol>
<li><b>En {VALID}:</b> cada una de sus {cfg['n_seeds_valid']} semillas tiene una curva de ganancia (ganancia contactando
al 0,1 %, 0,2 %, … de los clientes, ordenados de mayor a menor probabilidad). Se promedian las curvas; el porcentaje
donde la curva promedio es máxima es el <b>tamaño de la lista</b> del modelo. Queda fijo: {TEST} no puede cambiarlo.</li>
<li><b>En {TEST}:</b> cada una de sus {cfg['n_seeds']} semillas ordena a los clientes y contacta a su propio porcentaje
superior, con ese mismo porcentaje; se cuentan los BAJA+2 reales y se calcula la ganancia.</li>
<li>Eso da {cfg['n_seeds']} ganancias: la <b>distribución del modelo</b> en {TEST}.</li>
<li>El modelo base hace lo mismo con su propio porcentaje, y ambas distribuciones se comparan semilla por semilla.</li>
</ol><p style="margin-bottom:0;color:{MUTED};font-size:13.5px">El AUC no usa ningún porcentaje. La ganancia máxima (el
punto más alto de la curva, conociendo el resultado) se muestra solo como referencia. El modelo se entrena para
detectar BAJA+1 y BAJA+2, pero la ganancia solo premia a los BAJA+2.</p></div>""", unsafe_allow_html=True)

    if test and all('values' in row for row in test['per_config']):
        color_g = ACCENT['ganancia']
        section(f'Los {test["top_k"]} mejores modelos de cada lado · {TEST}', color_g,
                f"Ganancia de cada modelo en {TEST}, con su porcentaje de clientes fijado en {VALID}. Cada caja resume "
                f"sus {cfg['n_seeds']} semillas.")
        wr_txt = r_txt(test['winner_r'])
        boxes, rows = [], []
        for row in sorted(test['per_config'], key=lambda r_: (r_['side'] != 'baseline', r_['rank'])):
            name = (f"Base · {row['rank']}.º" if row['side'] == 'baseline' else f"r = {wr_txt} · {row['rank']}.º")
            vals = np.array(row['values']['profit'])
            boxes.append((name, vals, NAVY if row['side'] == 'baseline' else color_g))
            tr = trials.get(row['study'])
            hp = tr[tr['trial'] == row['trial']].iloc[0] if tr is not None and (tr['trial'] == row['trial']).any() else None
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            rows.append([name] + ([str(int(hp['num_leaves'])), hp_txt(round(hp['learning_rate'], 4)),
                                   str(int(hp['min_data_in_leaf'])), hp_txt(round(hp['feature_fraction'], 2)),
                                   (hp_txt(round(hp['bagging_fraction'], 2)) if 'bagging_fraction' in hp else '—')]
                                  if hp is not None else ['—'] * 5)
                        + [pct(row['list_share'], 1), pesos(vals.mean()), pesos(med), f'{pesos(q1)} – {pesos(q3)}',
                           num(np.mean(row['values']['auc']), 4)])
        fig = go.Figure()
        for name, vals, c in boxes:
            fig.add_trace(go.Box(y=vals / 1e6, name=name, marker=dict(color=c, size=6, opacity=0.75),
                                 line=dict(color=c, width=1.5), fillcolor=rgba(c, 0.12), boxpoints='all', jitter=0.35,
                                 pointpos=0, boxmean=True, hovertemplate=name + ': $ %{y:.2f} M<extra></extra>'))
        show(style(fig, f'Ganancia de cada modelo en {TEST} (cada punto es una semilla)', '', 'Millones de $',
                   legend_on=False, height=400))
        table(pd.DataFrame(rows, columns=['Modelo', 'Hojas', 'Tasa de aprendizaje', 'Mín. clientes por hoja',
                                          'Fracción de variables', 'Fracción de clientes', '% contactado',
                                          'Ganancia (media)', 'Mediana', 'Q1 – Q3', 'AUC (media)']))
        caption('la caja va del primer al tercer cuartil; la línea es la mediana y la línea punteada, la media. "% '
                f'contactado" es el tamaño de la lista de cada modelo, fijado en {VALID} con el promedio de sus '
                f"{cfg['n_seeds_valid']} semillas.")

    fl = summary.get('final_list')
    if fl:
        section(f'Lista final de clientes · {TEST}', ACCENT['ganancia'],
                'La lista que se entregaría: la mejor configuración de cada lado, con sus puntajes de '
                f"{TEST} promediados sobre sus {fl['n_seeds']} semillas, contacta su porcentaje de clientes fijado en "
                f'{VALID}. La ganancia se calcula con las clases reales de {TEST}.')
        items = []
        for side_key, title, c in [('baseline', 'Modelo base', NAVY),
                                   ('winner', f"r elegido ({r_txt(fl['sides'].get('winner', {}).get('r'))})",
                                    ACCENT['ganancia'])]:
            sd = fl['sides'].get(side_key)
            if sd:
                items.append((title, pesos(sd['profit']),
                              f"{num(sd['n_contacted'], 0)} clientes ({pct(sd['share'], 1)}); "
                              f"{sd['baja2_contacted']} de {sd['baja2_total']} BAJA+2<br>"
                              f"Máximo posible: {pesos(sd['best_profit'])} contactando al {pct(sd['best_share'], 1)}", c))
        kpis(items, ACCENT['ganancia'])
        cols_dl = st.columns(len(items))
        for col_, side_key in zip(cols_dl, [k_ for k_ in ['baseline', 'winner'] if k_ in fl['sides']]):
            path_ = RESULTS / key / f'final_list_{side_key}.csv'
            if path_.exists():
                with col_:
                    st.download_button(f"Descargar lista ({'modelo base' if side_key == 'baseline' else 'r elegido'})",
                                       path_.read_bytes(), file_name=path_.name, mime='text/csv')
        caption('el máximo posible se calcula conociendo el resultado y solo sirve de referencia. La comparación que '
                f"define el veredicto es la de las distribuciones de los {cfg['top_k']} mejores modelos de cada lado.")

    which = st.radio('Medida de ganancia', ['profit', 'profit_peak'], horizontal=True,
                     format_func={'profit': 'Ganancia con el porcentaje fijado en ' + VALID + ' (define el veredicto)',
                                  'profit_peak': 'Ganancia máxima de la curva (referencia)'}.get)
    metric_view(which, 'Ganancia' if which == 'profit' else 'Ganancia máxima', ACCENT['ganancia'], money=True)
    block('AUC', ACCENT['auc'],
         'Mide la capacidad del modelo para ordenar a los clientes según su probabilidad de baja (BAJA+1 y BAJA+2): 1 '
         'corresponde a un orden perfecto y 0,5, al azar. No depende de la cantidad de clientes contactados.')
    metric_view('auc', 'AUC', ACCENT['auc'], money=False)

# ======================= Tiempo =======================
with tabs[T_TIME]:
    color = ACCENT['tiempo']
    page_head('Tiempo', 'Reducción del tiempo',
              'El tiempo total suma la preparación de los datos, la búsqueda de hiperparámetros, un entrenamiento del '
              'modelo final y la predicción. Todas las mediciones se realizaron en el mismo equipo.')
    if test:
        t = test['time']
        exp = summary.get('exploration', {})
        section('El ahorro', color)
        kpis([('Ahorro total', pct(t['reduction'], 0),
               status(t['meets_threshold'], f"Al menos {pct(cfg['time_reduction'], 0)}", 'Inferior a lo exigido'), color),
              ('En la búsqueda', pct(t['search_only_reduction'], 0),
               f"{dur(t['winner']['search_seconds'])} contra {dur(t['baseline']['search_seconds'])}", color),
              ('En el modelo final', pct(t['final_train_reduction'], 0),
               f"{dur(t['winner']['final_train_seconds'])} contra {dur(t['baseline']['final_train_seconds'])}", color),
              ('Exploración completa', dur(exp.get('total_seconds', 0)),
               f"todas las búsquedas y {num(exp.get('n_trainings', 0), 0)} entrenamientos; costo único, excluido "
               'de la comparación', NAVY)], color)
        concl('good' if t['meets_threshold'] else 'bad',
              f"Con r = {r_txt(test['winner_r'])}, el proceso completo tarda {pct(abs(t['reduction']), 0)} "
              f"{'menos' if t['reduction'] >= 0 else 'más'}: {dur(t['winner']['total_seconds'])} contra "
              f"{dur(t['baseline']['total_seconds'])}.")

        section('Composición del tiempo', color)
        parts = [('prep_seconds', 'Preparación de datos'), ('search_seconds', 'Búsqueda de hiperparámetros'),
                 ('final_train_seconds', 'Entrenamiento del modelo final'), ('predict_seconds', 'Predicción')]
        names = ['Modelo base', f"r elegido ({r_txt(test['winner_r'])})"]
        scale, unit = (3600, 'Horas') if t['baseline']['total_seconds'] >= 3 * 3600 else (60, 'Minutos')
        fig = go.Figure()
        for (part, label), c in zip(parts, TIME_PARTS):
            fig.add_trace(go.Bar(y=names, x=[t[s][part] / scale for s in ['baseline', 'winner']], name=label,
                                 orientation='h', marker=dict(color=c, line=dict(width=2, color='#fff')),
                                 customdata=[dur(t[s][part]) for s in ['baseline', 'winner']],
                                 hovertemplate='%{y} · ' + label + ': %{customdata}<extra></extra>'))
        for name, s in zip(names, ['baseline', 'winner']):
            fig.add_annotation(x=t[s]['total_seconds'] / scale, y=name, text=f"<b>{dur(t[s]['total_seconds'])}</b>",
                               showarrow=False, xanchor='left', xshift=8, font=dict(color=INK, size=13))
        fig.update_layout(barmode='stack', bargap=0.5)
        fig.update_yaxes(autorange='reversed')
        show(style(fig, 'Tiempo total por etapa', unit, '', height=280))
        caption('la búsqueda de hiperparámetros explica casi todo el tiempo. El modelo final se computa una sola vez (el promedio '
                'de sus entrenamientos): las semillas se utilizan para medir y no se suman.')

    if curve:
        tab = pd.DataFrame(curve['table']).sort_values('r')
        section('Evolución del tiempo según r', color,
                'Reducir r disminuye el tiempo hasta cierto punto; a partir de allí el tiempo se estabiliza y seguir reduciendo r solo pone en riesgo la '
                'ganancia.')
        left, right = st.columns(2, gap='small')
        with left:
            base_t = float(curve['baseline_seconds'])
            scale, unit = (3600, 'Horas') if base_t >= 3 * 3600 else (60, 'Minutos')
            fig = go.Figure()
            fig.add_hline(y=base_t / scale, line=dict(color=NAVY, width=1.5), annotation_text=f'modelo base: {dur(base_t)}',
                          annotation_position='top left', annotation_font=dict(color=NAVY, size=11))
            fig.add_trace(go.Scatter(x=tab['r'], y=tab['time_seconds'] / scale, mode='lines+markers', name='Tiempo',
                                     line=dict(color=color, width=2.5), fill='tozeroy', fillcolor=rgba(color, 0.10),
                                     marker=dict(size=8, color=color, line=dict(width=2, color='#fff')),
                                     customdata=np.stack([tab['r'].map(r_txt), tab['time_seconds'].map(dur)], axis=1),
                                     hovertemplate='r = %{customdata[0]}: %{customdata[1]}<extra></extra>'))
            if curve['time_flattens']:
                mark_r(fig, curve['r_flat'], f"se estabiliza desde r = {r_txt(curve['r_flat'])}", INK_2, top=True)
            if winner_r is not None:
                mark_r(fig, winner_r, f'r elegido ({r_txt(winner_r)})', color)
            show(r_axis(style(fig, 'Tiempo según r' + (' (búsqueda + entrenamiento)' if grid_mode else ' (entrenamiento)'),
                              '', unit, legend_on=False), short=True))
        with right:
            if trials:
                allt = pd.concat([df.assign(study=s) for s, df in trials.items()])
                fig = go.Figure()
                for is_base, c, name in [(False, color, 'Con submuestreo'), (True, NAVY, 'Modelo base')]:
                    d = allt[(allt['study'] == 'baseline') == is_base]
                    fig.add_trace(go.Box(x=d['r'].map(r_txt), y=d['seconds_per_tree'] * 1000, name=name,
                                         marker=dict(color=c, size=4, opacity=0.6), line=dict(color=c, width=1.5),
                                         fillcolor=rgba(c, 0.12), boxpoints='outliers',
                                         hovertemplate='r = %{x}: %{y:,.1f} ms por árbol<extra></extra>'))
                order = [r_txt(r) for r in sorted(allt['r'].unique(), reverse=True)]
                fig.update_xaxes(type='category', categoryorder='array', categoryarray=order)
                show(style(fig, 'Tiempo por árbol en las búsquedas', 'r', 'Milisegundos por árbol'))
        caption('a la izquierda, el tiempo con el que se compara cada r. A la derecha, el tiempo por árbol en todas las '
                'pruebas de Optuna: aísla el efecto de r, ya que no depende de la cantidad de árboles de cada configuración.')

# ======================= Búsquedas de Optuna =======================
with tabs[T_SEARCH]:
    color = ACCENT['busquedas']
    page_head('Búsquedas de Optuna', 'Búsqueda de hiperparámetros',
              'Cada **prueba** de Optuna entrena el modelo con una combinación de hiperparámetros y lo evalúa en el mes de '
              f"{VALID} por su AUC. Cada búsqueda hizo {cfg['n_trials']} pruebas.")
    space = {k: v for k, v in cfg['search_space'].items() if k != 'bagging_fraction' or cfg.get('bagging_baseline', True)}
    section('El espacio de búsqueda', color,
            'Hiperparámetros que ajusta Optuna y rango de búsqueda de cada uno. Es el mismo para el modelo base y '
            'para cada valor de r, de modo que ninguno dispone de mayor margen.')
    table(pd.DataFrame([[HP_INFO.get(k, (k, ''))[0], HP_INFO.get(k, (k, ''))[1], f'{hp_txt(lo)} a {hp_txt(hi)}',
                         ('entero' if kind == 'int' else 'decimal') + (', escala logarítmica' if log_ else ', escala lineal')]
                        for k, (lo, hi, kind, log_) in space.items()],
                       columns=['Hiperparámetro', 'Qué controla', 'Rango', 'Tipo']), text=True)
    caption(f"parámetros fijos en todas las búsquedas: hasta {num(cfg['max_rounds'], 0)} árboles, con parada temprana si el AUC de "
            f"{VALID} no mejora en {cfg['early_stopping_rounds']} árboles consecutivos. En escala logarítmica, Optuna explora "
            'con la misma intensidad el tramo entre 10 y 100 que el tramo entre 100 y 1000.')
    if not trials:
        st.info('Todavía no hay pruebas de Optuna (etapa "search").')
    else:
        study = st.selectbox('Búsqueda', list(trials), format_func=study_label)
        d = trials[study].sort_values('trial')
        c = NAVY if study == 'baseline' else color
        section(study_label(study), c)
        best = d.loc[d['valid_auc_score'].idxmax()]
        kpis([('Mejor AUC de validación', num(best['valid_auc_score'], 4), f"prueba {int(best['trial'])}", c),
              ('Duración de la búsqueda', dur(d['trial_seconds'].sum()), f"{len(d)} pruebas", c),
              ('Duración mediana de una prueba', dur(d['trial_seconds'].median()), 'mediana', c)], c)
        left, right = st.columns(2, gap='small')
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d['trial'], y=d['valid_auc_score'], mode='markers', name='Cada prueba',
                                     marker=dict(color=rgba(c, 0.55), size=7, line=dict(width=1, color='#fff')),
                                     customdata=d['r'].map(r_txt),
                                     hovertemplate='prueba %{x}: AUC %{y:.4f} · r = %{customdata}<extra></extra>'))
            fig.add_trace(go.Scatter(x=d['trial'], y=d['valid_auc_score'].cummax(), mode='lines', name='Mejor valor acumulado',
                                     line=dict(color=c, width=2.5, shape='hv'), hoverinfo='skip'))
            show(style(fig, 'AUC de validación en cada prueba', 'Número de prueba', 'AUC'))
        with right:
            if d['r'].nunique() > 1:
                fig = go.Figure(go.Box(x=d['r'].map(r_txt), y=d['valid_auc_score'], boxpoints='all', jitter=0.3, pointpos=0,
                                       line=dict(color=c, width=1.5), fillcolor=rgba(c, 0.10),
                                       marker=dict(color=c, size=6, opacity=0.7), name='Pruebas',
                                       hovertemplate='r = %{x}: AUC %{y:.4f}<extra></extra>'))
                fig.update_xaxes(type='category', categoryorder='array',
                                 categoryarray=[r_txt(r) for r in sorted(d['r'].unique(), reverse=True)])
                show(style(fig, 'AUC según el r que probó Optuna', 'r', 'AUC', legend_on=False))
            else:
                fig = go.Figure(go.Bar(x=d['trial'], y=d['trial_seconds'], name='Duración',
                                       marker=dict(color=rgba(c, 0.6), line=dict(width=0)),
                                       customdata=d['trees_built'],
                                       hovertemplate='prueba %{x}: %{y:,.1f} s · %{customdata} árboles<extra></extra>'))
                show(style(fig, 'Duración de cada prueba', 'Número de prueba', 'Segundos', legend_on=False))
        caption('la línea escalonada marca el mejor AUC encontrado hasta cada prueba: cuando se estabiliza, prolongar la búsqueda '
                'ya no mejora el modelo.')
        section('Exploración de Optuna dentro de cada rango', c,
                'Cada punto es una prueba: su valor del hiperparámetro (horizontal, sobre todo el rango permitido) y el AUC '
                'obtenido (vertical). La estrella es la mejor prueba.')
        names_hp = [k for k in space if k in d]
        fig = make_subplots(rows=1, cols=len(names_hp), shared_yaxes=True, horizontal_spacing=0.025,
                            subplot_titles=[HP_SHORT.get(k, k) for k in names_hp])
        top_auc = d['valid_auc_score']
        for i, k in enumerate(names_hp, start=1):
            lo, hi, kind, log_ = space[k]
            fig.add_trace(go.Scatter(x=d[k], y=top_auc, mode='markers', showlegend=False,
                                     marker=dict(color=rgba(c, 0.55), size=7, line=dict(width=1, color='#fff')),
                                     customdata=d['trial'],
                                     hovertemplate='prueba %{customdata}<br>' + HP_INFO.get(k, (k,))[0]
                                                   + ': %{x:.3~f}<br>AUC %{y:.4f}<extra></extra>'), row=1, col=i)
            fig.add_trace(go.Scatter(x=[best[k]], y=[best['valid_auc_score']], mode='markers', showlegend=False,
                                     marker=dict(color=c, size=15, symbol='star', line=dict(width=1.5, color='#fff')),
                                     hovertemplate='mejor prueba: %{x:.3~f}<extra></extra>'), row=1, col=i)
            pad = (np.log10(hi) - np.log10(lo)) * 0.04 if log_ else (hi - lo) * 0.04
            rng = [np.log10(lo) - pad, np.log10(hi) + pad] if log_ else [lo - pad, hi + pad]
            fig.update_xaxes(type='log' if log_ else 'linear', range=rng, row=1, col=i,
                             **(log_ticks(lo, hi) if log_ else {}))
            for edge in (lo, hi):
                fig.add_vline(x=edge, line=dict(color=AXIS, width=1, dash='dot'), row=1, col=i)
        style(fig, '', '', '', height=330, legend_on=False)
        fig.update_yaxes(title=dict(text='AUC de validación', font=dict(color=INK_2, size=12)), row=1, col=1)
        fig.update_layout(margin=dict(l=12, r=12, t=40, b=12))
        fig.update_annotations(font=dict(size=12, color=NAVY_DARK))
        show(fig)
        caption('las líneas punteadas son los bordes del rango. Si las mejores pruebas se concentran contra un borde, el rango '
                'podría resultar insuficiente de ese lado; si se distribuyen en el interior, el rango es adecuado. Optuna comienza con pruebas '
                'aleatorias y luego se concentra en las regiones con mejores resultados, por eso los puntos se agrupan.')
        section('Detalle de las pruebas', c)
        cols = {'trial': 'Prueba', 'valid_auc_score': 'AUC', 'r': 'r',
                'num_leaves': 'Hojas', 'learning_rate': 'Tasa de aprendizaje', 'min_data_in_leaf': 'Mín. de clientes por hoja',
                'feature_fraction': 'Fracción de variables', 'bagging_fraction': 'Fracción de clientes (bagging)',
                'best_iter': 'Árboles del modelo', 'trees_built': 'Árboles construidos', 'trial_seconds': 'Segundos'}
        st.dataframe(d[[k for k in cols if k in d]].rename(columns=cols), **FULL, hide_index=True)

# ======================= Metodología y ejecución =======================
with tabs[T_RUN]:
    page_head('Metodología y ejecución', 'Condiciones del experimento',
              'Datos, meses y criterios con los que se evaluó la hipótesis, e información técnica para reproducir o '
              'auditar el experimento.')
    block('Condiciones de ejecución', NAVY)
    table(pd.DataFrame([
        ['Datos', BASES.get(summary['base'], summary['base'])],
        ['Elección de r', MODES_LONG[summary['r_mode']]],
        ['Aplicación de r', APPLY[summary['r_apply']]],
        ['Entrenamiento', f"{month(cfg['train_months'][0])} a {month(cfg['train_months'][-1])}"],
        ['Validación (para elegir r)', VALID],
        ['Prueba (para confirmar)', TEST],
        ['Pruebas de Optuna por búsqueda', num(cfg['n_trials'], 0)],
        ['Semillas', f"{cfg['n_seeds_valid']} para elegir, {cfg['n_seeds']} para confirmar"],
        ['Medida de ganancia', f"ganancia en el porcentaje de clientes de cada modelo: el máximo de su curva promedio "
                               f"de {cfg['n_seeds_valid']} semillas en {VALID}, aplicado a sus {cfg['n_seeds']} semillas en {TEST}"],
        ['Tolerancia en ganancia y AUC', pct(delta, 0)],
        ['Ahorro de tiempo exigido', pct(cfg['time_reduction'], 0)],
    ], columns=['Dato', 'Valor']), text=True)

    block('Detalles técnicos', NAVY)
    env, timings = summary.get('environment', {}), summary.get('timings', {})
    ds = cfg['derived_seeds']
    section('Máquina y corrida', NAVY)
    table(pd.DataFrame([
        ['Máquina', f"{env.get('platform')} · {env.get('cpu_count')} núcleos · {num(env.get('ram_gb', 0), 0)} GB de RAM"],
        ['Pico de memoria', f"{num(timings.get('peak_memory_gb', 0), 1)} GB"],
        ['Variables del modelo', num(timings.get('n_features', 0), 0)],
        ['Valores de r probados', ', '.join(r_txt(r) for r in sorted(cfg['r_list'], reverse=True)) + ' (y r = 1, el modelo base)'],
        ['Semilla maestra', str(cfg['master_seed'])],
        ['Semillas derivadas', f"Optuna {ds['optuna']} · dentro de las búsquedas {ds['search']} · modelos finales "
                               f"{', '.join(map(str, seeds[:3]))}, …"],
    ], columns=['Dato', 'Valor']), text=True)
    section('Configuración completa', NAVY)
    st.json(cfg, expanded=False)
