import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import mysql.connector
from lib.lib import consulta

# ----------------------
# Colores de la marca
# ----------------------
PRIMARY_COLOR   = "#4ACDCE"
SECONDARY_COLOR = "#E044A7"
TERTIARY_COLOR  = "#5266B0"

# -- Cache de carga de datos --
@st.cache_data
def load_data():
    neg = consulta(st.secrets["query_negocios"]["query1"])
    tran = consulta(st.secrets["query_transacciones"]["query2"])
    return pd.merge(
        neg, tran,
        left_on="id", right_on="NegocioId",
        how="inner"
    )

# --- Carga y preprocesamiento ---
df = load_data()

# Fechas originales
df['fecha_afiliacion'] = (
    pd.to_datetime(df['fechaAfiliacion'])
      .fillna(pd.to_datetime(df['createdAt']))
)
df['fecha_transaccion'] = pd.to_datetime(df['fecha_transaccion'])

# Cálculo del inicio de semana (lunes)
df['semana_afiliacion'] = (
    df['fecha_afiliacion']
    - pd.to_timedelta(df['fecha_afiliacion'].dt.weekday, unit='d')
)
df['semana_transaccion'] = (
    df['fecha_transaccion']
    - pd.to_timedelta(df['fecha_transaccion'].dt.weekday, unit='d')
)

# Definimos cohort semanal usando la fecha de lunes
df['periodo_affiliacion'] = df['semana_afiliacion'].dt.strftime('%Y-%m-%d')

# Calculamos la semana relativa desde la afiliación
df['semana_relativa'] = (
    (df['semana_transaccion'] - df['semana_afiliacion'])
    / pd.Timedelta(days=7)
).astype(int) + 1
df = df[df['semana_relativa'] >= 1]

# Para asegurar que tenemos todas las semanas hasta la máxima
def get_all_weeks(df):
    max_w = df['semana_relativa'].max()
    return pd.DataFrame({'semana_relativa': range(1, max_w+1)})

all_weeks = get_all_weeks(df)

# --- Sidebar de filtros ---
st.sidebar.header("Filtros de cohorte y actividad")
periods = sorted(df['periodo_affiliacion'].unique())
selected_periods = st.sidebar.multiselect(
    "Selecciona periodo(s) de afiliación:", periods, default=[]
)
min_trx    = st.sidebar.slider("Mín. transacciones por semana:", 1, 10, 1)
min_amount = st.sidebar.number_input(
    "Monto mínimo por transacción:", 
    min_value=0.0, value=80.0, step=10.0
)

df['transaccion_valida'] = df['monto'] >= min_amount

# --- Título y logo ---
st.title("Curvas de Retención Semanal")
st.image("logo_nuevo.png", width=100)

# ==============================
# Comparación de cohorts
# ==============================
if selected_periods:
    retention_all = []
    cohort_sizes = df.groupby('periodo_affiliacion')['id'].nunique().to_dict()
    for coh in selected_periods:
        df_c = df[df['periodo_affiliacion'] == coh]
        size = cohort_sizes.get(coh, 0)
        wk = (
            df_c[df_c['transaccion_valida']]
            .groupby(['id','semana_relativa'])['id_transaccion']
            .count()
            .reset_index(name='trans_validas')
        )
        wk['activo'] = wk['trans_validas'] >= min_trx
        temp = (
            wk[wk['activo']]
            .drop_duplicates(['id','semana_relativa'])
            .groupby('semana_relativa')['id']
            .nunique()
            .reset_index(name='neg_activos')
        )
        temp['periodo_affiliacion'] = coh
        temp['retencion_%'] = temp['neg_activos'] / size * 100
        retention_all.append(temp)
    df_compare = (
        pd.concat(retention_all, ignore_index=True)
        if retention_all else pd.DataFrame()
    )

    fig_compare = px.line(
        df_compare,
        x='semana_relativa', y='retencion_%',
        color='periodo_affiliacion',
        markers=True,
        color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR],
        labels={'semana_relativa':'Semana','retencion_%':'Retención (%)'}
    )
    fig_compare.update_layout(
        title="Comparación de Cohortes",
        yaxis=dict(range=[0,100]),
        font_color=PRIMARY_COLOR
    )
    st.plotly_chart(fig_compare, use_container_width=True)
else:
    st.info("Selecciona al menos una cohorte para comparar.")

# ==============================
# Detalle de un solo cohort
# ==============================
if selected_periods:
    cohort = selected_periods[0]
    df_coh = df[df['periodo_affiliacion'] == cohort]
    size = df_coh['id'].nunique()
    wk = (
        df_coh[df_coh['transaccion_valida']]
        .groupby(['id','semana_relativa'])['id_transaccion']
        .count()
        .reset_index(name='trans_validas')
    )
    wk['activo'] = wk['trans_validas'] >= min_trx
    temp = (
        wk[wk['activo']]
        .drop_duplicates(['id','semana_relativa'])
        .groupby('semana_relativa')['id']
        .nunique()
        .reset_index(name='neg_activos')
    )
    weekly_counts = (
        all_weeks.merge(temp, on='semana_relativa', how='left')
        .fillna({'neg_activos':0})
    )
    weekly_counts['retencion_%'] = weekly_counts['neg_activos'] / size * 100

    st.subheader(f"Datos Cohorte {cohort}")
    st.dataframe(
        weekly_counts.style.format({
            'retencion_%':'{:.2f}%', 
            'neg_activos':'{:.0f}'
        })
    )

# ==============================
# Matriz y heatmap de retención
# ==============================
sizes = df.groupby('periodo_affiliacion')['id'].nunique().to_dict()
# Filtrar cohorts desde enero 2024
filtered_cohorts = sorted([
    c for c in sizes 
    if pd.to_datetime(c) >= pd.to_datetime('2024-01-01')
])
ret_all = []
max_weeks = 0

for coh in filtered_cohorts:
    size = sizes.get(coh, 0)
    if size == 0:
        continue
    df_c = df[df['periodo_affiliacion'] == coh]
    wk = (
        df_c[df_c['transaccion_valida']]
        .groupby(['id','semana_relativa'])
        .agg(trans_validas=('id_transaccion','count'))
        .reset_index()
    )
    wk['activo'] = wk['trans_validas'] >= min_trx
    temp = (
        wk[wk['activo']]
        .groupby('semana_relativa')
        .agg(neg_activos=('id','nunique'))
        .reset_index()
    )
    if not temp.empty:
        max_weeks = max(max_weeks, temp['semana_relativa'].max())
    all_w = pd.DataFrame({'semana_relativa':range(1,max_weeks+1)})
    full = all_w.merge(temp, on='semana_relativa', how='left').fillna({'neg_activos':0})
    full['retencion_%'] = full['neg_activos'] / size * 100
    full['periodo_affiliacion'] = coh
    full['tamaño_cohorte']   = size
    ret_all.append(full)

df_ret_all = pd.concat(ret_all, ignore_index=True)

retention_weighted_avg = (
    df_ret_all
    .groupby('semana_relativa')
    .apply(lambda x: x['neg_activos'].sum()/x['tamaño_cohorte'].sum()*100)
    .reset_index(name='retencion_ponderada_%')
)

ret_matrix = df_ret_all.pivot(
    index='periodo_affiliacion',
    columns='semana_relativa',
    values='retencion_%'
)
for w in range(1, max_weeks+1):
    if w not in ret_matrix.columns:
        ret_matrix[w] = np.nan
ret_matrix = ret_matrix.sort_index(axis=1)
ret_matrix.loc['Promedio Ponderado'] = retention_weighted_avg.set_index('semana_relativa')['retencion_ponderada_%']
# justo después de pivotar y antes de crear el heatmap:
ret_matrix = ret_matrix.round(2)


st.subheader("Heatmap de Retención por Cohorte (Desde Enero 2024)")
# CAMBIO IMPORTANTE: Configurar el heatmap con escala de color global
# - Establecer zmin=0, zmax=100 para que todos los valores del 40% tengan el mismo color
# - Utilizar una escala de color continua entre los colores de la marca
fig_matrix = px.imshow(
    ret_matrix,
    zmin=0, zmax=100,  # Escala global fija 0-100%
    color_continuous_scale=[
        [0, PRIMARY_COLOR],   # 0% retención
        [0.5, SECONDARY_COLOR],  # 50% retención
        [1, TERTIARY_COLOR]   # 100% retención
    ],
    aspect='auto',
    text_auto='.1f%',  # Mostrar un decimal
    labels={'x':'Semana','y':'Cohorte','color':'Retención %'},
    title='Heatmap de Retención por Cohorte (Desde Enero 2024)'
)

fig_matrix.update_layout(
    xaxis_side='top',
    xaxis_title='Semana relativa desde afiliación',
    yaxis_title='Cohorte (Semana de afiliación)',
    coloraxis_colorbar=dict(
        title='Retención %',
        tickvals=[0, 25, 50, 75, 100],  # Marcas específicas en la barra de color
        ticktext=['0%', '25%', '50%', '75%', '100%']
    ),
    font_color=PRIMARY_COLOR,
    margin=dict(l=50, r=20, t=50, b=20)
)

st.plotly_chart(fig_matrix, use_container_width=True)

# ==============================
# Gráfica de retención promedio
# ==============================
st.subheader("Retención Promedio Ponderada por Semana")
fig_avg = px.line(
    retention_weighted_avg,
    x='semana_relativa', y='retencion_ponderada_%',
    markers=True,
    color_discrete_sequence=[PRIMARY_COLOR],
    labels={
        'semana_relativa':'Semana relativa desde afiliación',
        'retencion_ponderada_%':'Retención promedio (%)'
    }
)
retention_weighted_avg['tendencia'] = (
    retention_weighted_avg['retencion_ponderada_%']
    .rolling(window=4, min_periods=1)
    .mean()
)
fig_avg.add_scatter(
    x=retention_weighted_avg['semana_relativa'],
    y=retention_weighted_avg['tendencia'],
    mode='lines',
    name='Tendencia (mov. 3 sem.)',
    line=dict(color=SECONDARY_COLOR, width=3, dash='dash')
)
fig_avg.update_layout(
    title="Retención Promedio Ponderada por Semana",
    xaxis_title="Semana relativa desde afiliación",
    yaxis_title="% de negocios activos",
    yaxis=dict(range=[0,100]),
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.2),
    font_color=PRIMARY_COLOR
)
fig_avg.add_annotation(
    x=2, y=95,
    text=f"Criterio: ≥{min_trx} trx de ≥ Q{min_amount}",
    showarrow=False,
    font=dict(size=10, color=TERTIARY_COLOR),
    align="left"
)
st.plotly_chart(fig_avg, use_container_width=True)

# ==============================
# Tabla de datos final
# ==============================
st.subheader("Datos de Retención por Cohorte")
display_matrix = ret_matrix.round(1).copy()
cohort_sizes = pd.Series(sizes).reindex(display_matrix.index[:-1])
display_matrix.insert(0, 'Tamaño', cohort_sizes)
for col in display_matrix.columns:
    if col != 'Tamaño':
        display_matrix[col] = display_matrix[col].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else ""
        )
    else:
        display_matrix[col] = display_matrix[col].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else ""
        )
st.dataframe(display_matrix)
