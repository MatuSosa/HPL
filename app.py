import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página estilo Sprinklr (Tema Oscuro Corporativo)
st.set_page_config(page_title="Sprinklr Core - Airline Monitoring", layout="wide")

# --- 1. DEFINICIÓN DE LA FUNCIÓN DE CARGA LOCAL ---
@st.cache_data
def descargar_dataset_real():
    archivo_local = "Tweets.csv"
    try:
        df_raw = pd.read_csv(archivo_local)
        
        # Seleccionamos las columnas clave, incluyendo 'airline'
        df_clean = df_raw[['text', 'airline_sentiment', 'negativereason', 'airline']].copy()
        df_clean.columns = ['Text', 'Sentiment', 'Topic', 'Airline']
        
        # Agregamos una columna de usuario simulada para la visualización estática
        df_clean['User'] = "@passenger_ir"
        
        # Limpieza de nulos en los tópicos de quejas
        df_clean['Topic'] = df_clean['Topic'].fillna("General / Other")
        
        # Traducimos las categorías de tópicos más comunes de Kaggle
        mapa_topicos = {
            'Customer Service Issue': 'Atención al Cliente',
            'Late Flight': 'Vuelo Demorado',
            'Cancelled Flight': 'Vuelo Cancelado',
            'Lost Luggage': 'Equipaje Perdido',
            'Bad Flight': 'Experiencia de Vuelo Mala',
            'Flight Booking Problems': 'Problemas de Reserva',
            'Flight Attendant Complaints': 'Quejas del Personal',
            'Long Lines': 'Filas Largas',
            'Damaged Luggage': 'Equipaje Dañado'
        }
        df_clean['Topic'] = df_clean['Topic'].replace(mapa_topicos)
        
        # Limpiamos espacios en blanco y pasamos a minúsculas los sentimientos
        df_clean['Sentiment'] = df_clean['Sentiment'].astype(str).str.strip().str.lower()
        
        # Mapeo explícito al español
        mapa_sentimientos = {'positive': 'Positivo', 'negative': 'Negativo', 'neutral': 'Neutro'}
        df_clean['Sentiment'] = df_clean['Sentiment'].map(mapa_sentimientos)
        
        return df_clean
        
    except FileNotFoundError:
        st.error(f"No se encontró el archivo '{archivo_local}' en la carpeta del proyecto. Asegurate de que esté en la misma carpeta que este script.")
        return pd.DataFrame(columns=['Text', 'Sentiment', 'Topic', 'User', 'Airline'])

# --- 2. INICIALIZACIÓN Y CARGA DE DATOS ---
with st.spinner("Leyendo y preprocesando dataset real de Twitter de forma local..."):
    df_master = descargar_dataset_real()

# --- RESTRICCIÓN DE DATOS PARA LA SIMULACIÓN (MUESTRA REAL MIXTA MULTI-AEROLÍNEA) ---
# Desordenamos el dataset completo al azar y tomamos 1500 registros para que entren todas las marcas.
if 'dataset' not in st.session_state:
    st.session_state.dataset = df_master.sample(n=1500, random_state=42).copy()
    st.session_state.estado = "Monitoreo Histórico Completo (Datos Reales)"

# --- INTERFAZ DE USUARIO (SPRINKLR SIMULATOR) ---
st.title("📊 Sprinklr AI - Multi-Airline Social Listening")

# --- AQUÍ ESTÁ EL ARREGLO DEL NOMBRE DINÁMICO ---
# Creamos un nombre limpio para mostrar en el subtítulo según la selección
aerolinea_titulo = "Global Airlines Pool"

# FILTRO REAL DE AEROLÍNEAS EN LA BARRA LATERAL
st.sidebar.header("Centro de Control e Ingesta")
lista_aerolineas = ["Todas"] + list(df_master['Airline'].unique())
aerolinea_seleccionada = st.sidebar.selectbox("✈️ Seleccionar Aerolínea a Monitorear:", lista_aerolineas)

if aerolinea_seleccionada != "Todas":
    aerolinea_titulo = aerolinea_seleccionada

# Actualizamos el subtítulo dinámicamente
st.subheader(f"Client: {aerolinea_titulo} | Real-Time NLP Analysis Pipeline")
st.markdown("---")

st.sidebar.markdown("**Simulación de Ingesta (NLP Engine):**")

# Botón 1: Reset al Estado Histórico Real Mixto
if st.sidebar.button("Resetear Datos", use_container_width=True):
    st.session_state.dataset = df_master.sample(n=1500, random_state=42).copy()
    st.session_state.estado = "Monitoreo Histórico Completo (Datos Reales)"
    st.toast("Datos reseteados al estado histórico real.")

# Botón 2: Inyección de Crisis (Filtra por la aerolínea seleccionada si corresponde)
if st.sidebar.button("Inyectar Flujo Crisis (Negativo)", use_container_width=True):
    df_negativos = df_master[df_master['Sentiment'] == 'Negativo']
    if aerolinea_seleccionada != "Todas":
        df_negativos = df_negativos[df_negativos['Airline'] == aerolinea_seleccionada]
        
    if not df_negativos.empty:
        tweets_negativos = df_negativos.sample(min(150, len(df_negativos)), random_state=42)
        st.session_state.dataset = pd.concat([st.session_state.dataset, tweets_negativos], ignore_index=True)
        st.session_state.estado = f"Crisis Activa Detectada en {aerolinea_seleccionada if aerolinea_seleccionada != 'Todas' else 'el Pool'}"
        st.toast("¡Alerta de anomalía! Alto volumen de quejas reales ingresando.")
    else:
        st.sidebar.error("No se encontraron suficientes tweets Negativos.")

# Botón 3: Inyección de Campaña
if st.sidebar.button("Inyectar Campaña (Positivo)", use_container_width=True):
    df_positivos = df_master[df_master['Sentiment'] == 'Positivo']
    if aerolinea_seleccionada != "Todas":
        df_positivos = df_positivos[df_positivos['Airline'] == aerolinea_seleccionada]
        
    if not df_positivos.empty:
        tweets_positivos = df_positivos.sample(min(150, len(df_positivos)), random_state=42)
        st.session_state.dataset = pd.concat([st.session_state.dataset, tweets_positivos], ignore_index=True)
        st.session_state.estado = "Mitigación Activa / Impacto de Campaña"
        st.toast("Procesando menciones de fidelización de usuarios.")
    else:
        st.sidebar.error("No se encontraron suficientes tweets Positivos.")

# --- 3. APLICACIÓN DEL FILTRO DE VISUALIZACIÓN ---
df_actual = st.session_state.dataset
if aerolinea_seleccionada != "Todas":
    df_visualizacion = df_actual[df_actual['Airline'] == aerolinea_seleccionada]
else:
    df_visualizacion = df_actual

# --- PROCESAMIENTO DE MÉTRICAS EN TIEMPO REAL ---
total_menciones = len(df_visualizacion)
positivos = len(df_visualizacion[df_visualizacion["Sentiment"] == "Positivo"])
negativos = len(df_visualizacion[df_visualizacion["Sentiment"] == "Negativo"])
neutros = len(df_visualizacion[df_visualizacion["Sentiment"] == "Neutro"])

nps_simulado = int(((positivos - negativos) / total_menciones) * 100) if total_menciones > 0 else 0

# Render de las tarjetas de KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Total Tweets Analizados", value=total_menciones)
col2.metric(label="Menciones Negativas", value=negativos, delta=f"+150" if "Crisis" in st.session_state.estado else None, delta_color="inverse")
col3.metric(label="Menciones Positivas", value=positivos, delta=f"+150" if "Campaña" in st.session_state.estado else None)
col4.metric(label="Net Sentiment Index", value=f"{nps_simulado}%", delta="ALERTA CRÍTICA" if nps_simulado < -20 else "ESTABLE")

st.markdown(f"**Filtro Actual:** Visibilidad enfocada en `{aerolinea_seleccionada}` | **Estado:** `{st.session_state.estado}`")
st.markdown("---")

# --- 4. RENDER DE GRÁFICOS DINÁMICOS ---
col_graph1, col_graph2 = st.columns(2)
color_map = {"Positivo": "#2ecc71", "Neutro": "#f1c40f", "Negativo": "#e74c3c"}

with col_graph1:
    st.write("### Distribución de Sentimientos (Pipeline PLN)")
    if total_menciones > 0:
        fig_pie = px.pie(df_visualizacion, names="Sentiment", color="Sentiment", color_discrete_map=color_map, hole=0.4)
        fig_pie.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No hay datos para mostrar con el filtro actual.")

with col_graph2:
    st.write("### Tópicos Críticos o Motivos de Quejas")
    df_topics = df_visualizacion[df_visualizacion['Topic'] != 'General / Other']
    if not df_topics.empty:
        fig_bar = px.bar(df_topics, y="Topic", color="Sentiment", color_discrete_map=color_map, barmode="stack", orientation='h')
        fig_bar.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), yaxis={'categoryorder':'total ascending'}, xaxis_title="Cantidad de Tweets")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Sin anomalías específicas reportadas en esta selección.")

st.markdown("---")

# --- 5. STREAM EN VIVO (TABLA DE ENTRADA) ---
st.write("### Feed de Ingesta Directa - Clasificación en Tiempo Real")
st.markdown("Output del pipeline de PLN (se muestra la columna de procedencia de datos):")

def color_sentiment_row(val):
    if val == "Positivo": return 'background-color: rgba(46, 204, 113, 0.15)'
    elif val == "Negativo": return 'background-color: rgba(231, 76, 60, 0.15)'
    return 'background-color: rgba(241, 196, 15, 0.05)'

if not df_visualizacion.empty:
    df_mostrar = df_visualizacion[['Airline', 'User', 'Text', 'Sentiment', 'Topic']].iloc[::-1]
    styled_df = df_mostrar.style.map(color_sentiment_row, subset=['Sentiment'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)