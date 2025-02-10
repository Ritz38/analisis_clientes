import pandas as pd
import numpy as np
import geopandas as gpd
import scipy.stats as stats
import matplotlib.pyplot as plt
import streamlit as st

def encontrar_punto_mas_cercano(geom, gdf_no_nulos):
    """
    Encuentra el índice del punto más cercano a una geometría dada.

    Args:
        geom (shapely.geometry.Point): Punto a comparar.
        gdf_no_nulos (gpd.GeoDataFrame): GeoDataFrame con puntos válidos.

    Returns:
        int: Índice del punto más cercano.
    """
    return gdf_no_nulos.distance(geom).idxmin()

def encontrar_historial_mas_cercano(historial, df_no_nulos):
    """
    Encuentra el índice del historial de compras más cercano a un valor dado.

    Args:
        historial (float): Valor de historial de compras a comparar.
        df_no_nulos (pd.DataFrame): DataFrame sin valores nulos en la columna 'Frecuencia_Compra'.

    Returns:
        int: Índice del historial más cercano en df_no_nulos.
    """
    return (df_no_nulos['Historial_Compras'] - historial).abs().idxmin()

def interpolar(df):
    # Interpolacion para nombres
    df['Nombre'] = df['Nombre'].ffill()

    # Interpolacion para edades
    prom_edad_f = int(df[df['Género'] == 'Femenino']['Edad'].mean())

    df.loc[(df['Género'] == 'Femenino') & (df['Edad'].isna()), 'Edad'] = prom_edad_f

    prom_edad_m = int(df[df['Género'] == 'Masculino']['Edad'].mean())

    df.loc[(df['Género'] == 'Masculino') & (df['Edad'].isna()), 'Edad'] = prom_edad_f


    #Interpolacion para latitud y longitud
    df['Longitud'] = df['Longitud'].interpolate(method='linear')
    df['Latitud'] = df['Latitud'].interpolate(method='linear')


    df = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_xy(df.Longitud, df.Latitud))

    # Interpolacion para ingresos
    # Filtrar puntos con valores conocidos
    gdf_no_nulos = df.dropna(subset=['Ingreso_Anual_USD'])

    # Aplicar la función para encontrar el índice del punto más cercano
    idx_mas_cercano = df['geometry'].apply(encontrar_punto_mas_cercano, gdf_no_nulos=gdf_no_nulos)

    # Obtener los valores interpolados de 'Ingresos'
    ingresos_interpolados = gdf_no_nulos.loc[idx_mas_cercano, 'Ingreso_Anual_USD'].values

    # Rellenar los NaN con los valores interpolados
    df['Ingreso_Anual_USD'] = df['Ingreso_Anual_USD'].fillna(pd.Series(ingresos_interpolados, index=df.index))



    # Interpolacion para Historial_Compras
    # Filtrar puntos con valores conocidos
    gdf_no_nulos = df.dropna(subset=['Historial_Compras'])

    # Aplicar la función para encontrar el índice del punto más cercano
    idx_mas_cercano = df['geometry'].apply(encontrar_punto_mas_cercano, gdf_no_nulos=gdf_no_nulos)

    # Obtener los valores interpolados de 'Ingresos'
    ingresos_interpolados = gdf_no_nulos.loc[idx_mas_cercano, 'Historial_Compras'].values

    # Rellenar los NaN con los valores interpolados
    df['Historial_Compras'] = df['Historial_Compras'].fillna(pd.Series(ingresos_interpolados, index=df.index))

    # Interpolacion Frecuencia_Compra
    # Filtrar los datos sin nulos en 'Frecuencia_Compra'
    df_no_nulos = df.dropna(subset=['Frecuencia_Compra'])

    # Aplicar la función para encontrar el índice del historial más cercano
    idx_mas_cercano = df['Historial_Compras'].apply(encontrar_historial_mas_cercano, df_no_nulos=df_no_nulos)

    # Obtener los valores interpolados de 'Frecuencia_Compra'
    frecuencia_interpolada = df_no_nulos.loc[idx_mas_cercano.dropna(), 'Frecuencia_Compra'].values

    # Rellenar los valores NaN en 'Frecuencia_Compra'
    df.loc[df['Frecuencia_Compra'].isna(), 'Frecuencia_Compra'] = pd.Series(frecuencia_interpolada)



    # Interpolacion Género
    df['Género'] = df['Género'].mask(df['Género'].isna(), df.groupby('Nombre')['Género'].transform(lambda x: x.ffill()))
    return df

def leer_archivo(url):
    df = interpolar(pd.read_csv(url))
    return df

def analisis_correlacion(df):
    """Analiza la correlación entre Edad e Ingreso_Anual_USD y permite segmentarla por Género y Frecuencia de Compra.
    
    Args:
        df (pd.DataFrame): DataFrame con las columnas 'Edad', 'Ingreso_Anual_USD', 'Género' y 'Frecuencia_Compra'.
    """


    correlacion_global = df["Edad"].corr(df["Ingreso_Anual_USD"])
    st.write(f"📊 **Correlación Global (Edad vs Ingreso Anual):** {correlacion_global:.2f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Edad"], df["Ingreso_Anual_USD"], alpha=0.5, c="blue")
    ax.set_title(f"Correlación Global: {correlacion_global:.2f}")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Ingreso Anual (USD)")
    st.pyplot(fig)

    genero = st.selectbox("Selecciona el Género:", ["Masculino", "Femenino"])
    frecuencia = st.selectbox("Selecciona la Frecuencia de Compra:", ["Baja", "Media", "Alta"])

    

    # 📌 Filtrar datos según selección
    subset = df[(df["Género"] == genero) & (df["Frecuencia_Compra"] == frecuencia)]

    if len(subset) > 2:  # Necesitamos al menos 3 datos para calcular correlación
        correlacion = subset["Edad"].corr(subset["Ingreso_Anual_USD"])
        st.write(f"**Correlación para {genero} - Frecuencia {frecuencia}:** {correlacion:.2f}")

        # 📉 Gráfico de dispersión segmentado
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(subset["Edad"], subset["Ingreso_Anual_USD"], alpha=0.5, color="red")
        ax.set_title(f"{genero} - {frecuencia} (r={correlacion:.2f})")
        ax.set_xlabel("Edad")
        ax.set_ylabel("Ingreso Anual (USD)")
        st.pyplot(fig)
    else:
        st.warning(f"⚠️ No hay suficientes datos para calcular la correlación en {genero} - {frecuencia}.")

def mapa_clientes(df, filtro=None):
    ruta_mapa = "https://naturalearth.s3.amazonaws.com/50m_cultural\
/ne_50m_admin_0_countries.zip"
    mundo_dataframe = gpd.read_file(ruta_mapa)

    fig, ax = plt.subplots()
    mundo_dataframe.plot(ax=ax, color='white', edgecolor='black')
    df.plot.scatter(ax=ax, x='Longitud', y='Latitud', c=df[filtro] if filtro!='Global' else 'blue', cmap='coolwarm')
    ax.set_title(f'Mapa de Clientes {filtro if filtro!='Global' else "Global"}')
    st.pyplot(fig)

def analisis_cluster(df):
    st.write("Distribución de Clusters:")
    st.write(df['Frecuencia_Compra'].value_counts())

def grafico_barras(df):
    fig, ax = plt.subplots()
    df.groupby('Género')['Frecuencia_Compra'].value_counts().unstack().plot(kind='bar', stacked=True, ax=ax)
    ax.set_title("Frecuencia de Compra por Género")
    st.pyplot(fig)

def mapa_calor_ingresos(df):
    fig, ax = plt.subplots()
    df.plot.scatter(x='Longitud', y='Latitud', c='Ingreso_Anual_USD', cmap='hot', alpha=0.5, ax=ax)
    ax.set_title("Mapa de Calor de Ingresos")
    st.pyplot(fig)

def calcular_distancias(df):
    top_ingresos = df.nlargest(10, 'Ingreso_Anual_USD')
    st.write("Distancias entre compradores de mayores ingresos:")
    st.write(top_ingresos.geometry.distance(top_ingresos.geometry.shift()))


st.title("Análisis de Clientes")
opcion = st.radio("Seleccione el método de carga de datos:", ("Subir archivo", "Ingresar URL"))

df = None
if opcion == "Ingresar URL":
    ruta = st.text_input("Ingrese la URL del archivo CSV:")
    if ruta:
        df = leer_archivo(ruta)
elif opcion == "Subir archivo":
    archivo_subido = st.file_uploader("Suba un archivo CSV", type="csv")
    if archivo_subido:
        df = leer_archivo(archivo_subido)

if df is not None:
    st.subheader("Análisis de Correlación")
    analisis_correlacion(df)
    st.subheader("Mapa de Clientes")
    filtro = st.selectbox("Seleccione filtro:", ['Global'] + list(df.columns))
    mapa_clientes(df, filtro)
    st.subheader("Análisis de Clúster")
    analisis_cluster(df)
    st.subheader("Gráfico de Barras")
    grafico_barras(df)
    st.subheader("Mapa de Calor de Ingresos")
    mapa_calor_ingresos(df)
    st.subheader("Cálculo de Distancias")
    calcular_distancias(df)
