import pandas as pd
import numpy as np
import geopandas as gpd
import scipy.stats as stats
import matplotlib.pyplot as plt
import streamlit as st

def encontrar_punto_mas_cercano(geom, gdf_no_nulos):
    """Encuentra el índice del punto más cercano a una geometría dada.

    Args:
        geom (shapely.geometry.Point): Punto a comparar.
        gdf_no_nulos (gpd.GeoDataFrame): GeoDataFrame con puntos válidos.

    Returns:
        int: Índice del punto más cercano.
    """
    return gdf_no_nulos.distance(geom).idxmin()

def encontrar_historial_mas_cercano(historial, df_no_nulos):
    """Encuentra el índice del historial de compras más cercano a un valor dado.

    Args:
        historial (float): Valor de historial de compras a comparar.
        df_no_nulos (pd.DataFrame): DataFrame sin valores nulos en la columna 'Frecuencia_Compra'.

    Returns:
        int: Índice del historial más cercano en df_no_nulos.
    """
    return (df_no_nulos['Historial_Compras'] - historial).abs().idxmin()

def interpolar(df):
    """Interpola los valores faltantes en un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con datos faltantes.

    Returns:
        pd.DataFrame: DataFrame con valores interpolados.
    """

    # Interpolación para nombres
    df['Nombre'] = df['Nombre'].ffill()

    # Interpolación para edades
    prom_edad_f = int(df[df['Género'] == 'Femenino']['Edad'].mean())
    df.loc[(df['Género'] == 'Femenino') & (df['Edad'].isna()), 'Edad'] = prom_edad_f

    prom_edad_m = int(df[df['Género'] == 'Masculino']['Edad'].mean())
    df.loc[(df['Género'] == 'Masculino') & (df['Edad'].isna()), 'Edad'] = prom_edad_f

    # Interpolación para latitud y longitud
    df['Longitud'] = df['Longitud'].interpolate(method='linear')
    df['Latitud'] = df['Latitud'].interpolate(method='linear')

    df = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_xy(df.Longitud, df.Latitud))

    # Interpolación para ingresos
    gdf_no_nulos = df.dropna(subset=['Ingreso_Anual_USD'])
    idx_mas_cercano = df['geometry'].apply(encontrar_punto_mas_cercano, gdf_no_nulos=gdf_no_nulos)
    ingresos_interpolados = gdf_no_nulos.loc[idx_mas_cercano, 'Ingreso_Anual_USD'].values
    df['Ingreso_Anual_USD'] = df['Ingreso_Anual_USD'].fillna(pd.Series(ingresos_interpolados, index=df.index))

    # Interpolación para Historial_Compras
    gdf_no_nulos = df.dropna(subset=['Historial_Compras'])
    idx_mas_cercano = df['geometry'].apply(encontrar_punto_mas_cercano, gdf_no_nulos=gdf_no_nulos)
    ingresos_interpolados = gdf_no_nulos.loc[idx_mas_cercano, 'Historial_Compras'].values
    df['Historial_Compras'] = df['Historial_Compras'].fillna(pd.Series(ingresos_interpolados, index=df.index))

    # Interpolación Frecuencia_Compra
    df_no_nulos = df.dropna(subset=['Frecuencia_Compra'])
    idx_mas_cercano = df['Historial_Compras'].apply(encontrar_historial_mas_cercano, df_no_nulos=df_no_nulos)
    frecuencia_interpolada = df_no_nulos.loc[idx_mas_cercano.dropna(), 'Frecuencia_Compra'].values
    df.loc[df['Frecuencia_Compra'].isna(), 'Frecuencia_Compra'] = pd.Series(frecuencia_interpolada)

    # Interpolación Género
    df['Género'] = df['Género'].mask(df['Género'].isna(), df.groupby('Nombre')['Género'].transform(lambda x: x.ffill()))
    return df

def leer_archivo(url):
    """Lee un archivo CSV y aplica la función de interpolación.

    Args:
        url (str): URL del archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con datos interpolados.
    """
    df = interpolar(pd.read_csv(url))
    return df

def analisis_correlacion(df):
    """Analiza la correlación entre Edad e Ingreso_Anual_USD y permite segmentarla por Género y Frecuencia de Compra.

    Args:
        df (pd.DataFrame): DataFrame con las columnas 'Edad', 'Ingreso_Anual_USD', 'Género' y 'Frecuencia_Compra'.
    """

    correlacion_global = df["Edad"].corr(df["Ingreso_Anual_USD"])
    st.write(f" **Correlación Global (Edad vs Ingreso Anual):** {correlacion_global:.2f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Edad"], df["Ingreso_Anual_USD"], alpha=0.5, c="blue")
    ax.set_title(f"Correlación Global: {correlacion_global:.2f}")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Ingreso Anual (USD)")
    st.pyplot(fig)

    genero = st.selectbox("Selecciona el Género:", ["Masculino", "Femenino"])
    frecuencia = st.selectbox("Selecciona la Frecuencia de Compra:", ["Baja", "Media", "Alta"])

    subset = df[(df["Género"] == genero) & (df["Frecuencia_Compra"] == frecuencia)]

    if len(subset) > 2:
        correlacion = subset["Edad"].corr(subset["Ingreso_Anual_USD"])
        st.write(f"**Correlación para {genero} - Frecuencia {frecuencia}:** {correlacion:.2f}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(subset["Edad"], subset["Ingreso_Anual_USD"], alpha=0.5, color="red")
        ax.set_title(f"{genero} - {frecuencia} (r={correlacion:.2f})")
        ax.set_xlabel("Edad")
        ax.set_ylabel("Ingreso Anual (USD)")
        st.pyplot(fig)
    else:
        st.warning(f"⚠️ No hay suficientes datos para calcular la correlación en {genero} - {frecuencia}.")

def mapa_clientes(df, filtro=None):
    """Crea un mapa de clientes segmentado por filtro.

    Args:
        df (pd.DataFrame): DataFrame con datos de clientes.
        filtro (str, optional): Columna para filtrar los datos. Defaults to None.
    """
    ruta_mapa = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    mundo_dataframe = gpd.read_file(ruta_mapa)
    mundo_dataframe = mundo_dataframe[mundo_dataframe['CONTINENT'] == 'South America']

    fig, ax = plt.subplots()
    mundo_dataframe.plot(ax=ax, color='white', edgecolor='black')

    if filtro != "Global":
        categorias = df[filtro].astype(str).unique()
        colores = plt.cm.get_cmap("coolwarm", len(categorias))

        color_map = {categoria: colores(i) for i, categoria in enumerate(categorias)}
        colores_puntos = df[filtro].map(color_map)

        sc = ax.scatter(df["Longitud"], df["Latitud"], c=colores_puntos, edgecolor="black")
        ax.legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c], markersize=4, label=c) for c in categorias],
                  title=filtro, loc="upper right")
    else:
        ax.scatter(df["Longitud"], df["Latitud"], color="blue", alpha=0.6, edgecolor="black")
    st.pyplot(fig)

def analisis_cluster(df):
    """Muestra la distribución de clusters según la frecuencia de compra.

    Args:
        df (pd.DataFrame): DataFrame con la columna 'Frecuencia_Compra'.
    """
    st.write("Distribución de Clusters:")
    st.write(df['Frecuencia_Compra'].value_counts())

def grafico_barras(df):
    """Crea un gráfico de barras apiladas de la frecuencia de compra por género.

    Args:
        df (pd.DataFrame): DataFrame con las columnas 'Género' y 'Frecuencia_Compra'.
    """
    fig, ax = plt.subplots()
    df.groupby('Género')['Frecuencia_Compra'].value_counts().unstack().plot(kind='bar', stacked=True, ax=ax)
    ax.set_title("Frecuencia de Compra por Género")
    st.pyplot(fig)

def mapa_calor_ingresos(df):
    """Crea un mapa de calor de ingresos usando la longitud y latitud.

    Args:
        df (pd.DataFrame): DataFrame con las columnas 'Longitud', 'Latitud' e 'Ingreso_Anual_USD'.
    """
    fig, ax = plt.subplots()
    df.plot.scatter(x='Longitud', y='Latitud', c='Ingreso_Anual_USD', cmap='hot', alpha=0.5, ax=ax)
    ax.set_title("Mapa de Calor de Ingresos")
    st.pyplot(fig)

def calcular_distancias(df):
    """Calcula y muestra las distancias entre los 10 compradores con mayores ingresos.

    Args:
        df (pd.DataFrame): DataFrame con las columnas 'Ingreso_Anual_USD' y 'geometry' (geometría de los puntos).
    """
    top_ingresos = df.nlargest(10, 'Ingreso_Anual_USD')
    st.write("Distancias entre compradores de mayores ingresos:")
    st.write(top_ingresos.geometry.distance(top_ingresos.geometry.shift()))


st.title("Análisis de Clientes")
st.write('Hecho por: Juan Pablo Zuluaga Mesa')
st.write('Link de sugerencia: https://raw.githubusercontent.com/gabrielawad/programacion-para-ingenieria/refs/heads/main/archivos-datos/aplicaciones/analisis_clientes.csv')
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
    filtro = st.selectbox("Seleccione filtro:", ['Global', 'Género', 'Frecuencia_Compra'])
    mapa_clientes(df, filtro)
    st.subheader('La implementacion llego hasta aqui, el demas analisis proximamente...')
    st.subheader("Análisis de Clúster")
    analisis_cluster(df)
    st.subheader("Gráfico de Barras")
    grafico_barras(df)
    st.subheader("Mapa de Calor de Ingresos")
    mapa_calor_ingresos(df)
    st.subheader("Cálculo de Distancias")
    calcular_distancias(df)
