import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal

def frequencias_absolutas(dataframe, columnas_categoricas):
    """
    Función que dado un dataframe y una lista con las columnas categóricas de éste
    pinta los gráficos de frecuencia absoluta de cada columna con sus valores anotados 
    """
    num_cols = len(columnas_categoricas)
    num_filas = (num_cols // 2) + (num_cols % 2)
    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 
    for i, col in enumerate(columnas_categoricas):
        ax = axes[i] if num_cols > 1 else axes
        freq = dataframe[col].value_counts()
        barra = sns.barplot(x=freq.index, y=freq, ax=ax, palette='cool', hue = freq.index, legend = False)
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('Categorías')
        ax.set_ylabel('Frecuencia Absoluta')
        ax.tick_params(axis='x', rotation=45)

        for index, value in enumerate(freq):
            barra.text(index, value + 0.1, str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def frecuencias_relativas(dataframe, columnas_categoricas):
    """
    Función que dado un dataframe y una lista con las columnas categóricas de éste
    pinta los gráficos de frecuencia relativa de cada columna con sus valores anotados 
    """
    num_cols = len(columnas_categoricas)
    num_filas = (num_cols // 2) + (num_cols % 2)
    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 
    for i, col in enumerate(columnas_categoricas):
        ax = axes[i] if num_cols > 1 else axes
        freq = round(dataframe[col].value_counts()/len(dataframe)*100,2)
        barra = sns.barplot(x=freq.index, y=freq, ax=ax, palette='cool', hue = freq.index, legend = False)
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('Categorías')
        ax.set_ylabel('Frecuencia Relativa')
        ax.tick_params(axis='x', rotation=45)
        for index, value in enumerate(freq):
            barra.text(index, value, str(value), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def get_IQR(df, lista_columnas):
    """
    Función que calcula el IQR de cada variable númerica
    """
    iqr = {}
    for columna in lista_columnas:
        iqr[columna] = (df[columna].quantile(0.75) - df[columna].quantile(0.25))
    return

def variabilidad(df, columnas_numericas):
    """
    Función que calcula
    el coeficiente de varianza de cada columna numércia
    """
    df_var = df[columnas_numericas].describe().loc[["std","mean"]].T
    df_var["CV"] = df_var["std"]/df_var["mean"]
    return df_var

def boxplot_histograma(dataframe, columna):
    """
    Función que dibuja los diagramas de caja e histogramas
    con la densidad categórica de las variables numéricas.
    """ 
    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=dataframe[columna], whis= 3) 

    plt.subplot(1, 2, 2)
    sns.histplot(data=dataframe, x=columna, kde=True)
    plt.title(f'Histograma con densidad categórica de {columna}')

    plt.tight_layout()
    plt.show()

def barplot_categorica(df, columna_cat):
    """
    Función que crea un gráfico donde se muestra la frecuencia relativa
    de una columna categórica    
    """
    freq = round(df[columna_cat].value_counts()/len(df)*100,2)
    barra = sns.barplot(x=freq.index, y=freq, palette='cool', hue = freq.index, legend = False)
    plt.xlabel("MetabolicSyndrome")
    plt.ylabel("Frecuencia Relativa")
    plt.title("Distribución Sindrome Metabólico")

    for index, value in enumerate(freq):
        barra.text(index, value, str(value), ha='center', va='bottom')

    plt.tight_layout()
plt.show()

def bivariante_cat(df, cat_col1, cat_col2,  show_values= True):
    """
    Función que crea una gráfica que compara las 
    frecuencias relativas dedos variables categóricas 
    """
    
    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
    total_counts = df[cat_col1].value_counts()
    
    
    count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis=1)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=count_data, palette= "cool")

    plt.title(f'Relación entre {cat_col1} y {cat_col2}')
    plt.xlabel(cat_col1)
    plt.ylabel('Frecuencia Relativa')

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2%}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.show()


def bivariante_num(df, columna_num, columna_cat):
    """
    Función para realizar un diagrama de caja para el análisis bivariante entre una 
    columna numérica y la variable objetivo binaria.
    """
    plt.figure(figsize=(8,5))
    sns.boxplot(hue = columna_cat, y = columna_num, data=df, whis = 3, palette= "cool")
    plt.title(f'Análisis Bivariante entre  {columna_num} y {columna_cat}')
    plt.xlabel(columna_cat)
    plt.ylabel (columna_num)
    plt.show()

def histograma_bivariante(df, columna_num, columna_cat):
    """
    Función que pinta los histogramas de una variable numérica y una varible categórica 
    para realizar un análisis bivariante
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(data = df, x= columna_num, hue = columna_cat, kde = True, palette= "cool")
    

    plt.title(f'Análisis Bivariante entre {columna_num} y {columna_cat}')
    plt.xlabel(columna_num)
    plt.show()

def histogramas_multivariante(df, cat_col, num_col, group_size, bins = "auto"):
    """
   Función que crea múltiples histogramas para variables numéricas, 
   considerando una variable categórica para agrupar y comparar estas distribuciones.
   La función itera sobre las categorías únicas en lotes del tamaño especificado por group_size
    """
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        for cat in subset_cats:
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], kde=True, label=str(cat), bins = bins)
        
        plt.title(f'Histograms of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xlabel(num_col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

def trivariant_analysis(df, directora, cat2, num1, group_size = 3, bins = "auto"):
    """
    Función realiza un análisis trivariante generando histogramas para visualizar la distribución de los datos
    en relación con la variable directora y otras dos variables categóricas y numéricas.
    """
    col_directora = directora
    col_1 = cat2
    col_2 = num1
    diccionario_multivariante = {}
    for valor in df[col_directora].unique():
        diccionario_multivariante[valor] = df.loc[df[col_directora] == valor,[col_2,col_1]] 

    for valor,df_datos in diccionario_multivariante.items():
        print(f"Respuesta {valor}:")
        histogramas_multivariante(df_datos,col_1,col_2, group_size= group_size, bins = bins)

def tricategorical_analysis(df, directora, otras, relativa = False, muestra_valores = False):
    col_directora = directora
    col_1 = otras[0]
    col_2 = otras[1]
    diccionario_multivariante = {}
    for valor in df[col_directora].unique():
        diccionario_multivariante[valor] = df.loc[df[col_directora] == valor,[col_2,col_1]] 

    for valor,df_datos in diccionario_multivariante.items():
        print(f"Respuesta {valor}:")
        bivariante_cat(df_datos,col_2,col_1, show_values= True)
        