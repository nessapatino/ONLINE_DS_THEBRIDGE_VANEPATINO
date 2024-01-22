import pandas as pd


def tipifica_variables(df, umbral_categorica, umbral_continua):
    df_tipificacion = pd.DataFrame([df.nunique(), df.nunique() / len(df) * 100, df.dtypes]).T.rename(columns={0: "Card", 1: "%_Card", 2: "Tipo"})
    df_tipificacion["Clasificada_como"] = "Categorica"
    df_tipificacion.loc[df_tipificacion.Card == 2, "Clasificada_como"] = "Binaria"
    df_tipificacion.loc[df_tipificacion["Card"] >= umbral_categorica, "Clasificada_como"] = "Numerica Discreta"
    df_tipificacion.loc[(df_tipificacion["%_Card"] >= umbral_continua) & (df_tipificacion["Card"] >= umbral_categorica), "Clasificada_como"] = "Numerica Continua"

    resultado = pd.DataFrame({
        "nombre_variable": df_tipificacion.index,
        "tipo_sugerido": df_tipificacion["Clasificada_como"]
    })

    return resultado

def describe_df(df):
    tipos = df.dtypes
    nulos_porcentaje = df.isnull().mean() * 100
    unicos = df.nunique()
    cardinalidad_porcentaje = unicos / len(df) * 100

    df_describe = pd.DataFrame({
        "Tipo": tipos,
        "Porcentaje_Nulos": nulos_porcentaje,
        "Valores_Unicos": unicos,
        "Porcentaje_Cardinalidad": cardinalidad_porcentaje
    }).T



    return df_describe


