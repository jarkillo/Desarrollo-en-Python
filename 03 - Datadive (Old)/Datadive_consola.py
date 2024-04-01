from PIL import Image
import io
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import requests
from fpdf import FPDF
plt.ioff()
# ... importar otras bibliotecas según sea necesario

lista_de_graficos = []

# Carga de datos desde la API


def carga_base_de_datos():
    response = requests.get("http://127.0.0.1:8000/datasets")
    datasets = response.json()["datasets"]

    print("\nLista de datasets disponibles:")
    for idx, dataset in enumerate(datasets):
        print(f"{idx}. {dataset['name']} (ID: {dataset['id']})")

    selected_idx = int(
        input("\nSeleccione el índice del dataset que desea cargar: "))
    selected_dataset_id = datasets[selected_idx]["id"]

    response = requests.get(
        f"http://127.0.0.1:8000/preview/dataset/{selected_dataset_id}")

    if response.status_code == 200:
        preview_content = response.json()
        preview_df = pd.DataFrame(preview_content["preview"])
        print("\nVista previa del dataset:")
        print(preview_df)

        decision = input(
            "\n¿Desea descargar el dataset completo? (s/n): ").lower()
        if decision == 's':
            response = requests.get(
                f"http://127.0.0.1:8000/download/dataset/{selected_dataset_id}")
            if response.status_code == 200:
                dataset_content = response.json()
                df = pd.DataFrame(dataset_content["content"])
                print("\nDataset descargado con éxito!")
                return df
            else:
                print("Error al descargar el dataset completo.")
        else:
            print("Operación cancelada por el usuario.")
    else:
        print("Error al obtener la vista previa del dataset.")
    return None

# Carga de datos desde un archivo local


def carga_archivo():
    ruta_archivo = input("Ingrese la ruta del archivo: ")
    extension = ruta_archivo.split(".")[-1]

    try:
        assert extension in ["csv", "xlsx", "json"]
    except AssertionError:
        print("Extensión inválida. Intente de nuevo.")
        return None

    if extension == "csv":
        return pd.read_csv(ruta_archivo)
    elif extension == "xlsx":
        return pd.read_excel(ruta_archivo)
    elif extension == "json":
        return pd.read_json(ruta_archivo)


def revisar_dataset(df):
    # Mostrar las dimensiones del DataFrame
    print("Dimensiones del DataFrame:", df.shape)

    # Mostrar los tipos de datos
    print("\nTipos de datos:\n", df.dtypes)

    # Mostrar las primeras filas del DataFrame
    print("\nPrimeras filas del DataFrame:\n", df.head())

    # Mostrar las últimas filas del DataFrame
    print("\nÚltimas filas del DataFrame:\n", df.tail())

    # Mostrar un resumen estadístico para las columnas numéricas
    print("\nDescripción estadística del DataFrame:\n", df.describe())

    # Mostrar un resumen de valores faltantes
    print("\nValores faltantes en el DataFrame:\n", df.isnull().sum())

    # Visualizaciones básicas: Histogramas para columnas numéricas
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histograma de {col}')
        plt.savefig('mnt/data/histogram' + col+'.png')
        lista_de_graficos.append('mnt/data/histogram' + col+'.png')
        plt.close()

    # Visualizaciones básicas: Diagramas de caja para columnas numéricas
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Diagrama de caja de {col}')
        plt.savefig('mnt/data/caja' + col+'.png')
        lista_de_graficos.append('mnt/data/caja' + col+'.png')
        plt.close()
    # Aquí puedes agregar más análisis o visualizaciones según sea necesario


def encontrar_atipicos(df):
    atipicos = {}
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        filtro = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
        atipicos[col] = df[filtro]

    return atipicos


def eliminar_atipicos(df):

    df_limpio = df.copy()
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        filtro = ~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
        df_limpio = df_limpio[filtro]

    return df_limpio


def estadisticas_descriptivas(df):
    # Calculando estadísticas descriptivas
    descripciones = df.describe(include='all')
    asimetria = df.skew()
    curtosis = df.kurtosis()

    # Asegurándose de que asimetria y curtosis tengan el mismo índice que descripciones
    asimetria = asimetria.reindex(descripciones.columns, fill_value='NaN')
    curtosis = curtosis.reindex(descripciones.columns, fill_value='NaN')

    # Añadiendo asimetria y curtosis al DataFrame descripciones
    descripciones.loc['asimetria'] = asimetria
    descripciones.loc['curtosis'] = curtosis

    # Mostrando estadísticas descriptivas
    print("Estadísticas Descriptivas:\n", descripciones)

    # Visualizaciones para columnas numéricas
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(10, 4))

        # Histograma
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Histograma de {col}')
        plt.savefig('mnt/data/histo2' + col+'.png')
        plt.close()

        # Diagrama de caja
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Diagrama de caja de {col}')

        plt.tight_layout()
        plt.savefig('mnt/data/caja2' + col+'.png')
        lista_de_graficos.append('mnt/data/caja2' + col+'.png')
        plt.close()

    # Considerar visualizaciones adicionales para datos categóricos si es necesario

    return descripciones


def ofrecer_analisis_profundo(df):
    print("Basado en su conjunto de datos, las siguientes opciones de análisis podrían ser relevantes:")

    # Ejemplo de recomendación basada en el tipo de variables
    if df.select_dtypes(include='number').shape[1] > 1:
        print("\n1. Análisis de Regresión")
        print("   - Explora relaciones lineales entre variables.")
        print("   - Predice valores de una variable basándose en otras.")
        print("   - Recomendado si tienes variables numéricas y deseas entender cómo se relacionan entre sí o predecir una variable.")

    if len(df.columns) > 3:
        print("\n2. Clustering")
        print("   - Agrupa datos en clusters basados en similitudes.")
        print("   - Útil para descubrir patrones y agrupaciones no evidentes.")
        print("   - Recomendado si tienes muchas variables y deseas explorar agrupaciones o patrones naturales en tus datos.")

    if df.select_dtypes(include='number').shape[1] > 2:
        print("\n3. Análisis de Componentes Principales (PCA)")
        print("   - Reduce la dimensionalidad de los datos.")
        print("   - Mantiene la mayor cantidad de información posible.")
        print("   - Recomendado para simplificar conjuntos de datos complejos y facilitar su visualización y análisis.")

    if 'fecha' in df.columns or 'tiempo' in df.columns:
        print("\n4. Análisis de Series Temporales")
        print("   - Analiza tendencias y patrones a lo largo del tiempo.")
        print("   - Recomendado si tus datos están indexados por tiempo y deseas explorar cómo cambian las variables a lo largo del tiempo.")

    seleccion = int(input("\nSeleccione una opción de análisis (1-4): "))
    return seleccion


def realizar_analisis_profundo(df, opcion):
    if opcion == 1:
        # Análisis de Regresión
        return realizar_analisis_regresion(df)
    elif opcion == 2:
        # Clustering
        return realizar_clustering(df)
    elif opcion == 3:
        # Análisis de Componentes Principales (PCA)
        return realizar_pca(df)
    elif opcion == 4:
        # Análisis de Series Temporales
        return realizar_analisis_series_temporales(df)
    else:
        print("Opción no válida")
        return None


def realizar_analisis_regresion(df):

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    # Seleccionar la variable objetivo y las características
    # Esto es solo un ejemplo, necesitarás ajustar esto según tu conjunto de datos
    # Reemplaza 'variable_objetivo' con el nombre de tu columna objetivo
    y = df['variable_objetivo']
    # Usa todas las otras columnas como características
    X = df.drop('variable_objetivo', axis=1)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # Calcular métricas de rendimiento
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Mostrar resultados
    print("Error Cuadrático Medio (MSE):", mse)
    print("Coeficiente de Determinación (R^2):", r2)

    # Opcional: Visualizar las predicciones
    plt.scatter(y_test, y_pred)
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Valores Reales vs. Predicciones")
    plt.savefig('mnt/data/scatt.png')
    lista_de_graficos.append('mnt/data/scatt.png')
    plt.close()

    return modelo


def realizar_clustering(df):
    # Preparar los datos para el clustering
    # Aquí, suponemos que todas las columnas son relevantes para el clustering
    X = df.select_dtypes(include=['float64', 'int64'])

    # Determinar el número óptimo de clusters
    # Esto se hace comúnmente usando el método del codo
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Método del Codo')
    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS')
    plt.savefig('mnt/data/wcss.png')
    lista_de_graficos.append('mnt/data/wcss.png')
    plt.close()

    # Elegir el número de clusters (n_clusters) basado en el gráfico anterior
    n_clusters = int(input("Ingrese el número óptimo de clusters: "))

    # Aplicar K-Means al conjunto de datos
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Visualizar los clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y_kmeans,
                    palette=sns.color_palette('hsv', n_colors=n_clusters))
    plt.title('Clusters de datos')
    plt.savefig('mnt/data/cluster.png')
    lista_de_graficos.append('mnt/data/cluster.png')
    plt.close()

    return kmeans


def realizar_pca(df):
    # Seleccionar solo las columnas numéricas para PCA
    X = df.select_dtypes(include=['float64', 'int64'])

    # Estandarizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA
    pca = PCA(n_components=2)  # Reducir a 2 dimensiones como ejemplo
    X_pca = pca.fit_transform(X_scaled)

    # Convertir a DataFrame para facilitar la visualización
    df_pca = pd.DataFrame(data=X_pca, columns=[
                          'Componente Principal 1', 'Componente Principal 2'])

    # Visualizar los resultados
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca['Componente Principal 1'],
                df_pca['Componente Principal 2'])
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('PCA de 2 Componentes')
    plt.savefig('mnt/data/pca.png')
    lista_de_graficos.append('mnt/data/pca.png')
    plt.close()

    # Mostrar la varianza explicada por cada componente
    print("Varianza explicada por cada componente:",
          pca.explained_variance_ratio_)

    return df_pca, pca


def realizar_analisis_series_temporales(df, columna_fecha, columna_datos):
    # Convertir la columna de fecha en índice de tiempo
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df.set_index(columna_fecha, inplace=True)

    # Descomposición de la serie temporal
    resultado_descomposicion = seasonal_decompose(
        df[columna_datos], model='additive')
    resultado_descomposicion.plot()

    # Análisis de autocorrelación y autocorrelación parcial
    plot_acf(df[columna_datos])
    plot_pacf(df[columna_datos])

    # Modelo ARIMA (ajustar los parámetros p, d, q según sea necesario)
    modelo_arima = ARIMA(df[columna_datos], order=(1, 1, 1))
    resultado_arima = modelo_arima.fit()
    print(resultado_arima.summary())

    # Predicciones (ajustar los períodos según sea necesario)
    df['predicciones'] = resultado_arima.predict(start=pd.to_datetime(
        'fecha_inicio'), end=pd.to_datetime('fecha_fin'), dynamic=False)
    plt.figure(figsize=(10, 6))


def interpretar_analisis(df, resultados):
    # La estructura de 'resultados' dependerá de cómo hayas estructurado las salidas de tus análisis
    # Supongamos que 'resultados' es un diccionario con claves que indican el tipo de análisis
    # y los valores son los resultados específicos de esos análisis.

    if 'regresion' in resultados:
        interpretar_regresion(resultados['regresion'])

    if 'clustering' in resultados:
        interpretar_clustering(resultados['clustering'], df)

    if 'pca' in resultados:
        interpretar_pca(resultados['pca'])

    if 'series_temporales' in resultados:
        interpretar_series_temporales(resultados['series_temporales'])


def interpretar_regresion(modelo_regresion):
    # Coeficientes
    coeficientes = modelo_regresion.coef_
    intercepto = modelo_regresion.intercept_
    print("Intercepto:", intercepto)
    print("Coeficientes:")
    for coef in coeficientes:
        print(coef)

    # Interpretación
    print("\nInterpretación:")
    print("El intercepto representa el valor esperado de Y cuando todas las variables independientes son 0.")
    print("Cada coeficiente representa el cambio en la variable dependiente por una unidad de cambio en la variable independiente correspondiente, manteniendo todas las otras variables constantes.")

    # Discutir cada coeficiente
    for i, coef in enumerate(coeficientes):
        print(f"El coeficiente para la variable {i} es {coef}, lo que indica que, manteniendo todas las demás variables constantes, un aumento de una unidad en esta variable está asociado con un cambio de {coef} unidades en la variable dependiente.")

    # Si tienes información sobre el R^2, puedes incluirla aquí


def interpretar_clustering(modelo_clustering, df):
    # Añadir las etiquetas del cluster al DataFrame original
    df['Cluster'] = modelo_clustering.labels_

    # Calcular las medias de cada variable para cada cluster
    cluster_means = df.groupby('Cluster').mean()
    print("Medias de cada variable para cada cluster:")
    print(cluster_means)

    # Visualización de los clusters
    # Esto es más efectivo si tienes 2 o 3 dimensiones principales
    # Aquí, se asume un análisis de 2 dimensiones
    if 'Componente Principal 1' in df.columns and 'Componente Principal 2' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Componente Principal 1', y='Componente Principal 2',
                        hue='Cluster', data=df, palette='bright')
        plt.title('Visualización de Clusters')
        plt.savefig('mnt/data/compond.png')
        lista_de_graficos.append('mnt/data/compond.png')
        plt.close()

    # Interpretación
    print("\nInterpretación:")
    for i in range(modelo_clustering.n_clusters):
        print(f"\nCluster {i}:")
        print("Características destacadas:")
        # Aquí puedes añadir lógica para interpretar cada cluster
        # Por ejemplo, identificar las variables con valores más altos o más bajos en cada cluster
        for col in cluster_means.columns:
            if cluster_means.at[i, col] > df[col].mean():
                print(f"  Alta {col} (media: {cluster_means.at[i, col]:.2f})")
            elif cluster_means.at[i, col] < df[col].mean():
                print(f"  Baja {col} (media: {cluster_means.at[i, col]:.2f})")

    # Nota: Esta es una interpretación básica. La interpretación real puede requerir un análisis más detallado, incluyendo la evaluación de la relevancia estadística de las diferencias entre clusters.


def interpretar_pca(pca):
    # Varianza explicada por cada componente
    varianza_explicada = pca.explained_variance_ratio_
    print("Varianza explicada por cada componente principal:")
    for i, var in enumerate(varianza_explicada):
        print(f"Componente Principal {i+1}: {var:.2f}")

    # Varianza acumulada
    varianza_acumulada = varianza_explicada.cumsum()
    print("\nVarianza acumulada por los componentes:")
    for i, var_acum in enumerate(varianza_acumulada):
        print(f"Primeros {i+1} componentes: {var_acum:.2f}")

    # Interpretación
    print("\nInterpretación:")
    for i, var in enumerate(varianza_explicada):
        print(
            f"El Componente Principal {i+1} captura el {var:.2f} de la varianza total de los datos.")
        if i < len(varianza_explicada) - 1:
            print(
                f"Combinando este componente con el/los siguiente(s), se explica el {varianza_acumulada[i]:.2f} de la varianza total.")

    # Nota: Esta es una interpretación básica. Para una interpretación más detallada, puedes examinar los vectores propios (cargas de los componentes) para entender qué variables contribuyen más a cada componente principal.


def interpretar_series_temporales(modelo_series_temporales, df, columna_fecha, columna_datos):
    # Tendencias y estacionalidad
    print("Análisis de Tendencias y Estacionalidad:")
    # Aquí puedes discutir los resultados de cualquier descomposición de tendencias y estacionalidad que hayas realizado

    # Calidad del modelo
    print("\nCalidad del Modelo:")
    # Aquí puedes discutir métricas como AIC, BIC, RMSE, etc., dependiendo de lo que hayas calculado
    # Por ejemplo:
    print(f"AIC: {modelo_series_temporales.aic}")
    print(f"BIC: {modelo_series_temporales.bic}")

    # Predicciones
    print("\nPredicciones del Modelo:")
    # Aquí puedes mostrar un gráfico de las predicciones del modelo en comparación con los datos reales
    # Por ejemplo:
    #df['Predicciones'] = modelo_series_temporales.predict(start=fecha_inicio, end=fecha_fin)
    df[[columna_datos, 'Predicciones']].plot()

    # Interpretación de las predicciones
    # Por ejemplo:
    print("Las predicciones del modelo muestran que se espera que la tendencia observada continúe/descienda/aumente.")

    # Nota: Esta es una interpretación básica. La interpretación real puede requerir un análisis más detallado, incluyendo la revisión de los residuos, la estabilidad del modelo a lo largo del tiempo, etc.


class PDF(FPDF):
    def header(self):
        # Puedes agregar un encabezado aquí
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Informe de Análisis de Datos', 0, 1, 'C')

    def footer(self):
        # Puedes agregar un pie de página aquí
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def add_grafico(self, buffer):
        # Mueve el buffer a la posición inicial
        buffer.seek(0)
        # Lee la imagen desde el buffer
        img = Image.open(buffer)
        # Guarda la imagen en un archivo temporal
        img.save("temp_img.png")
        # Añade la imagen al PDF
        self.image("temp_img.png", x=10, y=8, w=100)
        self.ln(85)  # Mover abajo


def generar_grafico():
    # Genera el gráfico
    plt.plot([1, 2, 3], [4, 5, 6])
    # Guarda el gráfico en un buffer de memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    lista_de_graficos.append(buf)
    plt.close()
    return buf


def generar_informe(resultados, graficos):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    for titulo, contenido in resultados.items():
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, titulo, 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, contenido)

    for buffer in graficos:
        pdf.add_grafico(buffer)

    pdf.output('informe_analisis.pdf')


def main():
    df = carga_base_de_datos() if input(
        "¿Desea cargar un dataset desde la API? (s/n): ").lower() == 's' else carga_archivo()
    revisar_dataset(df)
    df_sin_atipicos = eliminar_atipicos(df) if encontrar_atipicos(df) else df
    estadisticas = estadisticas_descriptivas(df_sin_atipicos)

    opcion_seleccionada = ofrecer_analisis_profundo(df_sin_atipicos)
    resultados_analisis = realizar_analisis_profundo(
        df_sin_atipicos, opcion_seleccionada)

    interpretaciones = interpretar_analisis(
        df_sin_atipicos, resultados_analisis)

    # Preparar los resultados para el informe
    resultados_para_informe = {
        "Estadísticas Descriptivas": estadisticas,
        "Resultados del Análisis": resultados_analisis,
        "Interpretaciones": interpretaciones
    }

    generar_informe(resultados_para_informe)


if __name__ == "__main__":
    main()
