# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.offline as pyo
import requests
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from itertools import combinations
import os

import tensorflow as tf

# Funciones


def menu_principal():
    while True:
        print("\nMENU")
        print("1. Estadistica Basica")
        print("2. Estadistica avanzada")
        print("3. Analisis predictivo")
        print("4. Analisis for Dummies")
        print("5. Cargar desde archivo")
        print("6. Cargar desde base de datos")
        print("7. Salir\n")

        opcion = input("Ingrese una opcion: ")

        if opcion == "1":
            try:
                menu_estadistica_basica(df)
            except NameError:
                print("No se ha cargado ninguna tabla. Carguela e intentelo de nuevo.")

        elif opcion == "2":
            try:
                menu_estadistica_avanzada(df)
            except NameError:
                print("No se ha cargado ningun archivo. Intente de nuevo.")

        elif opcion == "3":
            try:
                menu_analisis_predictivo(df)
            except NameError:
                print("No se ha cargado ningun archivo. Intente de nuevo.")

        elif opcion == "4":
            try:
                analisis_for_dummies(df)
            except NameError:
                print("No se ha cargado ningun archivo. Intente de nuevo.")

        elif opcion == "5":
            df = carga_archivo()
            if df is not None:
                print("Archivo cargado exitosamente.")
                # Realizar operaciones con el dataframe cargado
            else:
                print("No se pudo cargar el archivo.")

        elif opcion == "6":
            df = carga_base_de_datos()
            if df is not None:
                print("Datos cargados exitosamente.")
                # Realizar operaciones con el dataframe cargado
            else:
                print("No se pudo cargar los datos.")

        elif opcion == "7":
            print("Saliendo del programa...")
            salir = True
            return salir


def menu_estadistica_basica(df):
    while True:
        # Codigo para estadistica basica
        print("\nHa seleccionado Estadistica Basica")
        print("1. Media")
        print("2. Mediana")
        print("3. Moda")
        print("4. Desviacion estandar")
        print("5. Varianza")
        print("6. Asimetria")
        print("7. Curtosis")
        print("8. Estadisticas descriptivas")
        print("9. Comparar columnas")
        print("10. Volver al menu principal")

        opcion = input("Ingrese una opcion: ")

        if opcion == "1":
            # Codigo para calcular la media
            mean = df.mean()
            print("\nLa media es:", mean, sep="\n")

        elif opcion == "2":
            # Codigo para calcular la mediana
            median = df.median()
            print("\nLa mediana es: ", median, sep="\n")

        elif opcion == "3":
            # Codigo para calcular la moda
            mode = df.mode().iloc[0]
            print("\nLa moda es: ", mode, sep="\n")

        elif opcion == "4":
            # Codigo para calcular la desviacion estandar
            std = df.std()
            print("\nLa desviacion estandar es: ", std, sep="\n")

        elif opcion == "5":
            # Codigo para calcular la varianza
            var = df.var()
            print("\nLa varianza es: ", var, sep="\n")

        elif opcion == "6":
            # Codigo para calcular la asimetria
            skew = df.skew()
            print("\nLa asimetria es: ", skew, sep="\n")

        elif opcion == "7":
            # Codigo para calcular la curtosis
            kurt = df.kurt()
            print("\nLa curtosis es: ", kurt, sep="\n")

        elif opcion == "8":
            # Codigo para calcular estadisticas descriptivas
            desc = df.describe()
            print("\nLas estadisticas descriptivas son: ", desc, sep="\n")

        elif opcion == "9":
            # Codigo para comparar columnas
            comparar_dos_columnas(df)

        elif opcion == "10":
            print("\nVolviendo al menu principal... ")
            break

        else:
            print("Opcion invalida. Intente de nuevo.")
            continue


def print_menu_in_columns(menu_options, num_columns):
    # Obtiene cuántas filas se necesitan basado en el número de columnas
    num_rows = -(-len(menu_options) // num_columns)

    # Itera sobre las filas y las columnas para imprimir el menú
    for row in range(num_rows):
        row_items = []
        for col in range(num_columns):
            index = row + col * num_rows
            if index < len(menu_options):
                # Aquí obtenemos el tamaño máximo de las opciones en esta columna para alinear correctamente
                col_max_length = max(len(menu_options[i]) for i in range(
                    col * num_rows, min((col + 1) * num_rows, len(menu_options))))
                row_items.append(
                    f"{index + 1}. {menu_options[index]:<{col_max_length}}")
        print("   ".join(row_items))


def menu_estadistica_avanzada(df):
    while True:
        # Codigo para estadistica avanzada
        print("\nHa seleccionado Estadistica avanzada\n")

        menu_options = [
            "Histograma", "Cajas y Bigotes", "Diagrama de Dispersión", "Correlacion", "Covarianza",
            "Z score", "Correlacion de pearson", "Correlacion de spearman", "Correlacion de kendall",
            "Volver al menu principal"
        ]

        print_menu_in_columns(menu_options, 3)

        opcion = input("\nIngrese una opcion: ")

        if opcion == "1":
            # Codigo para graficar histograma
            seleccionar_columnas(df, "histograma")

        elif opcion == "2":
            # Codigo para graficar boxplot
            seleccionar_columnas(df, "boxplot")

        elif opcion == "3":
            # Codigo para graficar diagrama de dispersion
            seleccionar_columnas(df, "dispersion")

        elif opcion == "4":
            # Codigo para calcular la correlacion
            corr = df.corr()
            print("La correlacion es: ", corr)

        elif opcion == "5":
            # Codigo para calcular la covarianza
            cov = df.cov()
            print("La covarianza es: ", cov)

        elif opcion == "6":
            # Codigo para calcular el z score
            z = stats.zscore(df)
            print("El z score es: ", z)

        elif opcion == "7":
            # Codigo para calcular la correlacion de pearson
            corr = seleccionar_columnas(df, "pearson")

        elif opcion == "8":
            # Codigo para calcular la correlacion de spearman
            spearman = seleccionar_columnas(df, "spearman")

        elif opcion == "9":
            #   Codigo para calcular la correlacion de kendall
            kendall = seleccionar_columnas(df, "kendall")

        elif opcion == "10":
            print("Volviendo al menu principal...")
            break

        else:
            print("Opcion invalida. Intente de nuevo.")
            continue


def menu_analisis_predictivo(df):
    while True:
        # Codigo para analisis predictivo
        print("\nHa seleccionado Analisis predictivo")
        print("1. Regresion lineal")
        print("2. Regresion multiple")
        print("3. Regresion logistica - Clasificacion")
        print("4. Distribucion de poisson")
        print("5. Distribucion exponencial")
        print("6. Chi cuadrado")
        print("7. Anova")
        print("8. Correlacion de pearson")
        print("9. Correlacion de spearman")
        print("10. Correlacion de kendall")
        print("11. Chi cuadrado")
        print("12. Anova")
        print("13. Volver al menu principal")

        opcion = input("Ingrese una opcion: ")

        if opcion == "1":
            # Codigo para calcular la regresion lineal
            reg = seleccionar_columnas(df, "lregresion")

        elif opcion == "2":
            # Codigo para calcular la regresion multiple
            reg = seleccionar_columnas(df, "mregresion")

        elif opcion == "3":
            # Codigo para calcular la regresion logistica
            reg = seleccionar_columnas(df, "llogistica")

        elif opcion == "4":
            # Codigo para calcular la distribucion de poisson
            poisson = stats.poisson(df)
            print("La distribucion de poisson es: ", poisson)

        elif opcion == "5":
            # Codigo para calcular la distribucion exponencial
            exp = stats.expon(df)
            print("La distribucion exponencial es: ", exp)

        elif opcion == "6":
            # Codigo para calcular el chi cuadrado
            chi = stats.chisquare(df)
            print("El chi cuadrado es: ", chi)

        elif opcion == "7":
            # Codigo para calcular el anova
            anova = stats.f_oneway(df)
            print("El anova es: ", anova)

        elif opcion == "8":
            # Codigo para calcular la correlacion de pearson
            pearson = seleccionar_columnas(df, "pearson")
            print("La correlacion de pearson es: ", pearson)

        elif opcion == "9":
            # Codigo para calcular la correlacion de spearman
            spearman = seleccionar_columnas(df, "spearman")
            print("La correlacion de spearman es: ", spearman)

        elif opcion == "10":
            #   Codigo para calcular la correlacion de kendall
            kendall = seleccionar_columnas(df, "kendall")
            print("La correlacion de kendall es: ", kendall)

        elif opcion == "11":
            # Codigo para calcular el chi cuadrado
            chi = stats.chisquare(df)
            print("El chi cuadrado es: ", chi)

        elif opcion == "12":
            # Codigo para calcular el anova
            anova = stats.f_oneway(df)
            print("El anova es: ", anova)

        elif opcion == "13":
            print("Volviendo al menu principal...")
            break

        else:
            print("Opcion invalida. Intente de nuevo.")
            continue


def carga_base_de_datos():
    # 1. Realizar petición para obtener la lista de datasets
    response = requests.get("http://127.0.0.1:8000/datasets")
    datasets = response.json()["datasets"]

    # 2. Mostrar lista de datasets y pedir al usuario que seleccione uno
    print("\nLista de datasets disponibles:")
    for idx, dataset in enumerate(datasets):
        print(f"{idx}. {dataset['name']} (ID: {dataset['id']})")

    selected_idx = int(
        input("\nPor favor, selecciona el índice del dataset que deseas cargar: "))
    selected_dataset_id = datasets[selected_idx]["id"]

    # 3. Realizar petición para obtener una vista previa del dataset seleccionado
    response = requests.get(
        f"http://127.0.0.1:8000/preview/dataset/{selected_dataset_id}")

    # Verifica si la petición fue exitosa
    if response.status_code == 200:
        preview_content = response.json()
        preview_df = pd.DataFrame(preview_content["preview"])
        print("\nVista previa del dataset:")
        print(preview_df)

        # Preguntar al usuario si desea descargar el dataset completo
        decision = input(
            "\n¿Deseas descargar el dataset completo? (s/n): ").lower()
        if decision == 's':
            response = requests.get(
                f"http://127.0.0.1:8000/download/dataset/{selected_dataset_id}")
            if response.status_code == 200:
                dataset_content = response.json()
                df = pd.DataFrame(dataset_content["content"])
                print("\nDataset descargado con éxito!")
                return df
            else:
                print(
                    f"Error al descargar el dataset completo. Código de respuesta: {response.status_code}. Mensaje: {response.text}")
                return None
        else:
            print("Operación cancelada por el usuario.")
            return None
    else:
        print(
            f"Error al obtener la vista previa del dataset. Código de respuesta: {response.status_code}. Mensaje: {response.text}")
        return None


def carga_archivo():

    # Codigo para carga de archivo
    ruta_archivo = input("Ingrese la ruta del archivo: ")
    extension = ruta_archivo.split(".")[-1]  # Obtener la extension del archivo

    # Verificar que la extension sea csv
    try:
        assert extension == "csv" or extension == "xlsx" or extension == "json"
    except AssertionError:
        print("Extension invalida. Intente de nuevo.")
        return

    if extension == "csv":
        df = pd.read_csv(ruta_archivo)
        return df

    elif extension == "xlsx":
        df = pd.read_excel(ruta_archivo)
        return df

    elif extension == "json":
        df = pd.read_json(ruta_archivo)
        return df

    elif ruta_archivo == "":
        print("No se ingreso una ruta de archivo. Intente de nuevo.")
        return

    else:

        if ruta_archivo == "":
            print("No se ingreso una ruta de archivo. Intente de nuevo.")
            return


def basic_statistics(df):

    # Codigo para estadistica basica
    mean = df.mean()
    median = df.median()
    mode = df.mode()
    std = df.std()

    print("La media es: ", mean)
    print("La mediana es: ", median)
    print("La moda es: ", mode)
    print("La desviacion estandar es: ", std)


def seleccionar_columnas(df, grafico):
    print("\nColumnas disponibles en el DataFrame:")
    for idx, column in enumerate(df.columns, 1):
        print(f"{idx}. {column}")

    print(f"{len(df.columns) + 1}. Todas las columnas")

    column_selection = input(
        f"\nSeleccione las columnas para el {grafico} (separadas por comas) o 'Todas las columnas': ")

    # Si el usuario selecciona 'Todas las columnas'
    if column_selection == str(len(df.columns) + 1):
        if grafico == "histograma":

            df.hist(figsize=(15, 10))

        elif grafico == "dispersion":
            print('solo puedes elegir dos columnas')
            x = input("Ingrese la columna para el eje x: ")
            y = input("Ingrese la columna para el eje y: ")
            df.plot.scatter(x=x, y=y)

        elif grafico == "pearson":
            print("Lista de columnas:")
            for idx, col in enumerate(df.columns, start=1):
                print(f"{idx}. {col}")

            column_input = input(
                "Seleccione las columnas (por número, separadas por comas): ")

            # Intenta convertir la entrada en índices válidos
            try:
                selected_indices = [
                    int(idx) - 1 for idx in column_input.split(',')]
                selected_columns = [df.columns[idx]
                                    for idx in selected_indices]
            except ValueError:
                print(
                    "Entrada inválida. Asegúrese de ingresar solo números, separados por comas.")
                return

            correlaciones = df.corr(method="pearson")[selected_columns]
            print(correlaciones)

        elif grafico == "spearman":
            print("Lista de columnas:")
            for idx, col in enumerate(df.columns, start=1):
                print(f"{idx}. {col}")

            column_input = input(
                "Seleccione las columnas (por número, separadas por comas): ")

            # Intenta convertir la entrada en índices válidos
            try:
                selected_indices = [
                    int(idx) - 1 for idx in column_input.split(',')]
                selected_columns = [df.columns[idx]
                                    for idx in selected_indices]
            except ValueError:
                print(
                    "Entrada inválida. Asegúrese de ingresar solo números, separados por comas.")
                return

            correlaciones = df.corr(method="spearman")[selected_columns]
            print(correlaciones)

        elif grafico == "kendall":
            print("Lista de columnas:")
            for idx, col in enumerate(df.columns, start=1):
                print(f"{idx}. {col}")

            column_input = input(
                "Seleccione las columnas (por número, separadas por comas): ")

            # Intenta convertir la entrada en índices válidos
            try:
                selected_indices = [
                    int(idx) - 1 for idx in column_input.split(',')]
                selected_columns = [df.columns[idx]
                                    for idx in selected_indices]
            except ValueError:
                print(
                    "Entrada inválida. Asegúrese de ingresar solo números, separados por comas.")
                return

            correlaciones = df.corr(method="kendall")[selected_columns]
            print(correlaciones)

        elif grafico == "lregresion":

            print('No se puede hacer la regresion lineal con todas las columnas')

        elif grafico == "mregresion":

            # Tomar todas las columnas menos la seleccionada como variables independientes
            x = df.drop(columns=[df.columns[int(y_column) - 1]])
            # Tomar la columna seleccionada como la variable dependiente
            y = df[df.columns[int(y_column) - 1]]

            # Dividir el dataset en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=0)

            # Crear modelo de regresión lineal
            reg = LinearRegression().fit(x_train, y_train)

            # Predecir y para el conjunto de prueba
            y_pred = reg.predict(x_test)

            # Evaluar el modelo
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Resultados
            print(f"\nError Cuadrático Medio (MSE): {mse:.2f}")
            print(f"Coeficiente de Determinación (R^2): {r2:.2f}")
            print("\nSi R^2 es cercano a 1, significa que las variables seleccionadas explican bien la variabilidad de la variable dependiente.")
            print("\nInterpretación de Coeficientes:")
            print("\nIntercepto (valor esperado de Y cuando X=0):", reg.intercept_)
            for feature, coef in zip(x.columns, reg.coef_):
                print(f"{feature}: {coef}")
            print("\nUn coeficiente positivo indica que a medida que la variable independiente aumenta, la variable dependiente también tiende a aumentar, y viceversa para un coeficiente negativo.")

        elif grafico == "llogistica":

            # Mostrar las columnas y pedir al usuario que seleccione la columna dependiente
            print("\n".join([f"{idx+1}. {col}" for idx,
                  col in enumerate(df.columns)]))

            y_column = input(
                "Seleccione el número de la columna para la variable dependiente: ")
            x = df.drop(columns=df.columns[int(y_column) - 1])
            y = df[df.columns[int(y_column) - 1]]

            # Dividir el dataset en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=0)

            # Crear modelo de regresión logística
            logreg = LogisticRegression(max_iter=10000).fit(
                x_train, y_train)  # Aumentamos max_iter por si acaso

            # Predecir y para el conjunto de prueba
            y_pred = logreg.predict(x_test)

            # Evaluar el modelo
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            # Resultados
            print(f"\nPrecisión del modelo: {acc:.2f}")
            print(f"Reporte de Clasificación:\n{report}")

            # Solicitar datos de usuario para hacer una predicción
            print("\nIngrese los valores para las siguientes variables:")
            user_data = []
            for col in x.columns:
                # Asumimos que todos los datos son numéricos
                val = float(input(f"{col}: "))
                user_data.append(val)
            user_data_array = np.array(user_data).reshape(1, -1)
            probability = logreg.predict_proba(user_data_array)[
                :, 1]  # Probabilidad de la clase 1
            print(
                f"\nLa probabilidad predicha para la clase 1 es: {probability[0]*100:.2f}%")

            # Interpretar los coeficientes
            print("\nInterpretación de Coeficientes:")
            print("\nIntercepto:", logreg.intercept_[0])
            for feature, coef in zip(x.columns, logreg.coef_[0]):
                print(f"{feature}: {coef}")
            print('''\nNota: Un coeficiente positivo indica que a medida que la variable {feature} aumenta,
                la log-odds de la clase 1 aumenta (y por lo tanto la probabilidad de la clase 1 aumenta).
                Un coeficiente negativo tiene el efecto opuesto.''')

        else:
            df.boxplot(figsize=(15, 10))
        plt.tight_layout()
        plt.show()
    else:
        try:
            # Convierte la selección del usuario en una lista de índices de columnas
            selected_indices = [
                int(idx) - 1 for idx in column_selection.split(',')]

            # Convierte los índices de columnas en nombres de columnas
            selected_columns = [df.columns[idx] for idx in selected_indices]

            if grafico == "histograma":

                df[selected_columns].hist(figsize=(15, 10))
                plt.tight_layout()
                plt.show()

            elif grafico == "dispersion":

                df[selected_columns].plot.scatter(
                    x=selected_columns[0], y=selected_columns[1])
                plt.tight_layout()
                plt.show()

            elif grafico == "pearson":
                pearson = stats.pearsonr(
                    df[selected_columns[0]], df[selected_columns[1]])
                return pearson

            elif grafico == "spearman":
                spearman = stats.spearmanr(
                    df[selected_columns[0]], df[selected_columns[1]])
                return spearman

            elif grafico == "kendall":
                kendall = stats.kendalltau(
                    df[selected_columns[0]], df[selected_columns[1]])
                return kendall

            elif grafico == "lregresion":
                x = df[selected_columns[0]].values.reshape(-1, 1)
                y = df[selected_columns[1]]

                # Crear modelo de regresión lineal
                reg = LinearRegression().fit(x, y)

                # Precedir y
                y_pred = reg.predict(x)

                # Gráfico
                plt.scatter(x, y, color='blue')
                plt.plot(x, y_pred, color='red')
                # Aquí se usa el nombre de la columna para X
                plt.xlabel(selected_columns[0])
                # Aquí se usa el nombre de la columna para Y
                plt.ylabel(selected_columns[1])
                plt.title('Regresión Lineal Simple')
                plt.show()

            elif grafico == "mregresion":

                # Mostrar las columnas y pedir al usuario que seleccione las columnas
                print("\n".join([f"{idx+1}. {col}" for idx,
                      col in enumerate(df.columns)]))

                x_columns = input(
                    "Seleccione los números de las columnas para las variables independientes separados por comas: ").split(',')
                y_column = input(
                    "Seleccione el número de la columna para la variable dependiente: ")

                # Convertir las entradas del usuario a índices de columnas
                x = df[df.columns[[int(idx) - 1 for idx in x_columns]]]
                y = df[df.columns[int(y_column) - 1]]

                # Dividir el dataset en conjuntos de entrenamiento y prueba
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.2, random_state=0)

                # Crear modelo de regresión lineal
                reg = LinearRegression().fit(x_train, y_train)

                # Predecir y para el conjunto de prueba
                y_pred = reg.predict(x_test)

                # Evaluar el modelo
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Resultados
                print(f"\nError Cuadrático Medio (MSE): {mse:.2f}")
                print(f"Coeficiente de Determinación (R^2): {r2:.2f}")
                print("\nSi R^2 es cercano a 1, significa que las variables seleccionadas explican bien la variabilidad de la variable dependiente.")
                print("\nInterpretación de Coeficientes:")
                print("\nIntercepto (valor esperado de Y cuando X=0):",
                      reg.intercept_)
                for feature, coef in zip(x.columns, reg.coef_):
                    print(f"{feature}: {coef}")
                print("\nUn coeficiente positivo indica que a medida que la variable independiente aumenta, la variable dependiente también tiende a aumenta. Un coeficiente negativo indica lo contrario.")

            elif grafico == "llogistica":

                # Mostrar las columnas y pedir al usuario que seleccione la columna dependiente
                print("\nColumnas disponibles:")
                print("\n".join([f"{idx+1}. {col}" for idx,
                      col in enumerate(df.columns)]))

                y_column = input(
                    "Seleccione el número de la columna para la variable dependiente: ")
                y = df[df.columns[int(y_column) - 1]]

                x_columns = input(
                    "Seleccione los números de las columnas para las variables independientes (separados por comas): ")
                x_columns_list = [
                    int(xcol) - 1 for xcol in x_columns.split(',')]
                x = df[df.columns[x_columns_list]]

                # Dividir el dataset en conjuntos de entrenamiento y prueba
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.2, random_state=0)

                # Crear modelo de regresión logística
                logreg = LogisticRegression(max_iter=10000).fit(
                    x_train, y_train)  # Aumentamos max_iter por si acaso

                # Predecir y para el conjunto de prueba
                y_pred = logreg.predict(x_test)

                # Evaluar el modelo
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)

                # Resultados
                print(f"\nPrecisión del modelo: {acc:.2f}")
                print(f"Reporte de Clasificación:\n{report}")

                # Solicitar datos de usuario para hacer una predicción
                print("\nIngrese los valores para las siguientes variables:")
                user_data = []
                for col in x.columns:
                    # Asumimos que todos los datos son numéricos
                    val = float(input(f"{col}: "))
                    user_data.append(val)
                user_data_array = np.array(user_data).reshape(1, -1)
                probability = logreg.predict_proba(user_data_array)[
                    :, 1]  # Probabilidad de la clase 1
                print(
                    f"\nLa probabilidad predicha para la clase 1 es: {probability[0]*100:.2f}%")

                # Interpretar los coeficientes
                print("\nInterpretación de Coeficientes:")
                print("\nIntercepto:", logreg.intercept_[0])
                for feature, coef in zip(x.columns, logreg.coef_[0]):
                    print(f"{feature}: {coef}")
                print('''\nNota: Un coeficiente positivo indica que a medida que la variable {feature} aumenta,
                      la log-odds de la clase 1 aumenta (y por lo tanto la probabilidad de la clase 1 aumenta).
                      Un coeficiente negativo tiene el efecto opuesto.''')

            else:

                df[selected_columns].boxplot(figsize=(15, 10))
                plt.tight_layout()
                plt.show()

        except (ValueError, IndexError):
            print("Selección no válida. Por favor, intente nuevamente.")

# Funciones para interpretar estadisticas descriptivas


def interpretar_promedio(serie):
    promedio = serie.mean()
    mensaje = f"El promedio de los datos es {promedio:.2f}. Esto representa el valor central de tus datos. "
    if promedio > 0:
        mensaje += f"Un valor de {promedio:.2f} indica que, en promedio, este es el valor típico que puedes esperar."
    else:
        mensaje += "Un valor negativo podría indicar que tienes datos atípicos negativos o que, en general, tus datos son negativos."
    return mensaje


def interpretar_mediana(serie):
    mediana = serie.median()
    mensaje = f"La mediana de los datos es {mediana:.2f}. Esto representa el valor que se encuentra justo en el medio de tus datos cuando están ordenados. "
    if mediana > 0:
        mensaje += f"Esto significa que la mitad de tus datos son menores o iguales a {mediana:.2f} y la otra mitad son mayores."
    else:
        mensaje += "Esto indica que la mitad de tus datos son negativos."
    return mensaje


def interpretar_variabilidad(serie):
    coef_var = (serie.std() / serie.mean()) * 100
    mensaje = f"El coeficiente de variación es del {coef_var:.2f}%. "
    if coef_var < 10:
        mensaje += "Esto indica que los datos tienen una variabilidad baja y son bastante consistentes alrededor del promedio."
    elif 10 <= coef_var < 30:
        mensaje += "Esto indica una variabilidad moderada en tus datos."
    else:
        mensaje += "Hay una alta variabilidad en tus datos, lo que significa que pueden haber valores extremos o una amplia dispersión alrededor del promedio."
    return mensaje


def seleccionar_columnas_for_dummies(df, grafico):
    print("\nColumnas disponibles en el DataFrame:")
    for idx, column in enumerate(df.columns, 1):
        print(f"{idx}. {column}")

    print(f"{len(df.columns) + 1}. Todas las columnas")

    column_selection = input(
        f"\nSeleccione las columnas para el {grafico} (separadas por comas) o 'Todas las columnas': ")

    try:
        # Convierte la selección del usuario en una lista de índices de columnas
        selected_indices = [
            int(idx) - 1 for idx in column_selection.split(',')]

        # Si el usuario selecciona "Todas las columnas"
        if len(df.columns) in selected_indices:
            return df.columns.tolist()

        # Convierte los índices de columnas en nombres de columnas
        selected_columns = [df.columns[idx] for idx in selected_indices]
        return selected_columns

    except ValueError:
        print("Selección no válida. Intente de nuevo.")
        return seleccionar_columnas_for_dummies(df, grafico)


def analisis_for_dummies(df):
    columnas_seleccionadas = seleccionar_columnas_for_dummies(df, "dummies")

    analizar_valores_faltantes(df)

    for columna in columnas_seleccionadas:
        data_columna = df[columna]

        if pd.api.types.is_numeric_dtype(data_columna):
            print("-" * 40)
            print(f"\nResumen rápido para la columna: {columna}\n{'-'*40}")
            print("-" * 40)
            print(resumen_metricas(data_columna))
            print("-" * 40)
            print(f"\nAnálisis para la columna: {columna}:\n{'='*40}")
            print("-" * 40)
            print(interpretar_promedio(data_columna))
            print("-" * 40)
            print(interpretar_mediana(data_columna))
            print("-" * 40)
            print(interpretar_variabilidad(data_columna))
            print("-" * 40)
            print(interpretar_moda(data_columna))
            print("-" * 40)
            print(interpretar_cuartiles(data_columna))
            print("-" * 40)
            # Interpretación de valores atípicos
            print("\nAnálisis de valores atípicos:\n" + "="*50)
            print("-" * 40)
            print(interpretar_outliers(data_columna))
            print("-" * 40)
            print(interpretar_asimetria(data_columna))
            print("-" * 40)
            print(interpretar_kurtosis(data_columna))
            print("-" * 40)
            mostrar_boxplot(data_columna, columna, columnas_seleccionadas)

            if len(columnas_seleccionadas) > 1:
                for i, col_x in enumerate(columnas_seleccionadas):
                    for j, col_y in enumerate(columnas_seleccionadas):
                        if i < j:  # Para evitar hacer la regresión de una columna consigo misma o repetirla
                            reg = realizar_regresion(df, col_x, col_y)

                comparar_columnas(df, columnas_seleccionadas)
                print("-" * 40)
                print(
                    "\nAnálisis de correlación entre columnas seleccionadas:\n" + "="*50)
                print(interpretar_correlacion(df, columnas_seleccionadas))
                print("-" * 40)
                print("\nAnálisis multivariante:\n" + "="*50)
                print("-" * 40)
                mostrar_mapa_calor(df, columnas_seleccionadas)
                mostrar_pairplot(df, columnas_seleccionadas)
                if len(columnas_seleccionadas) == 3:
                    # Suponiendo que estamos considerando exactamente 3 columnas por ahora
                    mostrar_crosstab_multivariante(df, *columnas_seleccionadas)

                # Genera graficos SCATTERPLOT
                mostrar_scatterplot(
                    df, columnas_seleccionadas[:], columnas_seleccionadas)

            # Genera graficos
            mostrar_histograma(data_columna)
            # Genera graficos
            generar_graficos_for_dummies(df, columnas_seleccionadas)

        else:
            # Si no es numérico, muestra los valores únicos y sus frecuencias
            value_counts = data_columna.value_counts()
            print("Frecuencias de valores únicos:\n", value_counts)
            mostrar_barplot(data_columna, columna)
            mostrar_pieplot(data_columna, columna)

            if len(columnas_seleccionadas) > 1:
                # Solo para 2 columnas de ejemplo
                mostrar_crosstab(
                    df, columnas_seleccionadas[0], columnas_seleccionadas[1])

    # Análisis temporal (si es aplicable)
    realizar_analisis_temporal = input(
        "¿Desea realizar un análisis temporal? (s/n): ").lower()
    if realizar_analisis_temporal == 's':
        analizar_series_temporales(df)

    # Clasificación simple
    realizar_clasificacion = input(
        "¿Desea realizar una clasificación simple? (s/n): ").lower()
    if realizar_clasificacion == 's':
        clasificador_simple(df)

    # Predicción simple
    realizar_prediccion = input(
        "¿Desea realizar una predicción simple? (s/n): ").lower()
    if realizar_prediccion == 's' and len(columnas_seleccionadas) > 1:
        col_x = input(
            f"Seleccione la columna independiente de entre {', '.join(columnas_seleccionadas)}: ")
        col_y = input(
            f"Seleccione la columna dependiente de entre {', '.join(columnas_seleccionadas)}: ")
        realizar_prediccion_simple(df, col_x, col_y)

    if realizar_prediccion == 's' and len(columnas_seleccionadas) > 1:
        col_x = input(
            f"Seleccione la columna independiente de entre {', '.join(columnas_seleccionadas)}: ")
        col_y = input(
            f"Seleccione la columna dependiente de entre {', '.join(columnas_seleccionadas)}: ")
        task_type = recomendar_modelo(df, col_y)
        if task_type == "regression":
            model = RandomForestRegressor()
        else:
            model = RandomForestClassifier()
        X = df.drop(columns=col_y)
        y = df[col_y]
        model.fit(X, y)
        print("\nImportancia de Características:\n" + "="*50)
        mostrar_importancia_caracteristicas(model, X.columns)

    desea_anomalias = input(
        "¿Desea detectar anomalías en el conjunto de datos? (s/n): ").lower()
    if desea_anomalias == 's':
        anomalias = detectar_anomalias(df)
        # Aquí puedes optar por mostrar estas anomalías o guardarlo para más análisis

    desea_pca = input(
        "¿Desea visualizar el conjunto de datos después de aplicar PCA? (s/n): ").lower()
    if desea_pca == 's':
        aplicar_pca_y_visualizar(df)

    # Análisis de Componentes Principales
    realizar_pca = input(
        "¿Desea realizar un análisis de componentes principales (PCA)? (s/n): ").lower()
    if realizar_pca == 's':
        # Aquí simplemente vamos a reducir a 2D para visualización
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df.select_dtypes(include=[np.number]))
        print(
            f"Varianza explicada por los componentes: {pca.explained_variance_ratio_}")
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.title('Análisis de Componentes Principales (2D)')
        plt.show()

    # Clustering
    realizar_clustering = input(
        "¿Desea realizar un análisis de clustering (K-Means)? (s/n): ").lower()
    if realizar_clustering == 's':
        n_clusters = int(input("Ingrese el número de clusters deseado: "))
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(df.select_dtypes(include=[np.number]))
        if realizar_pca == 's':  # Si ya hicimos PCA, mostramos los clusters en ese espacio
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters)
            plt.xlabel('Componente 1')
            plt.ylabel('Componente 2')
            plt.title('Clustering basado en PCA (2D)')
            plt.show()
        else:  # Si no, simplemente mostramos los tamaños de los clusters
            unique, counts = np.unique(clusters, return_counts=True)
            print(dict(zip(unique, counts)))

    # Red Neuronal
    realizar_red_neuronal = input(
        "¿Desea entrenar una red neuronal? (s/n): ").lower()
    if realizar_red_neuronal == 's':
        tipo = input(
            "¿Desea hacer clasificación o regresión con la red neuronal? (clasificacion/regresion): ").lower()
        entrenar_red_neuronal(df, tipo)

    # Generar conclusiones generales

    generar_conclusiones_generales(df, columnas_seleccionadas)


# Funciones para interpretar estadisticas descriptivas


def interpretar_moda(columna):
    moda = columna.mode()
    if len(moda) == 0:
        return "No hay ningún valor que se repita con mayor frecuencia que los demás."
    elif len(moda) == 1:
        return f"El valor que más se repite en los datos es {moda.iloc[0]}, lo cual podría indicar su prevalencia o popularidad."
    else:
        valores_moda = ", ".join([str(v) for v in moda])
        return f"Existen varios valores que se repiten con mayor frecuencia: {valores_moda}. Estos valores podrían indicar tendencias o patrones recurrentes."


def mostrar_histograma(serie):
    plt.hist(serie, bins=30, edgecolor='black', alpha=0.7)
    plt.title("Distribución de los datos")
    plt.xlabel("Valores")
    plt.ylabel("Frecuencia")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def interpretar_moda(data):
    moda = data.mode()
    if len(moda) > 1:
        return f"Los datos tienen múltiples modas: {', '.join(map(str, moda))}. Estos valores son los que aparecen con más frecuencia."
    else:
        return f"La moda de los datos es {moda[0]}. Esto significa que es el valor que aparece con más frecuencia."


def interpretar_cuartiles(data):
    q1 = data.quantile(0.25)
    q2 = data.median()
    q3 = data.quantile(0.75)
    return f"Los cuartiles de los datos son Q1: {q1}, Mediana (Q2): {q2}, Q3: {q3}. Esto nos da una idea de cómo se distribuyen los valores."


def interpretar_outliers(data):
    # Utilizamos el método del rango intercuartílico
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    atipicos = data[(data < lower_bound) | (data > upper_bound)]

    if len(atipicos) == 0:
        return "No se detectan valores atípicos en los datos, lo que indica que todos los valores están relativamente agrupados y no hay puntos extremadamente altos o bajos."
    else:
        return f"Se han detectado {len(atipicos)} valores atípicos. Estos valores extremos pueden afectar algunas estadísticas como el promedio y deberías considerarlos al analizar los datos."


def interpretar_asimetria(data):
    asimetria = data.skew()
    if asimetria > 0:
        return "Los datos tienen una asimetría positiva, lo que significa que están sesgados hacia la derecha."
    elif asimetria < 0:
        return "Los datos tienen una asimetría negativa, lo que significa que están sesgados hacia la izquierda."
    else:
        return "Los datos están simétricamente distribuidos."


def interpretar_kurtosis(data):
    kurtosis_val = data.kurtosis()
    if kurtosis_val > 0:
        return "La distribución de los datos es leptocúrtica, lo que significa que es más puntiaguda que una distribución normal."
    elif kurtosis_val < 0:
        return "La distribución de los datos es platicúrtica, lo que significa que es más plana que una distribución normal."
    else:
        return "La distribución de los datos es mesocúrtica, similar a una distribución normal."


def interpretar_correlacion(df, columnas_seleccionadas):
    # Calculamos la matriz de correlación para las columnas seleccionadas
    matriz_correlacion = df[columnas_seleccionadas].corr()

    interpretaciones = []

    for i, col1 in enumerate(columnas_seleccionadas):
        for j, col2 in enumerate(columnas_seleccionadas):
            # Evitamos repetir combinaciones y comparaciones de una columna consigo misma
            if i < j:
                coef = matriz_correlacion.at[col1, col2]
                if coef > 0.7:
                    interpretaciones.append(
                        f"Existe una fuerte correlación positiva entre {col1} y {col2}. A medida que {col1} aumenta, {col2} tiende a aumentar también.")
                elif coef < -0.7:
                    interpretaciones.append(
                        f"Existe una fuerte correlación negativa entre {col1} y {col2}. A medida que {col1} aumenta, {col2} tiende a disminuir.")
                elif abs(coef) < 0.1:
                    interpretaciones.append(
                        f"{col1} y {col2} tienen poca o ninguna correlación.")

    return "\n".join(interpretaciones)


# Funciones de comparar columnas

def comparar_columnas(df, columnas_seleccionadas):

    for col1, col2 in combinations(columnas_seleccionadas, 2):
        print(f"\nComparación entre {col1} y {col2}")
        print("-" * 40)

        promedio_col1, promedio_col2 = df[col1].mean(), df[col2].mean()

        if promedio_col1 > promedio_col2:
            print(f"El promedio de {col1} es mayor que el de {col2}.")
        elif promedio_col1 < promedio_col2:
            print(f"El promedio de {col2} es mayor que el de {col1}.")
        else:
            print(
                f"Los promedios de {col1} y {col2} son aproximadamente iguales.")

        # Puedes agregar más comparaciones aquí: variabilidad, mediana, etc.


def comparar_dos_columnas(df):
    print("\nColumnas disponibles en el DataFrame para comparar:")
    for idx, column in enumerate(df.columns, 1):
        print(f"{idx}. {column}")

    columna1 = int(input("Seleccione la primera columna para comparar: ")) - 1
    columna2 = int(input("Seleccione la segunda columna para comparar: ")) - 1

    data_columna1 = df[df.columns[columna1]]
    data_columna2 = df[df.columns[columna2]]

    print(
        f"\nComparación entre {df.columns[columna1]} y {df.columns[columna2]}:\n{'='*40}")

    # Comparación de promedios
    print("\nComparación de promedios:")
    promedio1 = data_columna1.mean()
    promedio2 = data_columna2.mean()
    print(f"Promedio de {df.columns[columna1]}: {promedio1:.2f}")
    print(f"Promedio de {df.columns[columna2]}: {promedio2:.2f}")
    if promedio1 > promedio2:
        print(
            f"El promedio de {df.columns[columna1]} es mayor que {df.columns[columna2]} por {(promedio1 - promedio2):.2f}.")
    elif promedio1 < promedio2:
        print(
            f"El promedio de {df.columns[columna2]} es mayor que {df.columns[columna1]} por {(promedio2 - promedio1):.2f}.")
    else:
        print("Los promedios de ambas columnas son iguales.")

    # Podemos seguir agregando más comparaciones: medianas, variabilidad, etc.
    # Por ejemplo:
    print("\nComparación de medianas:")
    mediana1 = data_columna1.median()
    mediana2 = data_columna2.median()
    # ... (similar al bloque anterior para mediana)
    # ... y así sucesivamente para otros estadísticos.

    return

# Generacion de graficos


def generar_graficos_for_dummies(df, columnas_seleccionadas):
    for columna in columnas_seleccionadas:
        # Comprobar que la columna es numérica
        if pd.api.types.is_numeric_dtype(df[columna]):
            plt.figure(figsize=(12, 5))

            # Histograma
            plt.subplot(1, 2, 1)
            sns.histplot(df[columna], kde=True)
            plt.title(f'Histograma de {columna}')
            plt.xlabel(columna)
            plt.ylabel('Frecuencia')

            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(y=df[columna])
            plt.title(f'Boxplot de {columna}')

            plt.tight_layout()
            plt.show()

        else:
            print(
                f"La columna {columna} no es numérica y no se puede graficar.")


def mostrar_boxplot(data_columna, columna, columnas_seleccionadas):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data_columna)
    plt.title(f"Boxplot para {columna}")
    plt.show()


def mostrar_scatterplot(df, columnas, columnas_seleccionadas):
    if len(columnas) < 2:
        return  # No se puede mostrar un scatter plot con menos de 2 columnas

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=columnas[0], y=columnas[1])
    plt.title(f"Scatter plot entre {columnas[0]} y {columnas[1]}")
    plt.show()


def mostrar_barplot(data_columna, columna, columnas_seleccionadas):
    plt.figure(figsize=(10, 6))
    data_columna.value_counts().plot(kind='bar')
    plt.title(f"Frecuencia de categorías para {columna}")
    plt.ylabel('Frecuencia')
    plt.xlabel('Categoría')
    plt.show()


def mostrar_pieplot(data_columna, columna, columnas_seleccionadas):
    plt.figure(figsize=(8, 8))
    data_columna.value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f"Proporción de categorías para {columna}")
    # Eliminamos el ylabel ya que no es necesario en un pie chart
    plt.ylabel('')
    plt.show()


def mostrar_crosstab(df, col1, col2, columnas_seleccionadas):
    ct = pd.crosstab(df[col1], df[col2])
    ct.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f"Relación entre {col1} y {col2}")
    plt.show()


def mostrar_mapa_calor(df, columnas_seleccionadas, size=(10, 8), cmap="coolwarm"):
    correlation_matrix = df[columnas_seleccionadas].corr()
    plt.figure(figsize=(size))
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap)
    plt.title("Mapa de calor de correlación")
    plt.show()


def mostrar_pairplot(df, columnas_seleccionadas):
    sns.pairplot(df[columnas_seleccionadas])
    plt.suptitle('Gráficos de dispersión por pares', y=1.02)
    plt.show()


def mostrar_crosstab_multivariante(df, columnas_seleccionadas, col1, col2, col3=None):
    if col3:  # Si se proporciona una tercera columna, la usamos para colorear
        ct = pd.crosstab(df[col1], df[col2], values=df[col3], aggfunc='mean')
        plt.figure(figsize=(12, 8))
        sns.heatmap(ct, annot=True, cmap="coolwarm")
        plt.title(f"Relación entre {col1}, {col2} y {col3}")
        plt.show()

    else:
        # Si no, usamos la función anterior
        mostrar_crosstab(df, col1, col2, columnas_seleccionadas)


# Machine Learning

def realizar_regresion(df, col_x, col_y):
    X = df[col_x].values.reshape(-1, 1)
    y = df[col_y].values

    reg = LinearRegression().fit(X, y)
    print(f"Coeficiente: {reg.coef_}, Intercepto: {reg.intercept_}")

    plt.scatter(X, y)
    plt.plot(X, reg.predict(X), color='red')
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f"Regresión lineal entre {col_x} y {col_y}")
    plt.show()

    return reg  # Devuelve el modelo entrenado


def realizar_prediccion_simple(df, col_x, col_y):
    modelo = realizar_regresion(df, col_x, col_y)
    nuevo_valor = float(
        input(f"Ingrese un nuevo valor para {col_x} para predecir {col_y}: "))
    prediccion = modelo.predict([[nuevo_valor]])
    print(
        f"Basado en el valor {nuevo_valor} para {col_x}, la predicción para {col_y} es: {prediccion[0]:.2f}")


def clasificador_simple(df):
    # El usuario selecciona la columna objetivo
    columna_objetivo = input(
        f"Seleccione la columna objetivo (etiqueta) para clasificación: {', '.join(df.columns)}: ")
    X = df.drop(columna_objetivo, axis=1).select_dtypes(include='number')
    y = df[columna_objetivo]

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Entrenar un árbol de decisión
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predicciones = clf.predict(X_test)

    # Métricas e interpretación
    precision = accuracy_score(y_test, predicciones)
    print(
        f"La precisión del modelo simple de clasificación es: {precision:.2f}")
    # Podemos agregar más interpretaciones en función de la precisión y otros factores aquí

    # Importancia de características
    importancias = pd.Series(clf.feature_importances_,
                             index=X.columns).sort_values(ascending=False)
    print("\nImportancia de las características:")
    print(importancias)


def recomendar_modelo(df, col_y):
    if pd.api.types.is_numeric_dtype(df[col_y]):
        print(
            "Para esta tarea de regresión, se recomienda usar un Random Forest Regressor.")
        return "regression"
    else:
        print("Para esta tarea de clasificación, se recomienda usar un Random Forest Classifier.")
        return "classification"


def mostrar_importancia_caracteristicas(model, columnas):
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    for i in range(len(importances)):
        if i < 10:  # Mostramos solo las 10 características más importantes
            print(
                f"Característica: {columnas[sorted_idx[i]]}, Importancia: {importances[sorted_idx[i]]:.4f}")


def detectar_anomalias(df):
    print("\nDetección de Anomalías:\n" + "="*50)

    # Asumimos que el 5% de los datos son atípicos
    model = IsolationForest(contamination=0.05)
    preds = model.fit_predict(df)
    indices_atipicos = np.where(preds == -1)
    n_atipicos = len(indices_atipicos[0])

    if n_atipicos > 0:
        print(f"Se detectaron {n_atipicos} datos atípicos.")
        respuesta = input(
            "¿Desea eliminar los datos atípicos? (s/n): ").lower()

        if respuesta == 's':
            df.drop(df.index[indices_atipicos], inplace=True)
            print(f"{n_atipicos} datos atípicos eliminados del DataFrame.")
        else:
            print("Los datos atípicos no fueron eliminados.")
    else:
        print("No se detectaron datos atípicos.")

    return df


def aplicar_pca_y_visualizar(df):
    print("\nAplicación de PCA y Visualización:\n" + "="*50)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data=principalComponents, columns=[
                               'Principal Component 1', 'Principal Component 2'])
    plt.figure(figsize=(10, 6))
    plt.scatter(principalDf['Principal Component 1'],
                principalDf['Principal Component 2'], alpha=0.5)
    plt.title('Visualización 2D después de PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# Series temporales


def analizar_series_temporales(df):
    columnas_fecha = [
        col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

    if not columnas_fecha:
        print("No se identificaron columnas con fechas para análisis temporal.")
        return

    print(f"Columnas de fecha identificadas: {', '.join(columnas_fecha)}")

    # El usuario selecciona la columna de fecha y otra numérica para graficar
    columna_fecha = input(
        f"Seleccione una columna de fecha para el análisis: {', '.join(columnas_fecha)}: ")
    columnas_numericas = df.select_dtypes(include='number').columns.tolist()
    columna_valor = input(
        f"Seleccione una columna numérica para graficar contra {columna_fecha}: {', '.join(columnas_numericas)}: ")

    # Graficar
    df.plot(x=columna_fecha, y=columna_valor, kind="line")
    plt.title(f"Tendencia de {columna_valor} a lo largo del tiempo")
    plt.xlabel(columna_fecha)
    plt.ylabel(columna_valor)
    plt.show()

    # Aquí podemos agregar interpretaciones básicas basadas en la tendencia observada

# Limpieza de tabla


def analizar_valores_faltantes(df):
    total_valores_faltantes = df.isnull().sum().sum()
    if total_valores_faltantes > 0:
        print(
            f"¡Atención! Hay un total de {total_valores_faltantes} valores faltantes en el conjunto de datos.")
        opcion = input(
            "¿Desea eliminar las filas con valores faltantes? (s/n): ").lower()
        if opcion == 's':
            df.dropna(inplace=True)
            print("Filas con valores faltantes eliminadas.")
        else:
            print("Se mantuvieron los valores faltantes en el conjunto de datos.")
    else:
        print("¡Buenas noticias! No hay valores faltantes en el conjunto de datos.")

# Resumen


def resumen_metricas(data_columna):
    resumen = f"Promedio: {data_columna.mean():.2f}\n"
    resumen += f"Mediana: {data_columna.median():.2f}\n"
    resumen += f"Moda: {data_columna.mode().iloc[0]}\n"
    resumen += f"Variabilidad (Desv. Est.): {data_columna.std():.2f}\n"
    resumen += f"Asimetría: {data_columna.skew():.2f}\n"
    resumen += f"Kurtosis: {data_columna.kurtosis():.2f}\n"
    return resumen


def generar_conclusiones_generales(df, columnas_seleccionadas):
    print("\n\nCONCLUSIONES GENERALES")
    print("="*50)

    # Resumen de Datos
    print("\nResumen de Datos:")
    print("-" * 50)
    print(
        f"El conjunto de datos tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
    print(
        f"Se analizaron las siguientes columnas: {', '.join(columnas_seleccionadas)}")

    # Tendencias y Observaciones Notables
    print("\nTendencias y Observaciones Notables:")
    print("-" * 50)

    # Ejemplo: Columna con la media más alta
    columna_max_media = df[columnas_seleccionadas].mean().idxmax()
    print(
        f"La columna '{columna_max_media}' tiene la media más alta de {df[columna_max_media].mean():.2f}.")

    # Ejemplo: Columna con mayor variabilidad (desviación estándar)
    columna_max_variabilidad = df[columnas_seleccionadas].std().idxmax()
    print(
        f"La columna '{columna_max_variabilidad}' tiene la mayor variabilidad con una desviación estándar de {df[columna_max_variabilidad].std():.2f}.")

    # Anomalías o Valores Atípicos
    print("\nAnomalías o Valores Atípicos:")
    print("-" * 50)
    # Aquí podrías mencionar columnas con valores atípicos o anomalías notables.

    # Conclusión General
    print("\nConclusión General:")
    print("-" * 50)
    # Esta es una sección más abstracta y puede requerir un análisis más detallado o inputs adicionales.
    # Pero podrías dar una visión general basada en las tendencias, observaciones y anomalías mencionadas.

    print("\n" + "="*50)


# Redes neuronales

def entrenar_red_neuronal(df, tipo='clasificacion'):
    # Enumerar y mostrar las columnas para el usuario
    print("Columnas disponibles:")
    for i, col_name in enumerate(df.columns, 1):
        print(f"{i}. {col_name}")

    # Solicitar al usuario que seleccione las columnas por número
    column_nums = input(
        "Introduce los números de las columnas características separadas por comas: ").split(',')
    # Convertir números en nombres de columnas
    feature_cols = [df.columns[int(num)-1] for num in column_nums]

    # Solicitar al usuario que seleccione la columna objetivo
    target_num = int(input("Introduce el número de la columna objetivo: "))
    target_col = df.columns[target_num-1]

    X = df[feature_cols]
    y = df[target_col]

    # Escalamos los datos para que sean adecuados para la red neuronal
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    if tipo == 'clasificacion':
        encoder = LabelEncoder().fit(y)
        y_encoded = encoder.transform(y)
        n_classes = len(encoder.classes_)
    else:
        y_encoded = y

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42)

    # 2. Construir la red neuronal:
    model = tf.keras.Sequential()

    # Capa de entrada
    model.add(tf.keras.layers.Dense(
        64, activation='relu', input_shape=(X_train.shape[1],)))

    # Capa oculta
    model.add(tf.keras.layers.Dense(32, activation='relu'))

    # Capa de salida
    if tipo == 'clasificacion':
        model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
        model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')

    # 3. Entrenar la red:
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_test, y_test))

    # 4. Evaluar el modelo:
    if tipo == 'clasificacion':
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Exactitud en datos de prueba: {accuracy*100:.2f}%")
        print("\nConclusión:")
        print("La exactitud te indica qué porcentaje de las predicciones fueron correctas.")
        print("En este caso, la red ha predicho correctamente el",
              accuracy*100, "% de las veces.")
        print("\nConsejo:")
        print("Si la exactitud es baja, considera recopilar más datos o ajustar la arquitectura de la red.")
    else:
        mse = model.evaluate(X_test, y_test)
        print(f"MSE en datos de prueba: {mse:.2f}")
        print("\nConclusión:")
        print("El MSE (Mean Squared Error) mide cuánto se desvían las predicciones del valor real. Cuanto más bajo, mejor.")
        print("\nConsejo:")
        print("Si el MSE es alto, prueba a ajustar la arquitectura de la red o considera recopilar más datos.")

    return model


# Funciones de guardado


def guardar_grafico(nombre):
    ruta = "visualizaciones"
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    plt.savefig(os.path.join(ruta, nombre + ".png"))

# Main


def main():
    salir = False
    while not salir:
        salir = menu_principal()


        # menu() retorna True si el usuario quiere salir del programa
if __name__ == "__main__":
    main()  # Ejecuta la funcion main
