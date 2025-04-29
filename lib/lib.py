import pandas as pd
import mysql.connector
import numpy as np
from datetime import datetime
import streamlit as st

def consulta(consulta):
    # Parámetros de conexión
    host = st.secrets["database"]["host"]
    user = st.secrets["database"]["user"]
    password = st.secrets["database"]["password"]
    database = st.secrets["database"]["database"]

    # Objeto de conexión
    conexion = None

    try:
        # Abrir la conexión
        conexion = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        # Crear un cursor para ejecutar la consulta
        cursor = conexion.cursor()

        # Ejecutar la consulta
        cursor.execute(consulta)

        # Obtener los resultados de la consulta
        resultados = cursor.fetchall()

        # Obtener los nombres de las columnas
        columnas = [desc[0] for desc in cursor.description]

        # Crear un DataFrame con los resultados y las columnas correspondientes
        df = pd.DataFrame(resultados, columns=columnas)

        return df

    except mysql.connector.Error as error:
        print("Error al ejecutar la consulta:", error)

    finally:
        # Cerrar la conexión
        if conexion is not None and conexion.is_connected():
            cursor.close()
            conexion.close()