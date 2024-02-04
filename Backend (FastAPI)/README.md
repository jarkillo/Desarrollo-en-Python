# Backend con FastAPI para Extracción de Datos de MongoDB

## Descripción
Este proyecto consiste en un backend desarrollado con FastAPI para extraer datos de un dataset en MongoDB. Se utiliza para gestionar productos, transacciones y usuarios, y es un proyecto que mantengo ya que me es util para conectar mis programas con la base de datos.

## Estructura del Repositorio
### Backend
- **db**
  - **env**: Directorio para el archivo `tokens.env` (no incluido en el repositorio) con los datos de acceso a la base de datos.
  - **models**
    - `products.py`: Modelo de cada producto.
    - `transactions.py`: Modelo de las transacciones.
    - `user.py`: Modelo de usuarios.
  - **schemas**
    - `products.py`: Transforma los productos de la base de datos a JSON.
    - `transactions.py`: Transforma las transacciones de la base de datos a JSON.
    - `user.py`: Transforma el usuario de la base de datos a JSON.
  - `client.py`: Código de conexión con la base de datos, utilizando los datos de `tokens.env`.
  - **routers**
    - `basic_auth_users.py`: Gestión de autorización básica.
    - `jwt_auth_users.py`: Control de encriptación con JWT.
    - `products.py`: Rutas para productos.
    - `users_db.py`: Consultas a la base de datos de usuarios.
    - `users.py`: API para entidades de usuarios.
- **Static**
  - **images**: Carpeta de imágenes.
- `main.py`: Archivo principal de la API.
- `requirements.txt`: Dependencias necesarias para el proyecto.

## Configuración
Para configurar el entorno de la base de datos, cree un archivo `tokens.env` en la carpeta `db/env` con la siguiente estructura:

MONGO_USER=nombre_de_usuario
MONGO_PASSWORD=contraseña
MONGO_HOST=dirección_del_servidor
MONGO_DB_NAME=nombre_de_la_DB

**Nota**: Este archivo no debe subirse al repositorio por razones de seguridad.

## Instalación
Para instalar las dependencias del proyecto, ejecute:

pip install -r requirements.txt

---------------

# FastAPI Backend for MongoDB Data Extraction

## Description
This project is a backend developed with FastAPI for extracting data from a MongoDB dataset. It is used to manage products, transactions, and users.

## Repository Structure
### Backend
- **db**
  - **env**: Directory for the `tokens.env` file (not included in the repository) with database access details.
  - **models**
    - `products.py`: Model for each product.
    - `transactions.py`: Model for transactions.
    - `user.py`: Model for users.
  - **schemas**
    - `products.py`: Transforms products from the database to JSON.
    - `transactions.py`: Transforms transactions from the database to JSON.
    - `user.py`: Transforms users from the database to JSON.
  - `client.py`: Database connection code, using data from `tokens.env`.
  - **routers**
    - `basic_auth_users.py`: Basic authorization management.
    - `jwt_auth_users.py`: JWT encryption control.
    - `products.py`: Routes for products.
    - `users_db.py`: Queries to the user database.
    - `users.py`: API for user entities.
- **Static**
  - **images**: Image folder.
- `main.py`: Main file of the API.
- `requirements.txt`: Required dependencies for the project.

## Configuration
To set up the database environment, create a `tokens.env` file in the `db/env` folder with the following structure:

MONGO_USER=username
MONGO_PASSWORD=password
MONGO_HOST=server_address
MONGO_DB_NAME=DB_name

**Note**: This file should not be uploaded to the repository for security reasons.

## Installation
To install the project dependencies, run:

pip install -r requirements.txt

