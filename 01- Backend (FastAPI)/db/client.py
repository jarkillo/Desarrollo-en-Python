# Este fichero será el encargado de gestionar la conexión a la base de datos

# Hemos instalado pymongo
# Hemos instalado decouple para coger el env

from pymongo import MongoClient
from decouple import config
import dns.resolver
import os

# Base de datos local
# db_client = MongoClient().local

# Resolver DNS

dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']


# Ruta al archivo env

ENV_PATH = os.path.join(os.path.dirname(__file__), 'env')

# Base de datos remota
# Para conectar con otra DB, añadir detalles en la carpeta env


# Base de datos remota a tienda

# Extrayendo las variables del .env
MONGO_USER = config("MONGO_USER", dotenv_path=ENV_PATH)
MONGO_PASSWORD = config("MONGO_PASSWORD", dotenv_path=ENV_PATH)
MONGO_HOST = config("MONGO_HOST", dotenv_path=ENV_PATH)
MONGO_DB_NAME = config("MONGO_DB_NAME", dotenv_path=ENV_PATH)

# Creando la cadena de conexión
CONNECTION_STRING = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}/?retryWrites=true&w=majority&tls=true"
db_client = MongoClient(CONNECTION_STRING)[MONGO_DB_NAME]

# El uso de os.path.join y os.path.dirname(__file__) ayuda a construir la ruta al archivo .env de manera dinámica,
# independientemente del lugar donde se ejecute el script. Esto debería resolver el problema del archivo .env en un subdirectorio.
