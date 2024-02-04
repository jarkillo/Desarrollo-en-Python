# Instala FastAPI: pip install "fastapi[all]"

from fastapi import FastAPI

# Con esto ya importamos los routers
from routers import products, jwt_auth_users, basic_auth_users, users_db


# Esto es para poder mostrar recursos estaticos
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Routers - Aqui declaramos los routers (users y products van a trabajar para la api Mai)

# ROUTERS DB
app.include_router(products.router)
# app.include_router(users.router)
app.include_router(users_db.router)

# Routers AUTH
app.include_router(jwt_auth_users.router)
app.include_router(basic_auth_users.router)

# Esa la usamos para incluir los ficheros estaticos. La ruta, StaticFiles(), directorio, nombre
app.mount("/static", StaticFiles(directory="static"), name="static")

# El standard para hablar en la red es el protocolo http

# lanzar servidor: uvicorn main:app --reload

# Tenemos dos documentaciones creadas automaticamente para usar la que mas nos guste.

# Para ver la documentacion: http://127.0.0.1:8000/docs
# Para ver la otra documentacion: http://127.0.0.1:8000/redoc

# Esto es nuevo de FastAPI  (haremos un get a algo de una barra, iremos viendo)
# Esto llama a la raiz (localhost en nuestro caso), solo podemos tener una peticion get a la raiz


@app.get("/")  # El get es lo que interpreta el navegador por defecto
async def root():  # Siempre que llamamos a un servidor, la funcion debe ser asincrona, asi que la declaramos así.
    # Si buscamos algo sincrono, no podriamos hacer nada hasta que el servidor devuelva una respuesta

    return ('Hello FastAPI')  # Esto es la respuesta de nuestro servidor


# Vamos a crear otro

@app.get("/url")
async def url():  # Algo que se entiende bien en API es un Json
    return {"url_mia": "https://manuellopez.online"}


# Funciones comunes:

# POST - Crear datos
# GET - Leeer datos
# PUT - Actualizar datos
# DELETE - Borrar datos
