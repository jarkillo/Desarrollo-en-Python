# Vamos a crear una API de usuarios

# httpexception es para los codigos de errores
from fastapi import APIRouter, HTTPException

from pydantic import BaseModel  # Queremos usar entidades


# lanzar servidor: uvicorn users:app --reload

router = APIRouter(tags=["User"])

# Entidad user (con BaseModel podemos crear entidades, es como una clase de tipo usuario con los siguientes datos
# pero no tenemos que hacer constructor de clases)


class User(BaseModel):
    id: int
    name: str
    surname: str
    url: str
    age: int

# Vamos a crear usuarios ficticios


users_list = [User(id=1, name="Jarko", surname="Kurst", url="http://manuellopez.online", age=36),
              User(id=2, name="Paco", surname="Pepe",
                   url="http://pacopepe.online", age=36),
              User(id=3, name="Luis", surname="Agudo", url="http://luisagudo.online", age=36)]


# A la antigua tendriamos que ir añadiendo a esta funcion los usuarios a mano, pero con las entidades podemos evitarnoslo


@router.get("/usersjson")
async def usersjson():
    return [{"name": "Jarko", "surname": "Kurst", "url": "http://manuellopez.online", "age": 36},
            {"name": "Paco", "surname": "Pepe",
                "url": "http://pacopepe.online", "age": 25},
            {"name": "Luis", "surname": "Agudo", "url": "http://luisagudo.online", "age": 45}]

# Pero ahora podemos lanzarla directamente


@router.get("/users")
async def users():

    return users_list

# Podemos hacerlo por PATH -> http://web.com/user/id


@router.get("/user/{id}")  # id es un parametro
async def user(id: int):  # tenemos que tipar el tipo de dato
    return search_user(id)

# O podemos llamarlo por Query
# Generalmente las querys podemos usarlas mejor para filtros, en caso de la id es mejor meterla
# por path, ya que si no añadimos nada al path daría error y pediría la id


@router.get("/user/")
async def user(id: int):  # tenemos que tipar el tipo de dato
    return search_user(id)


# Vamos a hacer una funcion para añadir usuarios


# con status code = parametro es para indicar que devuelva un codigo por defecto
@router.post("/user/", status_code=201)
async def user(user: User):
    if type(search_user(user.id)) == User:
        # Aqui le forzamos el 204 si esto se cumple y el mensaje de error solo saldrá en caso de ser un error por ejemplo 404
        raise HTTPException(status_code=204, detail="El usuario ya existe")
    else:
        users_list.append(user)
        return {"Correcto": "El usuario ha sido añadido"}, user

# Vamos a hacer una funcion para modificar usuarios


@router.put("/user/")
async def user(user: User):

    found = False

    # Recorremos toda la lista buscando la id
    for index, saved_user in enumerate(users_list):

        if saved_user.id == user.id:
            users_list[index] = user
            found = True
            return {"Correcto": "El usuario ha sido modificado"}

    if not found:
        return {"Error": "El usuario no existe"}
    else:
        return user


# Ahora vamos a hacer la funcion para borrar usuarios

@router.delete("/user/{id}")  # id es un parametro
async def user(id: int):  # tenemos que tipar el tipo de dato

    found = False

    for index, saved_user in enumerate(users_list):

        if saved_user.id == id:
            del users_list[index]
            found = True

            return {"Correcto": "El usuario ha sido eliminado"}

    if not found:
        return {"Error": "El usuario no existe"}

# Podemos llevar la busqueda a una funcion search_user


def search_user(id: int):

    users = filter(lambda user: user.id == id, users_list)
    try:

        # Devolvemos la lista correspondiente al resultado 0
        return list(users)[0]

    except:

        return {"Error": "No se ha encontrado el usuario"}
