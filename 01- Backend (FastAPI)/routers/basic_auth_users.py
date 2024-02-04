# Aqui vamos a montar la autorizacion

# Vamos a montarla en principio como nueva API

# uvicorn basic_auth_users:app --reload

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
# Importamos el autenticador y la forma en la que enviamos user y pass
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm


router = APIRouter(tags=["Basic Auth"])

# Vamos a crear una instancia de nuestro sistema de autenticacion, usaremos el standard oauth2

oauth2 = OAuth2PasswordBearer(tokenUrl="login")


class User(BaseModel):  # Creamos la clase usuario

    username: str
    full_name: str
    email: str
    disabled: bool

# Como vamos a ver todo sobre base de datos,
# Creamos la clase UserDB heredando olos datos de User y añadiendo password


class UserDB(User):
    password: str

# Vamos a crear un par de usuarios


users_db = {
    "jarko": {
        "username": "jarko",
        "full_name": "Jarko Kurst",
        "email": "jarkoarenal@gmail.com",
        "disabled": False,
        "password": "123456"
    },
    "guest": {
        "username": "guest",
        "full_name": "Invitado",
        "email": "No tiene",
        "disabled": True,
        "password": "000000"
    }

}


def search_user_db(username: str):
    if username in users_db:
        return UserDB(**users_db[username])  # Los 2 asteriscos sirven para


def search_user(username: str):
    if username in users_db:
        return User(**users_db[username])  # Los 2 asteriscos sirven para

# Vamos a crear un criterio de dependencia

# Con el depends aseguramos que esté validado


async def current_user(token: str = Depends(oauth2)):
    # Sabemos que el token es el usuario, asi que ponemos token y listo
    user = search_user(token)
    if not user:
        # Con status buscamos errores mas facil, y con header, por standard indicamos el tipo de autorización.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales de autenticación inválidas", headers={"WWW-Authenticate": "Bearer"})
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Usuario Baneado", headers={"WWW-Authenticate": "Bearer"})
    return user


# Funcion para mandar user y pass
@router.post("/login")
# le decimos que la funcion login usara un form que viene de la libreria oauth...
# El depends significa que va a recibir datos pero no depende de nadie
# Cuando algo depende de algo, no podremos acceder sin cumplir esa dependencia
async def login(form: OAuth2PasswordRequestForm = Depends()):

    # Vamos a buscar en la base de datos a ver si el user existe
    user_db = users_db.get(form.username)

    if not user_db:
        raise HTTPException(
            status_code=400, detail="El usuario no es correcto")

    user = search_user_db(form.username)
    if not form.password == user.password:

        raise HTTPException(
            status_code=400, detail="La contraseña no es correcta")

    # Vamos a devolver el token de acceso y el tipo de token, y el tipo será "bearer"
    return {"access_token": user.username, "token_type": "bearer"}


# Funcion para decirme cual es mi usuario
@router.get("/users/me")
# devolvemos el usuario que no tiene la contraseña (Recordemos que el DB es quien la tiene, asi evitamos la brecha de seguridad)
# Vamos a aplicar el depends al criterio de dependencia que hemos creado
async def me(user: User = Depends(current_user)):
    return user
