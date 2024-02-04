# Este fichero es para controlar la encriptacion con JWT del token
# Ya que en el otro estabamos devolviendo el username

# uvicorn jwt_auth_users:app --reload

# He copiado otro archivo y a partir de ahi modificamos

# Tenemos que instalar el token criptofrafico

# pip install "python-jose[cryptography]"

# Tambien hay que instalar passlib[bcrypt] -> para el algoritmo de encriptacion


from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
# Importamos el autenticador y la forma en la que enviamos user y pass
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Importamos los dos nuevos modulos
from jose import jwt, JWTError
from passlib.context import CryptContext

# Importamos fechas para los tiempos de acceso del token y timedelta para calculos de fecha

from datetime import datetime, timedelta

# Declaramos el algoritmo de encriptacion, podemos usar el que creamos

ALGORITHM = "HS256"
ACCESS_TOKEN_DURATION = 1  # Declaramos el tiempo que durará el token de acceso
# Aqui añadimos una semilla que servirá para encriptar de forma mas segura, ya que solo el backend la conoce
# (Podemos generarla con openssl rand -hex 32)

SECRET = "22e91da5e32d922dc6e89ae4936d310ec69596632a40881563b920cd023bd1e2"

router = APIRouter(tags=["Pro Auth"])

oauth2 = OAuth2PasswordBearer(tokenUrl="login")

# Contexto de encriptacion
crypt = CryptContext(schemes=["bcrypt"])  # Aqui algoritmo de criptografia


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
        "password": "$2a$12$.cohqE5Q.ASJ7nTa/miSFezXSKFBJhXjWuZC.qjITTDFY/SRNuIsa"
    },
    "guest": {
        "username": "guest",
        "full_name": "Invitado",
        "email": "No tiene",
        "disabled": True,
        "password": "$2a$12$SZhnDSMRHkvMB.ZIloaF2.KlHZc4L84LsBXKmmwScUhEwa4vNE3uO"
    }

}

# Funciones de busqueda


def search_user_db(username: str):
    if username in users_db:
        return UserDB(**users_db[username])  # Los 2 asteriscos sirven para


def search_user(username: str):
    if username in users_db:
        return User(**users_db[username])  # Los 2 asteriscos sirven para


async def auth_user(token: str = Depends(oauth2)):

    exception401 = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciales de autenticación inválidas",
        headers={"WWW-Authenticate": "Bearer"})

    try:
        username = jwt.decode(token, SECRET, algorithms=[ALGORITHM]).get("sub")
        # En caso de que esto funcione, comprobemos que hay nombre de usuario
        if username is None:
            raise exception401

    except JWTError:

        # Si falla JWT damos la excepcion
        raise exception401

    # Si no ha fallado
    return search_user(username)

# FUncion de validacion


async def current_user(user: User = Depends(auth_user)):

    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Usuario Baneado",
            headers={"WWW-Authenticate": "Bearer"})
    return user


# Funcion de login (tuto en el otro archivo) la modificamos para que mire la encriptacion

@router.post("/login")
async def login(form: OAuth2PasswordRequestForm = Depends()):

    user = search_user_db(form.username)

    if not crypt.verify(form.password, user.password):
        raise HTTPException(
            status_code=400, detail="El usuario no es correcto")

    # Vamos a crear el objeto del access token que será un JSON encriptado, con el usuario, el tiempo de acceso

    access_token = {"sub": user.username,
                    "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_DURATION)}

    # Vamos a devolver el token de acceso y el tipo de token, y el tipo será "bearer"
    # En el codigo original (basic), devolviamos el usuario, ahora vamos a devolver un token de verdad, con tiempo de acceso

    return {"access_token": jwt.encode(access_token, SECRET, algorithm=ALGORITHM), "token_type": "bearer"}


# Funcion para decirme cual es mi usuario
@router.get("/users/me")
async def me(user: User = Depends(current_user)):
    return user
