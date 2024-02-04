# Este fichero es para recoger lo que viene de base de datos y lo transforme a nuestro JSON (Diccionario)

def user_schema(user) -> dict:  # Recibe user y Va a devolver un diccionario
    return {"user_id": str(user["_id"]),
            "user_name": user["username"],
            "user_email": user["email"]}


def users_schema(users) -> list:
    return [user_schema(user) for user in users]
