from fastapi import APIRouter


# lanzar servidor: uvicorn products:app --reload
# Estar lanzando un servidor cada vez que cambiamos la API es estupido
# Asi que ahora vamos a crear un router para hacer esto en la carpeta routers
# Importamos APIRouter
# y creamos la variable router con APIrouter()
# y en vez de llamar a app, llamamos a router
# Ahora vamos a configurarla en el archivo main.py de la carpeta padre

# COn prefix indicamos que esto se va a hacer en products, y ya no hay que indicarlo, lo que sea /, sera products
# con responses podemos indicar errores a los fallos
# Con el tag agrupamos esto en la documentacion

router = APIRouter(prefix="/products",
                   tags=["Products"], responses={404: {"Message": "No encontrado"}})


products_list = ["Producto 1", "Producto 2",
                 "Producto 3", "Producto 4", "Producto 5"]


@router.get("/")
async def products():

    return products_list


@router.get("/{id}")
async def products(id: int):

    return products_list[id]
