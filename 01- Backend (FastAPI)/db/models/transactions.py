# Transaction Model

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Transactions(BaseModel):
    # ID único para cada transacción, creado por MongoDB
    transaction_id: Optional[str] = Field(default=None)
    user_id: int  # Obtiene la id del cliente que hizo la compra
    date: datetime  # Fecha de la transaccion
    product_id: int  # ID del producto comprado, se captura desde Productos
    amount: float  # Cantidad gastada en la compra
