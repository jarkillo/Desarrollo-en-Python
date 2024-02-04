# Product Model

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Products(BaseModel):
    # ID único para cada producto, creado por MongoDB
    product_id: Optional[str] = Field(default=None)
    product_name: str  # Nombre del producto
    product_category: str  # Categoría en la que se encuentra el producto
