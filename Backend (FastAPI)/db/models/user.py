# Aqui definimos modelos

### User model ###

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class User(BaseModel):
    # ID unico para cada usuario, creado por MongoDB
    user_id: Optional[str] = Field(default=None)
    user_name: str  # Nombre de usuario
    user_email: str  # Email del usuario
