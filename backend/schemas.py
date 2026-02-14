# schemas.py
from pydantic import BaseModel, Field, field_validator

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=128)

    @field_validator("password")
    @classmethod
    def password_max_72_bytes(cls, v: str):
        # bcrypt 限制：最多 72 bytes（utf-8）
        if len(v.encode("utf-8")) > 72:
            raise ValueError("password 最多 72 bytes")
        return v

class LoginRequest(BaseModel):
    username: str
    password: str

    @field_validator("password")
    @classmethod
    def password_max_72_bytes(cls, v: str):
        if len(v.encode("utf-8")) > 72:
            raise ValueError("password 最多 72 bytes")
        return v

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserMeResponse(BaseModel):
    id: int
    username: str
