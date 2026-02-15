# auth.py
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from database import get_db
from sql_models import User
# =========================
# 基本設定（請放到 config.py 更好）
# =========================
SECRET_KEY = "CHANGE_ME_TO_A_RANDOM_LONG_STRING"  # ⚠️ 上線前一定要換
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # token 有效期 1 天

# 密碼雜湊工具（bcrypt）
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 取 token 的方式：Authorization: Bearer <token>
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# =========================
# 密碼雜湊/驗證
# =========================
def hash_password(password: str) -> str:
    """將明文密碼轉成 bcrypt 雜湊"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """驗證明文密碼是否匹配雜湊"""
    return pwd_context.verify(plain_password, hashed_password)


# =========================
# JWT 產生/解析
# =========================
def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    產生 JWT
    subject 通常放 user_id（轉字串）
    """
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    解析 JWT，拿到當前登入的使用者（User ORM）
    """
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="驗證失敗，請重新登入",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if not sub:
            raise cred_exc
        user_id = int(sub)
    except (JWTError, ValueError):
        raise cred_exc

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise cred_exc
    return user


oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/login", auto_error=False)

def get_current_user_optional(
    token: str = Depends(oauth2_scheme_optional),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    ✅ 可選登入：
    - 沒帶 token → 回 None（Guest）
    - token 無效 → 回 None（視為 Guest）
    - token 有效 → 回 User
    """
    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if not sub:
            return None
        user_id = int(sub)
    except (JWTError, ValueError):
        return None

    return db.query(User).filter(User.id == user_id).first()