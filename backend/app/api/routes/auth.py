"""Authentication routes for password protection."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])

# OAuth2 scheme for token retrieval
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# JWT Configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7  # Token valid for 7 days


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    """Login request model."""
    password: str


def create_access_token() -> str:
    """Create a new JWT access token."""
    expires_delta = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    expire = datetime.utcnow() + expires_delta
    to_encode = {"exp": expire, "iat": datetime.utcnow(), "type": "access"}
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: Optional[str]) -> bool:
    """Verify a JWT token."""
    if not token:
        return False
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        exp = payload.get("exp")
        if exp is None:
            return False
        if datetime.utcnow() > datetime.fromtimestamp(exp):
            return False
        return True
    except JWTError:
        return False


def is_auth_enabled() -> bool:
    """Check if authentication is enabled (password is configured)."""
    return bool(settings.login_password)


async def get_current_token(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[str]:
    """Dependency to get the current token from the request."""
    return token


async def require_auth(token: Optional[str] = Depends(oauth2_scheme)) -> str:
    """Dependency that requires a valid token.
    
    Raises HTTPException if auth is enabled and token is invalid.
    """
    # If auth is not enabled, allow all requests
    if not is_auth_enabled():
        return ""
    
    # Verify token
    if not token or not verify_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with password to get an access token.
    
    Uses OAuth2 password flow for compatibility, but ignores username.
    """
    # Check if auth is configured
    if not is_auth_enabled():
        # Auth not configured, return a dummy token
        return Token(access_token=create_access_token())
    
    # Verify password
    if form_data.password != settings.login_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create and return token
    access_token = create_access_token()
    return Token(access_token=access_token)


@router.post("/login/json", response_model=Token)
async def login_json(data: LoginRequest):
    """Login with password (JSON endpoint for frontend).
    
    Alternative to the OAuth2 form-based login for easier frontend integration.
    """
    # Check if auth is configured
    if not is_auth_enabled():
        # Auth not configured, return a dummy token
        return Token(access_token=create_access_token())
    
    # Verify password
    if data.password != settings.login_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create and return token
    access_token = create_access_token()
    return Token(access_token=access_token)


@router.get("/verify")
async def verify_auth_token(token: str = Depends(require_auth)):
    """Verify if the current token is valid.
    
    Returns 200 if token is valid, 401 otherwise.
    """
    return {"status": "valid", "authenticated": True}


@router.get("/status")
async def auth_status():
    """Check if authentication is enabled on the server.
    
    This is a public endpoint that doesn't require authentication.
    """
    return {"enabled": is_auth_enabled()}
