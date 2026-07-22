import json
import os
import urllib.error
import urllib.request
from typing import Any

from fastapi import HTTPException, Request


SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_PUBLISHABLE_KEY = (
    os.getenv("SUPABASE_PUBLISHABLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or ""
)


def is_supabase_auth_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY)


def supabase_public_config() -> dict[str, str | bool]:
    enabled = is_supabase_auth_enabled()
    return {
        "enabled": enabled,
        "url": SUPABASE_URL if enabled else "",
        "publishable_key": SUPABASE_PUBLISHABLE_KEY if enabled else "",
        "anon_key": SUPABASE_PUBLISHABLE_KEY if enabled else "",
    }


def _bearer_token_from_request(request: Request, access_token: str | None = None) -> str:
    if access_token:
        return access_token.strip()

    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()

    return ""


def _get_supabase_user(token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "apikey": SUPABASE_PUBLISHABLE_KEY,
            "Authorization": f"Bearer {token}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=401, detail="Invalid or expired session.") from error
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail="Could not verify Supabase session.") from error

    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail="Invalid session user.")
    return payload


def current_user_id_from_request(request: Request, access_token: str | None = None) -> str | None:
    if not is_supabase_auth_enabled():
        return None

    token = _bearer_token_from_request(request, access_token)
    if not token:
        raise HTTPException(status_code=401, detail="Sign in to use DeepDoc.")

    user = _get_supabase_user(token)
    user_id = str(user.get("id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid session user.")

    return user_id


def optional_user_id_from_request(request: Request, access_token: str | None = None) -> str | None:
    if not is_supabase_auth_enabled():
        return None

    token = _bearer_token_from_request(request, access_token)
    if not token:
        return None

    return current_user_id_from_request(request, access_token=access_token)
