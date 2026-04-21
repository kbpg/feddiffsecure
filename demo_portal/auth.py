from __future__ import annotations

from functools import wraps

from flask import abort, redirect, request, session, url_for


def is_authenticated() -> bool:
    return bool(session.get("authenticated"))


def current_user() -> str:
    return str(session.get("username", ""))


def current_role() -> str:
    return str(session.get("role", "guest"))


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not is_authenticated():
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)

    return wrapped


def role_required(*allowed_roles: str):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not is_authenticated():
                return redirect(url_for("login", next=request.path))
            if current_role() not in allowed_roles:
                return abort(403)
            return view_func(*args, **kwargs)

        return wrapped

    return decorator
