#!/usr/bin/env python3
"""Account management CLI — run it on the server (AWS/EC2, Docker, local).

    python -m recapstudio.useradmin status
    python -m recapstudio.useradmin create kyaw --admin            # asks for a password
    python -m recapstudio.useradmin create editor1 --random        # prints a strong one
    python -m recapstudio.useradmin list
    python -m recapstudio.useradmin passwd kyaw
    python -m recapstudio.useradmin disable editor1
    python -m recapstudio.useradmin promote editor1
    python -m recapstudio.useradmin sessions kyaw --revoke
    python -m recapstudio.useradmin audit --limit 30

The first account you create switches the deployment from "shared access
key" to full multi-user mode automatically (``RECAP_AUTH_MODE=auto``), so no
restart and no code change is needed — the next browser reload asks for a
login.
"""
from __future__ import annotations

import argparse
import getpass
import json
import secrets
import string
import sys
import time
from typing import Optional

from . import auth, config, db
from .crypto import encryption_backend, secret_is_persistent

OK = "\033[92m"
WARN = "\033[93m"
ERR = "\033[91m"
DIM = "\033[2m"
OFF = "\033[0m"


def _print(msg: str = "") -> None:
    print(msg)


def _ask_password(username: str, confirm: bool = True) -> str:
    while True:
        password = getpass.getpass(f"'{username}' အတွက် စကားဝှက် (password): ")
        error = auth.check_password_policy(password, username)
        if error:
            _print(f"{ERR}✖ {error}{OFF}")
            continue
        if confirm:
            again = getpass.getpass("ထပ်မံ အတည်ပြုရန် (confirm): ")
            if again != password:
                _print(f"{ERR}✖ စကားဝှက် မတူညီပါ — ပြန်ကြိုးစားပါ{OFF}")
                continue
        return password


def random_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_=+"
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(length))
        if auth.check_password_policy(candidate, "") is None:
            return candidate


def _resolve_user(username: str) -> dict:
    user = auth.get_user_by_name(username)
    if not user:
        _print(f"{ERR}✖ '{username}' အကောင့် ရှာမတွေ့ပါ{OFF}")
        sys.exit(2)
    return user


def _fmt_time(value: float) -> str:
    if not value:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(value))


# ── commands ─────────────────────────────────────────────────────────────
def cmd_status(args: argparse.Namespace) -> int:
    db.init()
    report = auth.security_report()
    _print(f"{OK}▌ Recap Studio — security status{OFF}")
    _print(f"  data dir        : {config.DATA_DIR}")
    _print(f"  database        : {config.DB_PATH}  (schema v{db.SCHEMA_VERSION}, "
           f"{report['db'].get('mode', '?')})")
    _print(f"  auth mode       : {report['auth_mode']}")
    _print(f"  accounts        : {report['users']}")
    _print(f"  signup enabled  : {report['signup_enabled']}")
    _print(f"  session TTL     : {report['session_ttl_hours']}h")
    _print(f"  secret storage  : {encryption_backend()} "
           f"({'persistent' if secret_is_persistent() else 'EPHEMERAL — set RECAP_SECRET_KEY'})")
    _print(f"  rate limit      : {report['rate_limit_per_minute']}/min")
    if report["auth_mode"] == "open":
        _print(f"{WARN}  ⚠ မည်သူမဆို ဝင်နိုင်နေပါသည် — အကောင့် တစ်ခု ဖန်တီးပါ:{OFF}")
        _print("      python -m recapstudio.useradmin create <name> --admin")
    elif report["auth_mode"] == "legacy":
        _print(f"{WARN}  ⚠ shared access key mode — multi-user သို့ ပြောင်းရန် အကောင့် ဖန်တီးပါ{OFF}")
    return 0


def cmd_create(args: argparse.Namespace) -> int:
    db.init()
    error = auth.check_username(args.username)
    if error:
        _print(f"{ERR}✖ {error}{OFF}")
        return 2
    password = args.password or ""
    generated = False
    if args.random:
        password = random_password()
        generated = True
    elif not password:
        password = _ask_password(args.username)
    try:
        user = auth.create_user(args.username, password, role="admin" if args.admin else "user",
                                display_name=args.display_name or args.username,
                                must_change_password=bool(args.must_change),
                                quota_bytes=args.quota_bytes or 0,
                                max_concurrent_jobs=args.max_jobs or 0,
                                daily_job_limit=args.daily_jobs or 0)
    except ValueError as exc:
        _print(f"{ERR}✖ {exc}{OFF}")
        return 2
    auth.audit("user_create", target=user["username"], detail="cli")
    _print(f"{OK}✔ အကောင့် ဖန်တီးပြီးပါပြီ: {user['username']} ({user['role']}){OFF}")
    if generated:
        _print(f"  စကားဝှက် (တစ်ခါသာ ပြပါမည်): {OK}{password}{OFF}")
    if auth.user_count() == 1:
        _print(f"{WARN}  ℹ ဤသည် ပထမဆုံး အကောင့် ဖြစ်၍ ယခုမှစ၍ login မဖြစ်ဘဲ "
               f"စာမျက်နှာကို သုံး၍ မရတော့ပါ။{OFF}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    db.init()
    users = auth.list_users()
    if not users:
        _print(f"{WARN}အကောင့် မရှိသေးပါ{OFF}")
        return 0
    if args.json:
        _print(json.dumps(users, ensure_ascii=False, indent=2))
        return 0
    _print(f"{'USERNAME':<20}{'ROLE':<8}{'STATUS':<10}{'LAST LOGIN':<18}{'JOBS/24h':>9}")
    _print("-" * 66)
    for user in users:
        _print(f"{user['username']:<20}{user['role']:<8}{user['status']:<10}"
               f"{_fmt_time(user.get('last_login_at', 0)):<18}"
               f"{auth.jobs_today(user['id']):>9}")
    return 0


def cmd_passwd(args: argparse.Namespace) -> int:
    db.init()
    user = _resolve_user(args.username)
    password = args.password or (random_password() if args.random else
                                 _ask_password(args.username))
    try:
        auth.set_password(user["id"], password, must_change=bool(args.must_change))
    except ValueError as exc:
        _print(f"{ERR}✖ {exc}{OFF}")
        return 2
    auth.audit("password_reset", target=user["username"], detail="cli")
    _print(f"{OK}✔ '{user['username']}' ၏ စကားဝှက် ပြောင်းပြီးပါပြီ "
           f"(session အားလုံး ထွက်သွားပါပြီ){OFF}")
    if args.random:
        _print(f"  စကားဝှက် အသစ်: {OK}{password}{OFF}")
    return 0


def cmd_status_change(args: argparse.Namespace, status: str) -> int:
    db.init()
    user = _resolve_user(args.username)
    try:
        auth.set_status(user["id"], status)
    except ValueError as exc:
        _print(f"{ERR}✖ {exc}{OFF}")
        return 2
    auth.audit(f"user_{status}", target=user["username"], detail="cli")
    _print(f"{OK}✔ '{user['username']}' → {status}{OFF}")
    return 0


def cmd_role(args: argparse.Namespace, role: str) -> int:
    db.init()
    user = _resolve_user(args.username)
    try:
        auth.set_role(user["id"], role)
    except ValueError as exc:
        _print(f"{ERR}✖ {exc}{OFF}")
        return 2
    auth.audit("user_role", target=user["username"], detail=role)
    _print(f"{OK}✔ '{user['username']}' → {role}{OFF}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    db.init()
    user = _resolve_user(args.username)
    if not args.yes:
        answer = input(f"'{user['username']}' ကို အပြီးဖျက်မည် — သေချာပါသလား? (yes/no): ")
        if answer.strip().lower() not in {"y", "yes"}:
            _print("ဖျက်ခြင်း ပယ်ဖျက်လိုက်ပါသည်")
            return 1
    try:
        auth.delete_user(user["id"])
    except ValueError as exc:
        _print(f"{ERR}✖ {exc}{OFF}")
        return 2
    auth.audit("user_delete", target=user["username"], detail="cli")
    _print(f"{OK}✔ '{user['username']}' ဖျက်ပြီးပါပြီ{OFF}")
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    db.init()
    user = _resolve_user(args.username)
    if args.revoke:
        count = auth.revoke_user_sessions(user["id"])
        _print(f"{OK}✔ session {count} ခု ရုပ်သိမ်းပြီးပါပြီ{OFF}")
        return 0
    sessions = auth.list_sessions(user["id"])
    if not sessions:
        _print("session မရှိပါ")
        return 0
    for item in sessions:
        _print(f"  {_fmt_time(item['last_seen_at'])}  {item['ip']:<16} "
               f"{(item['user_agent'] or '')[:60]}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    db.init()
    for row in auth.recent_audit(args.limit):
        flag = "" if row["ok"] else f"{ERR}✖{OFF}"
        _print(f"  {_fmt_time(row['created_at'])}  {row['action']:<18} "
               f"{(row['username'] or '-'):<14} {row['target'][:28]:<28} "
               f"{row['detail'][:40]} {flag}")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    version = db.init(force=True)
    _print(f"{OK}✔ database ready: {config.DB_PATH} (schema v{version}){OFF}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m recapstudio.useradmin",
                                     description="Recap Studio account management")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="auth mode + security summary").set_defaults(func=cmd_status)
    sub.add_parser("migrate", help="create/upgrade the database").set_defaults(func=cmd_migrate)

    create = sub.add_parser("create", help="create an account")
    create.add_argument("username")
    create.add_argument("--admin", action="store_true", help="give admin rights")
    create.add_argument("--password", default="", help="(avoid: visible in shell history)")
    create.add_argument("--random", action="store_true", help="generate a strong password")
    create.add_argument("--display-name", default="")
    create.add_argument("--must-change", action="store_true",
                        help="force a password change on first login")
    create.add_argument("--quota-bytes", type=int, default=0)
    create.add_argument("--max-jobs", type=int, default=0, help="concurrent jobs")
    create.add_argument("--daily-jobs", type=int, default=0)
    create.set_defaults(func=cmd_create)

    listing = sub.add_parser("list", help="list accounts")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_list)

    passwd = sub.add_parser("passwd", help="reset a password")
    passwd.add_argument("username")
    passwd.add_argument("--password", default="")
    passwd.add_argument("--random", action="store_true")
    passwd.add_argument("--must-change", action="store_true")
    passwd.set_defaults(func=cmd_passwd)

    disable = sub.add_parser("disable", help="block sign-in")
    disable.add_argument("username")
    disable.set_defaults(func=lambda a: cmd_status_change(a, "disabled"))

    enable = sub.add_parser("enable", help="unblock sign-in")
    enable.add_argument("username")
    enable.set_defaults(func=lambda a: cmd_status_change(a, "active"))

    promote = sub.add_parser("promote", help="make an admin")
    promote.add_argument("username")
    promote.set_defaults(func=lambda a: cmd_role(a, "admin"))

    demote = sub.add_parser("demote", help="make a normal user")
    demote.add_argument("username")
    demote.set_defaults(func=lambda a: cmd_role(a, "user"))

    delete = sub.add_parser("delete", help="remove an account")
    delete.add_argument("username")
    delete.add_argument("--yes", action="store_true")
    delete.set_defaults(func=cmd_delete)

    sessions = sub.add_parser("sessions", help="list / revoke sessions")
    sessions.add_argument("username")
    sessions.add_argument("--revoke", action="store_true")
    sessions.set_defaults(func=cmd_sessions)

    audit = sub.add_parser("audit", help="recent security events")
    audit.add_argument("--limit", type=int, default=40)
    audit.set_defaults(func=cmd_audit)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config.ensure_dirs()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
