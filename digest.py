"""School digest: pulls IST / Veracross emails from Gmail, asks Claude to
extract anything that's a deadline, requires action, or requires attendance
for Ari (3rd grade) and Avir (1st grade), and emails the digest to the user.

Designed to run from GitHub Actions on a schedule and on-demand."""

from __future__ import annotations

import email
import email.utils
import imaplib
import json
import os
import smtplib
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import getaddresses, parsedate_to_datetime
from html import escape
from zoneinfo import ZoneInfo

import anthropic

LOCAL_TZ = ZoneInfo("America/Chicago")
SCHOOL_DOMAINS = ("istexas.org", "mail1.veracross.com")
KIDS = [
    {"name": "Ari", "grade": "3rd"},
    {"name": "Avir", "grade": "1st"},
]
LOOKBACK_DAYS = 21
HORIZON_DAYS = 14
MODEL = "claude-opus-4-7"


@dataclass
class Msg:
    subject: str
    sender: str
    date: datetime
    body: str

    def trimmed(self, limit: int = 6000) -> str:
        body = self.body.strip()
        if len(body) > limit:
            body = body[:limit] + "\n…[truncated]"
        return body


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    v = os.environ.get(name, default)
    if required and not v:
        sys.exit(f"Missing required env var: {name}")
    return v or ""


def imap_search_school_messages(user: str, password: str) -> list[Msg]:
    queries = [f'from:@{d} OR "{d}"' for d in SCHOOL_DOMAINS] + [
        'subject:IST OR "International School of Texas"',
    ]

    with imaplib.IMAP4_SSL("imap.gmail.com") as M:
        M.login(user, password)
        M.select('"[Gmail]/All Mail"', readonly=True)

        seen_uids: set[bytes] = set()
        msgs: list[Msg] = []
        for q in queries:
            full_q = f"newer_than:{LOOKBACK_DAYS}d ({q})"
            typ, data = M.uid("SEARCH", None, "X-GM-RAW", full_q)
            if typ != "OK" or not data or not data[0]:
                continue
            for uid in data[0].split():
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)
                typ, fetched = M.uid("FETCH", uid, "(RFC822)")
                if typ != "OK" or not fetched or not fetched[0]:
                    continue
                msgs.append(_parse(fetched[0][1]))

    msgs.sort(key=lambda m: m.date)
    return msgs


def _parse(raw: bytes) -> Msg:
    em = email.message_from_bytes(raw)
    subject = em.get("Subject", "")
    sender = ", ".join(addr for _, addr in getaddresses([em.get("From", "")]))
    date_hdr = em.get("Date", "")
    try:
        dt = parsedate_to_datetime(date_hdr)
    except (TypeError, ValueError):
        dt = datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    body = _extract_body(em)
    return Msg(subject=subject, sender=sender, date=dt.astimezone(LOCAL_TZ), body=body)


def _extract_body(em: email.message.Message) -> str:
    if em.is_multipart():
        text_parts: list[str] = []
        html_parts: list[str] = []
        for part in em.walk():
            ctype = part.get_content_type()
            if part.get_content_disposition() == "attachment":
                continue
            if ctype == "text/plain":
                text_parts.append(_decode(part))
            elif ctype == "text/html":
                html_parts.append(_decode(part))
        if text_parts:
            return "\n".join(text_parts)
        if html_parts:
            return _strip_html("\n".join(html_parts))
        return ""
    payload = _decode(em)
    if em.get_content_type() == "text/html":
        return _strip_html(payload)
    return payload


def _decode(part: email.message.Message) -> str:
    raw = part.get_payload(decode=True) or b""
    charset = part.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _strip_html(s: str) -> str:
    import re

    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_digest(msgs: list[Msg], api_key: str) -> dict:
    today = datetime.now(tz=LOCAL_TZ).date()
    horizon = today + timedelta(days=HORIZON_DAYS)

    if not msgs:
        return {
            "summary": "No school emails found in the lookback window.",
            "items": [],
            "generated_at": datetime.now(tz=LOCAL_TZ).isoformat(),
        }

    rendered = []
    for i, m in enumerate(msgs, 1):
        rendered.append(
            f"--- EMAIL {i} ---\n"
            f"Date: {m.date.isoformat()}\n"
            f"From: {m.sender}\n"
            f"Subject: {m.subject}\n\n"
            f"{m.trimmed()}\n"
        )
    corpus = "\n".join(rendered)

    kids_str = ", ".join(f"{k['name']} ({k['grade']} grade)" for k in KIDS)

    system = (
        "You read parent emails from a private school (International School of Texas / "
        "Veracross) and produce a tight action digest for a parent. Today is "
        f"{today.isoformat()} (America/Chicago). Kids: {kids_str}. "
        f"Only include items where the action, deadline, attendance, dress code, or "
        f"signup falls between today and {horizon.isoformat()}, "
        "OR where there is an unresolved action with no clear date but it's still "
        "relevant. Skip past events and pure FYI. Group by kid where the email "
        "scopes to a grade, otherwise put it under 'Both'. Be concrete: dress code, "
        "what to bring, what to sign, link if present. No fluff."
    )
    schema = {
        "type": "object",
        "properties": {
            "tomorrow": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kid": {"type": "string"},
                        "title": {"type": "string"},
                        "when": {"type": "string"},
                        "action": {"type": "string"},
                        "source_subject": {"type": "string"},
                    },
                    "required": ["kid", "title", "action"],
                },
            },
            "this_week": {"type": "array", "items": {"$ref": "#/properties/tomorrow/items"}},
            "upcoming": {"type": "array", "items": {"$ref": "#/properties/tomorrow/items"}},
            "open_actions": {"type": "array", "items": {"$ref": "#/properties/tomorrow/items"}},
        },
        "required": ["tomorrow", "this_week", "upcoming", "open_actions"],
    }

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=system,
        messages=[
            {
                "role": "user",
                "content": (
                    "Produce the digest as JSON matching this schema:\n"
                    f"{json.dumps(schema)}\n\n"
                    "'tomorrow' = items dated for tomorrow only.\n"
                    "'this_week' = items dated within the next 7 days (excluding tomorrow).\n"
                    "'upcoming' = items dated 8-14 days out.\n"
                    "'open_actions' = signup/sign/RSVP/payment items with no firm date but still open.\n"
                    "Return ONLY the JSON object, no prose, no markdown fence.\n\n"
                    "EMAILS:\n\n" + corpus
                ),
            }
        ],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
    data = json.loads(text)
    data["generated_at"] = datetime.now(tz=LOCAL_TZ).isoformat()
    data["email_count"] = len(msgs)
    return data


def render_html(digest: dict) -> str:
    today = datetime.now(tz=LOCAL_TZ)
    tomorrow = (today + timedelta(days=1)).strftime("%A, %b %-d")

    def section(title: str, items: list, intro: str = "") -> str:
        if not items:
            return ""
        rows = "".join(
            f"<li><strong>{escape(it.get('kid', ''))}:</strong> "
            f"{escape(it.get('title', ''))}"
            + (f" — <em>{escape(it['when'])}</em>" if it.get("when") else "")
            + f"<br><span style='color:#444'>{escape(it.get('action', ''))}</span>"
            + (
                f"<br><span style='color:#888;font-size:12px'>from: "
                f"{escape(it['source_subject'])}</span>"
                if it.get("source_subject")
                else ""
            )
            + "</li>"
            for it in items
        )
        intro_html = f"<p style='color:#555'>{escape(intro)}</p>" if intro else ""
        return (
            f"<h2 style='margin-top:24px'>{escape(title)}</h2>{intro_html}"
            f"<ul style='line-height:1.5'>{rows}</ul>"
        )

    body = (
        section(f"Tomorrow — {tomorrow}", digest.get("tomorrow", []))
        + section("This week", digest.get("this_week", []))
        + section("Upcoming (8-14 days)", digest.get("upcoming", []))
        + section("Open actions (signups / forms / payments)", digest.get("open_actions", []))
    )
    if not body:
        body = "<p>No actionable items found in the source emails.</p>"

    return f"""<!doctype html>
<html><body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:680px;margin:auto">
<h1 style="margin-bottom:0">School digest</h1>
<p style="color:#666;margin-top:4px">Generated {escape(digest.get('generated_at', ''))} · {digest.get('email_count', 0)} emails scanned</p>
{body}
<hr style="margin-top:32px;border:none;border-top:1px solid #eee">
<p style="color:#aaa;font-size:12px">Auto-generated from IST + Veracross emails. Reply with feedback to tune what gets surfaced.</p>
</body></html>"""


def render_text(digest: dict) -> str:
    lines = [
        f"School digest — generated {digest.get('generated_at', '')}",
        f"({digest.get('email_count', 0)} emails scanned)",
        "",
    ]

    def section(title: str, items: list) -> None:
        if not items:
            return
        lines.append(f"== {title} ==")
        for it in items:
            head = f"- [{it.get('kid','')}] {it.get('title','')}"
            if it.get("when"):
                head += f" ({it['when']})"
            lines.append(head)
            if it.get("action"):
                lines.append(f"    {it['action']}")
            if it.get("source_subject"):
                lines.append(f"    from: {it['source_subject']}")
        lines.append("")

    section("Tomorrow", digest.get("tomorrow", []))
    section("This week", digest.get("this_week", []))
    section("Upcoming (8-14 days)", digest.get("upcoming", []))
    section("Open actions", digest.get("open_actions", []))
    if len(lines) <= 3:
        lines.append("No actionable items found.")
    return "\n".join(lines)


def send_email(to_addrs: list[str], subject: str, text: str, html: str, smtp_user: str, smtp_pass: str) -> None:
    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    msg["Date"] = email.utils.formatdate(localtime=True)
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)


def main() -> int:
    gmail_user = env("GMAIL_USER", required=True)
    gmail_pass = env("GMAIL_APP_PASSWORD", required=True)
    anthropic_key = env("ANTHROPIC_API_KEY", required=True)
    to_email = env("DIGEST_TO", default=gmail_user)
    cc_email = env("DIGEST_CC", default="")
    dry_run = env("DRY_RUN", "") == "1"

    print(f"[digest] fetching school emails for {gmail_user}…", flush=True)
    msgs = imap_search_school_messages(gmail_user, gmail_pass)
    print(f"[digest] found {len(msgs)} candidate emails", flush=True)

    digest = build_digest(msgs, anthropic_key)
    html = render_html(digest)
    text = render_text(digest)

    tomorrow = (datetime.now(tz=LOCAL_TZ) + timedelta(days=1)).strftime("%a %b %-d")
    subject = f"School digest — {tomorrow}"

    if dry_run:
        print("[digest] DRY_RUN=1 — not sending. Preview:\n")
        print(text)
        return 0

    recipients = [to_email] + ([cc_email] if cc_email else [])
    send_email(recipients, subject, text, html, gmail_user, gmail_pass)
    print(f"[digest] sent to {recipients}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
