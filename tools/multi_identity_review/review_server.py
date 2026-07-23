#!/usr/bin/env python3
"""Localhost-only, append-only human review server."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

try:
    from .workflow import append_jsonl_event, resolve_initial_decision, validate_review_event
except ImportError:
    from workflow import append_jsonl_event, resolve_initial_decision, validate_review_event


MAX_BODY_BYTES = 1024 * 1024
STATIC_FILES = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}
CANDIDATE_QUEUE_FIELDS = {
    "candidate_id",
    "identity_id",
    "garment_id",
    "semantic_pose_slot",
    "candidate_index",
    "candidate_sha256",
    "image_path",
    "mask_path",
    "identity_reference_path",
    "cross_view_reference_path",
}


def confined_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("media path must be non-empty and relative")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("media path escapes configured root") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", payload) if isinstance(payload, dict) else payload
    if not isinstance(candidates, list):
        raise ValueError("candidate manifest must contain a list")
    return [dict(candidate) for candidate in candidates]


def candidate_queue_for_reviewer(
    candidates: list[dict[str, Any]],
    reviewer_id: str,
    *,
    review_mode: str = "candidate",
    adjudication_contexts: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    if reviewer_id not in ("REVIEWER_A", "REVIEWER_B", "ADJUDICATOR"):
        raise ValueError("invalid reviewer ID")
    if review_mode not in ("candidate", "group"):
        raise ValueError("invalid review mode")
    safe = [
        {key: value for key, value in candidate.items() if key in CANDIDATE_QUEUE_FIELDS}
        for candidate in candidates
    ]
    contexts = adjudication_contexts or {}
    if reviewer_id == "ADJUDICATOR":
        if review_mode == "candidate":
            safe = [
                {**candidate, "adjudication_context": contexts[candidate["candidate_id"]]}
                for candidate in safe
                if candidate.get("candidate_id") in contexts
            ]
        else:
            safe = [
                {
                    **candidate,
                    "group_adjudication_context": contexts[
                        f"{candidate['identity_id']}__{candidate['garment_id']}"
                    ],
                }
                for candidate in safe
                if f"{candidate.get('identity_id')}__{candidate.get('garment_id')}" in contexts
            ]
    return {
        "reviewer_id": reviewer_id,
        "decision_visibility": (
            "ADJUDICATION_CONTEXT_EXPLICITLY_AUTHORIZED"
            if reviewer_id == "ADJUDICATOR"
            else "PEER_INITIAL_DECISIONS_HIDDEN"
        ),
        "candidates": safe,
    }


def load_jsonl_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def adjudication_contexts(
    events: list[dict[str, Any]], *, group: bool
) -> dict[str, dict[str, str]]:
    id_field = "group_id" if group else "candidate_id"
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for event in events:
        if event.get("review_stage") != "INITIAL" or id_field not in event:
            continue
        grouped.setdefault(str(event[id_field]), {})[str(event["reviewer_id"])] = event
    contexts: dict[str, dict[str, str]] = {}
    for record_id, reviewers in grouped.items():
        reviewer_a = reviewers.get("REVIEWER_A")
        reviewer_b = reviewers.get("REVIEWER_B")
        if resolve_initial_decision(reviewer_a, reviewer_b) == "ADJUDICATION_REQUIRED":
            contexts[record_id] = {
                "REVIEWER_A": str(reviewer_a["decision"]),
                "REVIEWER_B": str(reviewer_b["decision"]),
            }
    return contexts


def validate_review_transition(
    existing: list[dict[str, Any]],
    event: dict[str, Any],
    *,
    group: bool,
) -> None:
    id_field = "group_id" if group else "candidate_id"
    related = [row for row in existing if row.get(id_field) == event[id_field]]
    if any(
        row.get("reviewer_id") == event["reviewer_id"]
        and row.get("review_stage") == event["review_stage"]
        for row in related
    ):
        raise ValueError("duplicate reviewer stage for record")
    if event["review_stage"] == "ADJUDICATION":
        initial = {
            row["reviewer_id"]: row
            for row in related
            if row.get("review_stage") == "INITIAL"
        }
        if resolve_initial_decision(
            initial.get("REVIEWER_A"), initial.get("REVIEWER_B")
        ) != "ADJUDICATION_REQUIRED":
            raise ValueError("adjudication requires a complete disagreeing initial pair")


class ReviewServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        media_root: Path,
        candidate_manifest: Path,
        review_log: Path,
        group_review_log: Path,
    ) -> None:
        if address[0] not in ("127.0.0.1", "localhost"):
            raise ValueError("review server must bind to localhost")
        super().__init__(("127.0.0.1", address[1]), handler)
        self.media_root = media_root.resolve()
        self.candidate_manifest = candidate_manifest.resolve()
        self.review_log = review_log.resolve()
        self.group_review_log = group_review_log.resolve()
        self.review_lock = threading.Lock()


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in STATIC_FILES:
            self._serve_static(STATIC_FILES[parsed.path])
            return
        if parsed.path == "/api/candidates":
            self._serve_candidates(parse_qs(parsed.query))
            return
        if parsed.path == "/media":
            self._serve_media(parse_qs(parsed.query))
            return
        self._json_error(HTTPStatus.NOT_FOUND, "NOT_FOUND")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/reviews":
            self._append_review(group=False)
            return
        if parsed.path == "/api/group-reviews":
            self._append_review(group=True)
            return
        self._json_error(HTTPStatus.NOT_FOUND, "NOT_FOUND")

    def log_message(self, format: str, *args: object) -> None:
        # Avoid request payloads and query strings in logs.
        print(f"review_server {self.client_address[0]} {args[1] if len(args) > 1 else '-'}")

    def _serve_static(self, name: str) -> None:
        path = Path(__file__).resolve().parent / name
        self._send_file(path, "text/html; charset=utf-8" if name.endswith(".html") else None)

    def _serve_candidates(self, query: dict[str, list[str]]) -> None:
        reviewer_id = query.get("reviewer_id", [""])[0]
        review_mode = query.get("review_mode", ["candidate"])[0]
        if reviewer_id not in ("REVIEWER_A", "REVIEWER_B", "ADJUDICATOR"):
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_REVIEWER_ID")
            return
        candidates = load_candidates(self.server.candidate_manifest)
        # Initial reviewers receive candidate evidence only, never peer decisions.
        context = None
        if reviewer_id == "ADJUDICATOR":
            context = adjudication_contexts(
                load_jsonl_events(
                    self.server.group_review_log if review_mode == "group" else self.server.review_log
                ),
                group=review_mode == "group",
            )
        try:
            queue = candidate_queue_for_reviewer(
                candidates,
                reviewer_id,
                review_mode=review_mode,
                adjudication_contexts=context,
            )
        except ValueError:
            self._json_error(HTTPStatus.BAD_REQUEST, "INVALID_REVIEW_MODE")
            return
        self._send_json(HTTPStatus.OK, queue)

    def _serve_media(self, query: dict[str, list[str]]) -> None:
        relative = unquote(query.get("path", [""])[0])
        try:
            path = confined_path(self.server.media_root, relative)
        except (ValueError, FileNotFoundError):
            self._json_error(HTTPStatus.NOT_FOUND, "MEDIA_NOT_FOUND")
            return
        self._send_file(path)

    def _append_review(self, *, group: bool) -> None:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self._json_error(HTTPStatus.LENGTH_REQUIRED, "CONTENT_LENGTH_REQUIRED")
            return
        length = int(length_header)
        if length < 1 or length > MAX_BODY_BYTES:
            self._json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "INVALID_BODY_SIZE")
            return
        try:
            event = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(event, dict):
                raise ValueError("event must be an object")
            validate_review_event(event, group=group)
            if event["review_stage"] == "INITIAL" and event["reviewer_id"] not in (
                "REVIEWER_A",
                "REVIEWER_B",
            ):
                raise ValueError("initial reviewer must be REVIEWER_A or REVIEWER_B")
            if event["review_stage"] == "ADJUDICATION" and event["reviewer_id"] != "ADJUDICATOR":
                raise ValueError("adjudication requires ADJUDICATOR")
            log_path = self.server.group_review_log if group else self.server.review_log
            with self.server.review_lock:
                existing = load_jsonl_events(log_path)
                validate_review_transition(existing, event, group=group)
                output = append_jsonl_event(log_path, event)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
            self._json_error(HTTPStatus.BAD_REQUEST, type(exc).__name__)
            return
        self._send_json(
            HTTPStatus.CREATED,
            {"event_hash": output["event_hash"], "append_only": True},
        )

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        payload = path.read_bytes()
        guessed = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", guessed)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _json_error(self, status: HTTPStatus, code: str) -> None:
        self._send_json(status, {"error": code})

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; connect-src 'self'",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--review-log", type=Path, required=True)
    parser.add_argument("--group-review-log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = ReviewServer(
        ("127.0.0.1", args.port),
        ReviewHandler,
        media_root=args.media_root,
        candidate_manifest=args.candidate_manifest,
        review_log=args.review_log,
        group_review_log=args.group_review_log,
    )
    print(f"review_server=http://127.0.0.1:{server.server_port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
