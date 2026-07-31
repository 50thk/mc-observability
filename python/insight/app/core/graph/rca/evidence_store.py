import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.core.graph.utils.token_counter import count_tokens

from .models import EvidenceRecord

DEFAULT_SINGLE_TOOL_MAX_TOKENS = 4_915
DEFAULT_RECORD_BUDGET_TOKENS = 19_660


@dataclass(frozen=True, slots=True)
class _Artifact:
    reference: str
    path: Path
    source: str
    capability: str
    tool: str
    query: dict[str, Any]
    sha256: str
    size_bytes: int
    token_count: int


class EvidenceStore:
    def __init__(
        self,
        *,
        storage_dir: Path | str | None = None,
        model_name: str = "gpt-4",
        single_tool_max_tokens: int = DEFAULT_SINGLE_TOOL_MAX_TOKENS,
        record_budget_tokens: int = DEFAULT_RECORD_BUDGET_TOKENS,
    ):
        self._owned_temp_dir = TemporaryDirectory(prefix="rca-evidence-") if storage_dir is None else None
        storage_dir = storage_dir or self._owned_temp_dir.name
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.single_tool_max_tokens = single_tool_max_tokens
        self.record_budget_tokens = record_budget_tokens
        self._artifacts: dict[str, _Artifact] = {}
        self._records: dict[str, EvidenceRecord] = {}
        self._used_record_tokens = 0
        self._capability_budgets: dict[str, int] = {}
        self._used_by_capability: dict[str, int] = {}
        self._spilled_refs: set[str] = set()
        self._discovery_refs: set[str] = set()
        self._inspected_refs: set[str] = set()
        self._budget_exhausted_capabilities: set[str] = set()
        self._unavailable_capabilities: set[str] = set()

    def capture(
        self,
        *,
        source: str,
        capability: str,
        tool: str,
        query: dict[str, Any],
        value: Any,
    ) -> dict[str, Any]:
        normalized, canonical = _canonical(value)
        reference, digest = _reference(
            source,
            capability,
            tool,
            query,
            canonical,
        )
        record = EvidenceRecord(
            evidence_id=reference,
            source=source,
            capability=capability,
            signal=capability,
            observation=canonical,
            tool=tool,
            query=dict(query),
        )
        inline = {
            "records": [record.model_dump(mode="json")],
            "truncated": False,
        }
        if self._fits(inline):
            if self._admit(record):
                return inline
            self._budget_exhausted_capabilities.add(capability)
            return self._error("evidence_budget_exhausted")

        return self._spill(
            reference=reference,
            digest=digest,
            source=source,
            capability=capability,
            tool=tool,
            query=query,
            normalized=normalized,
            canonical=canonical,
            recording=True,
        )

    def capture_discovery(
        self,
        *,
        source: str,
        capability: str,
        value: Any,
    ) -> Any:
        normalized, canonical = _canonical(value)
        if self._fits(normalized):
            return normalized
        reference, digest = _reference(
            source,
            capability,
            "discovery",
            {},
            canonical,
        )
        return self._spill(
            reference=reference,
            digest=digest,
            source=source,
            capability=capability,
            tool="discovery",
            query={},
            normalized=normalized,
            canonical=canonical,
            recording=False,
        )

    def inspect(
        self,
        evidence_ref: str,
        *,
        path: str = "",
        offset: int = 0,
        limit: int = 10,
    ) -> dict[str, Any]:
        artifact = self._artifacts.get(evidence_ref)
        if artifact is None:
            return self._error("evidence_ref_not_found")
        if offset < 0:
            return self._error("invalid_offset")
        if not 1 <= limit <= 100:
            return self._error("invalid_limit")

        try:
            artifact_path = artifact.path.resolve()
            if not artifact_path.is_relative_to(self.storage_dir):
                self._unavailable_capabilities.add(artifact.capability)
                return self._error("evidence_path_invalid")
            raw = artifact_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._unavailable_capabilities.add(artifact.capability)
            return self._error("evidence_file_not_found")
        except OSError:
            self._unavailable_capabilities.add(artifact.capability)
            return self._error("evidence_read_failed")

        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            self._unavailable_capabilities.add(artifact.capability)
            return self._error("evidence_invalid_json")

        try:
            selected = _resolve_pointer(value, path)
        except (KeyError, IndexError, TypeError, ValueError):
            # A wrong pointer is a navigation mistake, not lost evidence: hand back the
            # real paths so the next call can land instead of ending the capability.
            return self._error("evidence_path_not_found", outline=_outline(value))

        _, selected_canonical = _canonical(selected)
        base = {
            "error": None,
            "evidence_ref": evidence_ref,
            "path": path,
            "token_count": count_tokens(selected_canonical, self.model_name),
        }
        view = _view(
            selected,
            path,
            offset,
            limit,
            lambda candidate: self._fits({**base, **candidate}),
            self.model_name,
        )
        result = {**base, **view}
        if not self._fits(result):
            # Asking for too much is recoverable — the evidence is still there, so the
            # capability must not be marked exhausted over one oversized request.
            return self._error(
                "inspection_response_too_large",
                next_action="narrow_path_or_limit",
                outline=_outline(selected, max_entries=8, max_depth=2),
            )

        if evidence_ref in self._discovery_refs:
            return result

        observation = _serialize({key: value for key, value in result.items() if key != "error"})
        view_digest = hashlib.sha256(f"{evidence_ref}:{path}:{offset}:{observation}".encode()).hexdigest()[:16]
        record = EvidenceRecord(
            evidence_id=f"{artifact.source}:{view_digest}",
            source=artifact.source,
            capability=artifact.capability,
            signal=artifact.capability,
            observation=observation,
            tool=artifact.tool,
            query=artifact.query,
        )
        if not self._admit(record):
            self._budget_exhausted_capabilities.add(artifact.capability)
            return self._error("evidence_budget_exhausted")
        self._inspected_refs.add(evidence_ref)
        return result

    def reserve(self, capability: str, share_count: int) -> int:
        """Give a planned capability its own slice of the record budget.

        Capabilities in one round collect in parallel against a single store. Without
        a per-capability slice the first collector to finish can spend the whole
        budget, so which evidence survives depends on which MCP server answered
        faster — the same incident then yields different analyses run to run.

        An unreserved capability keeps the full budget, which is what standalone and
        single-capability callers expect.
        """
        share = self.record_budget_tokens // max(share_count, 1)
        return self._capability_budgets.setdefault(capability, share)

    def records_for(self, capability: str) -> list[EvidenceRecord]:
        return [record for record in self._records.values() if record.capability == capability]

    def is_spilled(self, capability: str) -> bool:
        return any(self._artifacts[reference].capability == capability for reference in self._spilled_refs)

    def has_uninspected(self, capability: str) -> bool:
        return any(
            reference not in self._inspected_refs and self._artifacts[reference].capability == capability
            for reference in self._spilled_refs
        )

    def budget_exhausted(self, capability: str) -> bool:
        return capability in self._budget_exhausted_capabilities

    def has_unavailable(self, capability: str) -> bool:
        return capability in self._unavailable_capabilities

    def _spill(
        self,
        *,
        reference: str,
        digest: str,
        source: str,
        capability: str,
        tool: str,
        query: dict[str, Any],
        normalized: Any,
        canonical: str,
        recording: bool,
    ) -> dict[str, Any]:
        path = (self.storage_dir / f"{digest}.json").resolve()
        if not path.is_relative_to(self.storage_dir):
            self._unavailable_capabilities.add(capability)
            return self._error(
                "evidence_storage_failed",
                inspection_required=False,
                next_action="narrow_query",
            )
        try:
            if reference not in self._artifacts:
                path.write_text(canonical, encoding="utf-8")
                self._artifacts[reference] = _Artifact(
                    reference=reference,
                    path=path,
                    source=source,
                    capability=capability,
                    tool=tool,
                    query=dict(query),
                    sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                    size_bytes=len(canonical.encode()),
                    token_count=count_tokens(canonical, self.model_name),
                )
        except OSError:
            self._unavailable_capabilities.add(capability)
            return self._error(
                "evidence_storage_failed",
                inspection_required=False,
                next_action="narrow_query",
            )

        artifact = self._artifacts[reference]
        if recording:
            self._spilled_refs.add(reference)
        else:
            self._discovery_refs.add(reference)
        base = {
            "evidence_ref": reference,
            "stored": True,
            "type": _kind(normalized),
            "token_count": artifact.token_count,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "inspection_required": True,
            "outline": _outline(normalized),
        }
        preview = _view(
            normalized,
            "",
            0,
            5,
            lambda candidate: self._fits({**base, "root_preview": candidate}),
            self.model_name,
        )
        result = {**base, "root_preview": preview}
        if self._fits(result):
            return result
        # The outline is worth more than the preview: keep the map, drop the sample.
        result["root_preview"] = {"kind": _kind(normalized)}
        if self._fits(result):
            return result
        result["outline"] = _outline(normalized, max_entries=4, max_depth=2)
        if self._fits(result):
            return result
        # A spill result that does not fit the per-call budget is worse than no map.
        result.pop("outline", None)
        return result

    def _admit(self, record: EvidenceRecord) -> bool:
        if record.evidence_id in self._records:
            return True
        size = count_tokens(_serialize(record.model_dump(mode="json")), self.model_name)
        if self._used_record_tokens + size > self.record_budget_tokens:
            return False
        capability_used = self._used_by_capability.get(record.capability, 0)
        capability_budget = self._capability_budgets.get(
            record.capability,
            self.record_budget_tokens,
        )
        if capability_used + size > capability_budget:
            return False
        self._records[record.evidence_id] = record
        self._used_record_tokens += size
        self._used_by_capability[record.capability] = capability_used + size
        return True

    def _fits(self, value: Any) -> bool:
        return count_tokens(_serialize(value), self.model_name) <= self.single_tool_max_tokens

    def _error(self, code: str, **details: Any) -> dict[str, Any]:
        result = {"error": code, **details}
        return result if self._fits(result) else {"error": code}


def _canonical(value: Any) -> tuple[Any, str]:
    canonical = _serialize(value)
    return json.loads(canonical), canonical


def _serialize(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _reference(
    source: str,
    capability: str,
    tool: str,
    query: dict[str, Any],
    canonical: str,
) -> tuple[str, str]:
    metadata = _serialize(
        {
            "source": source,
            "capability": capability,
            "tool": tool,
            "query": query,
        }
    )
    digest = hashlib.sha256(f"{metadata}\0{canonical}".encode()).hexdigest()
    return f"{source}:{digest[:16]}", digest


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    return "number"


def _pointer_token(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _outline(value: Any, *, max_entries: int = 12, max_depth: int = 4) -> list[dict[str, Any]]:
    """Map the shape of a payload as addressable JSON Pointer paths.

    A preview of the first few elements does not tell an agent where anything lives,
    so it has to guess pointers into a structure it has never seen — and one wrong
    guess used to end the capability. This lists what is actually there.
    """
    entries: list[dict[str, Any]] = []
    queue: list[tuple[str, Any, int]] = [("", value, 0)]
    while queue and len(entries) < max_entries:
        path, node, depth = queue.pop(0)
        if isinstance(node, dict):
            keys = list(node)[:12]
            entries.append({"path": path or "", "kind": "object", "keys": keys})
            if depth < max_depth:
                queue.extend(
                    (f"{path}/{_pointer_token(key)}", node[key], depth + 1)
                    for key in keys
                    if isinstance(node[key], (dict, list))
                )
        elif isinstance(node, list):
            entries.append({"path": path or "", "kind": "array", "len": len(node)})
            if depth < max_depth and node and isinstance(node[0], (dict, list)):
                queue.append((f"{path}/0", node[0], depth + 1))
        else:
            # Scalars matter too: "a 40k-char string lives here" is what tells the
            # agent to window it with offset/limit rather than ask for the whole value.
            entry = {"path": path or "", "kind": _kind(node)}
            if isinstance(node, str):
                entry["len"] = len(node)
            entries.append(entry)
    return entries


def _resolve_pointer(value: Any, path: str) -> Any:
    if path == "":
        return value
    if not path.startswith("/"):
        raise ValueError("JSON Pointer must start with /")
    current = value
    for raw_token in path[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", raw_token):
            raise ValueError("invalid JSON Pointer escape")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit():
                raise ValueError("array index must be a non-negative integer")
            current = current[int(token)]
        else:
            raise TypeError("cannot traverse a scalar")
    return current


def _child(
    value: Any,
    path: str,
    *,
    label: str,
    label_value: Any,
    model_name: str,
) -> dict[str, Any]:
    _, canonical = _canonical(value)
    return {
        label: label_value,
        "path": path,
        "kind": _kind(value),
        "token_count": count_tokens(canonical, model_name),
        "value": value,
    }


def _page(
    kind: str,
    values: list[Any],
    path: str,
    offset: int,
    limit: int,
    fits: Callable[[dict[str, Any]], bool],
    model_name: str,
) -> dict[str, Any]:
    total = len(values)
    children = []
    result = {
        "kind": kind,
        "range": {"offset": offset, "returned": 0, "total": total},
        "children": children,
        "has_more": offset < total,
    }
    for position in range(offset, min(total, offset + limit)):
        if kind == "object":
            key, value = values[position]
            child = _child(
                value,
                f"{path}/{_pointer_token(key)}",
                label="key",
                label_value=key,
                model_name=model_name,
            )
        else:
            value = values[position]
            child = _child(
                value,
                f"{path}/{position}",
                label="index",
                label_value=position,
                model_name=model_name,
            )
        children.append(child)
        result["range"]["returned"] = len(children)
        result["has_more"] = offset + len(children) < total
        if fits(result):
            continue
        child.pop("value")
        if fits(result):
            continue
        children.pop()
        result["range"]["returned"] = len(children)
        result["has_more"] = offset + len(children) < total
        break
    return result


def _string_view(
    value: str,
    offset: int,
    limit: int,
    fits: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    requested = min(limit, max(0, len(value) - offset))
    low, high = 0, requested
    best = {
        "kind": "string",
        "range": {"offset": offset, "returned": 0, "total": len(value)},
        "value": "",
        "has_more": offset < len(value),
    }
    while low <= high:
        length = (low + high) // 2
        candidate = {
            "kind": "string",
            "range": {
                "offset": offset,
                "returned": length,
                "total": len(value),
            },
            "value": value[offset : offset + length],
            "has_more": offset + length < len(value),
        }
        if fits(candidate):
            best = candidate
            low = length + 1
        else:
            high = length - 1
    return best


def _view(
    value: Any,
    path: str,
    offset: int,
    limit: int,
    fits: Callable[[dict[str, Any]], bool],
    model_name: str,
) -> dict[str, Any]:
    if isinstance(value, dict):
        return _page(
            "object",
            list(value.items()),
            path,
            offset,
            limit,
            fits,
            model_name,
        )
    if isinstance(value, list):
        return _page("array", value, path, offset, limit, fits, model_name)
    if isinstance(value, str):
        return _string_view(value, offset, limit, fits)
    return {
        "kind": _kind(value),
        "range": {"offset": 0, "returned": 1, "total": 1},
        "value": value,
        "has_more": False,
    }
