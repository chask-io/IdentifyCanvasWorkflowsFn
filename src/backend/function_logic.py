"""
Business logic for IdentifyCanvasWorkflowsFn.

Validates caller-provided workflow groupings against the canvas and
(optionally) persists them as named CanvasSelections. This Lambda performs
no LLM calls — the caller (canvas_designer agent) is responsible for
proposing the groupings.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from api.canvas_requests import canvas_api_manager
from chask_foundation.backend.models import OrchestrationEvent

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FunctionBackend:
    """Backend for IdentifyCanvasWorkflowsFn."""

    def __init__(
        self,
        orchestration_event: OrchestrationEvent,
        openai_api_key: Optional[str] = None,
    ):
        # openai_api_key is accepted for handler compatibility but unused —
        # this Lambda no longer performs LLM calls.
        del openai_api_key
        self.orchestration_event = orchestration_event
        logger.info(
            "Initialized FunctionBackend for org: %s",
            orchestration_event.organization.organization_id,
        )

    # ──────────────────────────── entry point

    def process_request(self) -> str:
        tool_args = self._extract_tool_args()

        canvas_uuid = tool_args.get("canvas_uuid")
        if not canvas_uuid:
            raise ValueError("Missing required parameter: canvas_uuid")

        workflows_raw = tool_args.get("workflows")
        if workflows_raw is None:
            raise ValueError("Missing required parameter: workflows")

        proposed = self._parse_workflows(workflows_raw)

        create_selections = self._coerce_bool(
            tool_args.get("create_selections", True), default=True
        )

        canvas = self._fetch_canvas_detail(canvas_uuid)
        nodes = self._extract_nodes(canvas)
        edges = self._extract_edges(canvas)

        cleaned = self._clean_workflows(proposed, nodes, edges)

        if not cleaned:
            return (
                "Los flujos propuestos no contienen nodos válidos del canvas. "
                "No se crearon selecciones."
            )

        if create_selections:
            cleaned = self._persist_selections(canvas_uuid, cleaned)

        return self._format_response(cleaned, persisted=create_selections)

    # ──────────────────────────── parsing

    @staticmethod
    def _parse_workflows(raw: Any) -> List[Any]:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"'workflows' must be a JSON array (got invalid string): {exc}"
                ) from exc
        if isinstance(raw, dict):
            raw = raw.get("workflows", [raw])
        if not isinstance(raw, list):
            raise ValueError(
                f"'workflows' must be a list, got {type(raw).__name__}"
            )
        return raw

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            s = value.strip().lower()
            if s == "":
                return default
            return s not in ("false", "0", "no")
        return bool(value)

    # ──────────────────────────── canvas fetching

    def _fetch_canvas_detail(self, canvas_uuid: str) -> Dict[str, Any]:
        response = canvas_api_manager.call(
            "get_canvas_detail",
            canvas_uuid=canvas_uuid,
            access_token=self.orchestration_event.access_token,
            organization_id=self.orchestration_event.organization.organization_id,
        )
        if response.get("status_code") not in (200, 201):
            raise RuntimeError(
                f"No se pudo obtener el canvas {canvas_uuid}: "
                f"{self._api_error_detail(response)}"
            )
        return response

    @staticmethod
    def _extract_nodes(canvas: Dict[str, Any]) -> List[Dict[str, Any]]:
        elements = canvas.get("elements") or {}
        nodes = elements.get("nodes") or canvas.get("nodes") or []
        return [n for n in nodes if isinstance(n, dict) and n.get("id")]

    @staticmethod
    def _extract_edges(canvas: Dict[str, Any]) -> List[Dict[str, Any]]:
        elements = canvas.get("elements") or {}
        edges = elements.get("edges") or canvas.get("edges") or []
        return [e for e in edges if isinstance(e, dict) and e.get("id")]

    # ──────────────────────────── validation / cleanup

    def _clean_workflows(
        self,
        workflows: List[Any],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        valid_node_ids = {str(n["id"]) for n in nodes}
        cleaned: List[Dict[str, Any]] = []

        for idx, wf in enumerate(workflows):
            if not isinstance(wf, dict):
                logger.warning(
                    "Skipping non-dict workflow at index %d: %r", idx, wf
                )
                continue

            raw_ids = wf.get("node_ids") or []
            if not isinstance(raw_ids, list):
                logger.warning(
                    "Workflow %r has non-list node_ids, skipping",
                    wf.get("name"),
                )
                continue

            kept: List[str] = []
            seen: set = set()
            for nid in raw_ids:
                nid_str = str(nid)
                if nid_str not in valid_node_ids:
                    logger.warning(
                        "Workflow %r references unknown node_id %s, dropping",
                        wf.get("name"), nid_str,
                    )
                    continue
                if nid_str in seen:
                    continue
                seen.add(nid_str)
                kept.append(nid_str)

            if not kept:
                logger.warning(
                    "Workflow %r has no valid node_ids after cleanup, dropping",
                    wf.get("name"),
                )
                continue

            edge_ids = self._derive_edge_ids(kept, edges)
            name = str(wf.get("name") or "").strip()
            if not name:
                name = f"Flujo {idx + 1}"
            description = str(wf.get("description") or "").strip()

            cleaned.append({
                "name": name,
                "description": description,
                "node_ids": kept,
                "edge_ids": edge_ids,
            })

        return cleaned

    @staticmethod
    def _derive_edge_ids(
        node_ids: List[str], edges: List[Dict[str, Any]]
    ) -> List[str]:
        node_set = set(node_ids)
        edge_ids: List[str] = []
        for edge in edges:
            source = str(edge.get("source") or edge.get("from") or "")
            target = str(edge.get("target") or edge.get("to") or "")
            if source in node_set and target in node_set:
                edge_ids.append(str(edge.get("id")))
        return edge_ids

    # ──────────────────────────── selection persistence

    def _persist_selections(
        self, canvas_uuid: str, workflows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        persisted: List[Dict[str, Any]] = []
        for wf in workflows:
            element_ids = {"nodes": wf["node_ids"], "edges": wf["edge_ids"]}
            # `name` is positional because ApiManager.call reserves the first
            # kwarg slot for the endpoint name.
            response = canvas_api_manager.call(
                "create_selection",
                canvas_uuid, wf["name"],
                element_ids=element_ids,
                access_token=self.orchestration_event.access_token,
                organization_id=self.orchestration_event.organization.organization_id,
            )

            if response.get("status_code") not in (200, 201):
                logger.error(
                    "Failed to create selection %s: %s",
                    wf["name"], self._api_error_detail(response),
                )
                persisted.append({
                    **wf,
                    "selection_uuid": None,
                    "error": self._api_error_detail(response),
                })
                continue

            persisted.append({**wf, "selection_uuid": response.get("uuid")})

        return persisted

    # ──────────────────────────── response formatting

    def _format_response(
        self, workflows: List[Dict[str, Any]], persisted: bool
    ) -> str:
        if len(workflows) == 1:
            return self._format_single(workflows[0], persisted)
        return self._format_multiple(workflows, persisted)

    @staticmethod
    def _format_single(wf: Dict[str, Any], persisted: bool) -> str:
        name = wf["name"]
        description = wf["description"]
        n_nodes = len(wf["node_ids"])
        n_edges = len(wf["edge_ids"])

        lead = f"1 flujo identificado: **{name}**"
        if description:
            lead += f" — {description}"

        if persisted and wf.get("selection_uuid"):
            detail = (
                f"(Selección {wf['selection_uuid']} creada, "
                f"{n_nodes} nodos, {n_edges} conexiones.)"
            )
        elif persisted:
            err = wf.get("error") or "error desconocido"
            detail = f"(No se pudo crear la selección: {err}.)"
        else:
            detail = f"({n_nodes} nodos, {n_edges} conexiones.)"

        return f"{lead}\n{detail}"

    @staticmethod
    def _format_multiple(
        workflows: List[Dict[str, Any]], persisted: bool
    ) -> str:
        lines = ["## Flujos identificados", ""]
        for wf in workflows:
            name = wf["name"]
            description = wf["description"]
            n_nodes = len(wf["node_ids"])
            n_edges = len(wf["edge_ids"])

            bullet = f"- **{name}**"
            if description:
                bullet += f" — {description}"
            lines.append(bullet)

            if persisted and wf.get("selection_uuid"):
                lines.append(
                    f"  (Selección {wf['selection_uuid']} creada, "
                    f"{n_nodes} nodos, {n_edges} conexiones.)"
                )
            elif persisted:
                err = wf.get("error") or "error desconocido"
                lines.append(f"  (No se pudo crear la selección: {err}.)")
            else:
                lines.append(f"  ({n_nodes} nodos, {n_edges} conexiones.)")
        return "\n".join(lines)

    # ──────────────────────────── helpers

    @staticmethod
    def _api_error_detail(response: Dict[str, Any]) -> str:
        if "error" in response:
            return str(response["error"])
        detail = {k: v for k, v in response.items() if k != "status_code"}
        return json.dumps(detail) if detail else "Unknown error"

    def _extract_tool_args(self) -> Dict[str, Any]:
        extra_params = self.orchestration_event.extra_params or {}
        tool_calls = extra_params.get("tool_calls", [])
        if not tool_calls:
            logger.warning("No tool calls found in orchestration event")
            return {}
        return tool_calls[0].get("args", {}) or {}
