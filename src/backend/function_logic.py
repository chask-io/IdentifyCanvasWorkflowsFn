"""
Business logic for IdentifyCanvasWorkflowsFn (v3).

Validates caller-provided selection groupings against the canvas and
(optionally) persists them as named CanvasSelections. The caller supplies
both node_ids and edge_ids per selection — this Lambda performs no LLM
calls and does not auto-derive edges. Any malformed or out-of-canvas ID
rejects the WHOLE call.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set

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
        # this Lambda performs no LLM calls.
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
            raise ValueError(
                "Llamada rechazada: el parámetro 'canvas_uuid' es requerido."
            )

        selections_raw = tool_args.get("selections")
        selections = self._parse_selections(selections_raw)

        create_selections = self._coerce_bool(
            tool_args.get("create_selections", True), default=True
        )

        canvas = self._fetch_canvas_detail(canvas_uuid)
        nodes = self._extract_nodes(canvas)
        edges = self._extract_edges(canvas)
        valid_node_ids = {str(n["id"]) for n in nodes}
        valid_edge_ids = {str(e["id"]) for e in edges}

        cleaned = self._validate_selections(
            selections, valid_node_ids, valid_edge_ids
        )

        if not create_selections:
            return self._format_preview(cleaned)

        persisted = self._persist_selections(canvas_uuid, cleaned)
        return self._format_created(persisted)

    # ──────────────────────────── parsing

    @staticmethod
    def _parse_selections(raw: Any) -> Dict[str, Any]:
        if raw is None:
            raise ValueError(
                "Llamada rechazada: el parámetro 'selections' es requerido "
                "y debe ser un objeto no vacío."
            )
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Llamada rechazada: 'selections' debe ser un objeto JSON "
                    f"(se recibió una cadena inválida: {exc})."
                ) from exc
        if not isinstance(raw, dict):
            raise ValueError(
                "Llamada rechazada: 'selections' debe ser un objeto (dict) "
                f"donde cada clave es el nombre de la selección (recibido: "
                f"{type(raw).__name__})."
            )
        if not raw:
            raise ValueError(
                "Llamada rechazada: 'selections' está vacío. Provee al "
                "menos una selección con nombre, node_ids y edge_ids."
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

    # ──────────────────────────── validation (fail-fast)

    def _validate_selections(
        self,
        selections: Dict[str, Any],
        valid_node_ids: Set[str],
        valid_edge_ids: Set[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Walk every entry and either return a cleaned dict or raise
        ValueError with a Spanish, LLM-readable rejection message.
        """
        cleaned: Dict[str, Dict[str, Any]] = {}

        for name, spec in selections.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "Llamada rechazada: cada clave de 'selections' debe ser "
                    "un nombre no vacío (string)."
                )

            if not isinstance(spec, dict):
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' debe ser un "
                    f"objeto con description, node_ids y edge_ids "
                    f"(recibido: {type(spec).__name__})."
                )

            if "node_ids" not in spec:
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' no incluye "
                    "el campo requerido 'node_ids' (array de identificadores "
                    "de nodos)."
                )
            node_ids = spec["node_ids"]
            if not isinstance(node_ids, list):
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' tiene "
                    f"'node_ids' que no es un array (recibido: "
                    f"{type(node_ids).__name__})."
                )
            if len(node_ids) == 0:
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' tiene "
                    "node_ids vacío. Cada selección debe incluir al menos "
                    "un nodo."
                )

            if "edge_ids" not in spec:
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' no incluye "
                    "'edge_ids'. Provee un array, aunque sea vacío []."
                )
            edge_ids = spec["edge_ids"]
            if not isinstance(edge_ids, list):
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' tiene "
                    f"'edge_ids' que no es un array (recibido: "
                    f"{type(edge_ids).__name__})."
                )

            description = spec.get("description", "")
            if description is None:
                description = ""
            if not isinstance(description, str):
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' tiene "
                    f"'description' que no es un string (recibido: "
                    f"{type(description).__name__})."
                )

            node_ids_str = [str(nid) for nid in node_ids]
            edge_ids_str = [str(eid) for eid in edge_ids]

            unknown_nodes = [
                nid for nid in node_ids_str if nid not in valid_node_ids
            ]
            if unknown_nodes:
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' referencia "
                    f"nodos inexistentes en el canvas: {unknown_nodes}. "
                    "Verifica los IDs llamando a GetCanvasContextFn primero."
                )

            unknown_edges = [
                eid for eid in edge_ids_str if eid not in valid_edge_ids
            ]
            if unknown_edges:
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' referencia "
                    f"edges inexistentes en el canvas: {unknown_edges}. "
                    "Verifica los IDs en el canvas."
                )

            cleaned[name] = {
                "description": description.strip(),
                "node_ids": node_ids_str,
                "edge_ids": edge_ids_str,
            }

        return cleaned

    # ──────────────────────────── selection persistence

    def _persist_selections(
        self,
        canvas_uuid: str,
        selections: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for name, spec in selections.items():
            element_ids = {
                "nodes": spec["node_ids"],
                "edges": spec["edge_ids"],
            }
            # `name` is positional because ApiManager.call reserves the first
            # kwarg slot for the endpoint name.
            response = canvas_api_manager.call(
                "create_selection",
                canvas_uuid, name,
                element_ids=element_ids,
                access_token=self.orchestration_event.access_token,
                organization_id=self.orchestration_event.organization.organization_id,
            )

            entry = {
                "name": name,
                "description": spec["description"],
                "node_ids": spec["node_ids"],
                "edge_ids": spec["edge_ids"],
            }

            if response.get("status_code") not in (200, 201):
                err = self._api_error_detail(response)
                logger.error("Failed to create selection %s: %s", name, err)
                entry["selection_uuid"] = None
                entry["error"] = err
            else:
                entry["selection_uuid"] = response.get("uuid")
                entry["error"] = None

            results.append(entry)

        return results

    # ──────────────────────────── response formatting

    @staticmethod
    def _format_preview(selections: Dict[str, Dict[str, Any]]) -> str:
        lines = ["## Selecciones validadas (preview, no persistidas)", ""]
        for name, spec in selections.items():
            description = spec["description"]
            n_nodes = len(spec["node_ids"])
            n_edges = len(spec["edge_ids"])
            bullet = f"- **{name}**"
            if description:
                bullet += f" — {description}"
            bullet += f" ({n_nodes} nodos, {n_edges} edges)"
            lines.append(bullet)
        return "\n".join(lines)

    @staticmethod
    def _format_created(results: List[Dict[str, Any]]) -> str:
        lines = ["## Selecciones creadas", ""]
        for entry in results:
            name = entry["name"]
            description = entry["description"]
            n_nodes = len(entry["node_ids"])
            n_edges = len(entry["edge_ids"])

            bullet = f"- **{name}**"
            if description:
                bullet += f" — {description}"
            lines.append(bullet)

            if entry.get("selection_uuid"):
                lines.append(
                    f"  (Selección {entry['selection_uuid']} creada, "
                    f"{n_nodes} nodos, {n_edges} edges.)"
                )
            else:
                err = entry.get("error") or "error desconocido"
                lines.append(f"  (No se pudo crear la selección: {err}.)")
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
