"""
Business logic for IdentifyCanvasWorkflowsFn (v3).

Validates caller-provided selection groupings against the canvas and
persists them as named CanvasSelections. The caller may supply legacy
node_ids/edge_ids or v2 element_ids with lanes, nodes, edges, data,
credentials, and flow.order. Any malformed or out-of-canvas ID rejects
the WHOLE call.
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

        canvas = self._fetch_canvas_detail(canvas_uuid)
        nodes = self._extract_nodes(canvas)
        edges = self._extract_edges(canvas)
        lanes = self._extract_lanes(canvas)
        data_catalog = self._extract_catalog(canvas, "dataCatalog", "data_catalog")
        credentials = self._extract_catalog(canvas, "credentials", "credentials")
        valid_node_ids = {str(n["id"]) for n in nodes}
        valid_edge_ids = {str(e["id"]) for e in edges}
        valid_lane_ids = {str(l["id"]) for l in lanes}
        valid_data_ids = {str(d["id"]) for d in data_catalog}
        valid_credential_ids = {str(c["id"]) for c in credentials}
        node_to_lane = {str(n["id"]): str(n.get("laneId")) for n in nodes if n.get("laneId")}

        cleaned = self._validate_selections(
            selections,
            valid_node_ids,
            valid_edge_ids,
            valid_lane_ids,
            valid_data_ids,
            valid_credential_ids,
            node_to_lane,
            data_catalog,
            credentials,
        )

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
                "menos una selección con nombre y element_ids."
            )
        return raw

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

    @staticmethod
    def _extract_lanes(canvas: Dict[str, Any]) -> List[Dict[str, Any]]:
        elements = canvas.get("elements") or {}
        lanes = elements.get("lanes") or canvas.get("lanes") or []
        return [l for l in lanes if isinstance(l, dict) and l.get("id")]

    @staticmethod
    def _extract_catalog(canvas: Dict[str, Any], elements_key: str, top_level_key: str) -> List[Dict[str, Any]]:
        elements = canvas.get("elements") or {}
        items = elements.get(elements_key)
        if items is None:
            items = canvas.get(top_level_key) or canvas.get(elements_key) or []
        return [item for item in items if isinstance(item, dict) and item.get("id")]

    # ──────────────────────────── validation (fail-fast)

    def _validate_selections(
        self,
        selections: Dict[str, Any],
        valid_node_ids: Set[str],
        valid_edge_ids: Set[str],
        valid_lane_ids: Set[str],
        valid_data_ids: Set[str],
        valid_credential_ids: Set[str],
        node_to_lane: Dict[str, str],
        data_catalog: List[Dict[str, Any]],
        credentials: List[Dict[str, Any]],
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
                    f"objeto con description y element_ids "
                    f"(recibido: {type(spec).__name__})."
                )

            element_ids = self._extract_element_ids(name, spec)

            if "nodes" not in element_ids:
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' no incluye "
                    "nodos. Usa 'node_ids' legacy o element_ids.nodes."
                )
            node_ids = element_ids["nodes"]
            if not isinstance(node_ids, list):
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' tiene "
                    f"nodos que no son un array (recibido: "
                    f"{type(node_ids).__name__})."
                )
            if len(node_ids) == 0:
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' tiene "
                    "node_ids vacío. Cada selección debe incluir al menos "
                    "un nodo."
                )

            edge_ids = element_ids.get("edges", [])
            if not isinstance(edge_ids, list):
                raise ValueError(
                    f"Llamada rechazada: la selección '{name}' tiene "
                    f"edges que no son un array (recibido: "
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
            lane_ids_str = self._validate_optional_ids(
                name, element_ids, "lanes", valid_lane_ids, "lanes"
            )
            data_ids_str = self._validate_optional_ids(
                name, element_ids, "data", valid_data_ids, "datos"
            )
            credential_ids_str = self._validate_optional_ids(
                name, element_ids, "credentials", valid_credential_ids, "credenciales"
            )

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

            if not lane_ids_str:
                lane_ids_str = sorted(
                    {node_to_lane[nid] for nid in node_ids_str if nid in node_to_lane}
                )

            data_ids_str = self._merge_linked_ids(
                data_ids_str, data_catalog, set(node_ids_str), set(lane_ids_str)
            )
            credential_ids_str = self._merge_linked_ids(
                credential_ids_str, credentials, set(node_ids_str), set(lane_ids_str)
            )

            flow = element_ids.get("flow", {})
            if flow:
                self._validate_flow(name, flow, set(node_ids_str), set(edge_ids_str))

            cleaned[name] = {
                "description": description.strip(),
                "node_ids": node_ids_str,
                "edge_ids": edge_ids_str,
                "element_ids": {
                    "lanes": lane_ids_str,
                    "nodes": node_ids_str,
                    "edges": edge_ids_str,
                    "data": data_ids_str,
                    "credentials": credential_ids_str,
                    "flow": flow if isinstance(flow, dict) else {},
                },
            }

        return cleaned

    @staticmethod
    def _extract_element_ids(name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        element_ids = spec.get("element_ids")
        if element_ids is not None:
            if not isinstance(element_ids, dict):
                raise ValueError(
                    f"Llamada rechazada: element_ids de la selección '{name}' debe ser un objeto."
                )
            return dict(element_ids)

        extracted = {}
        if "node_ids" in spec:
            extracted["nodes"] = spec["node_ids"]
        if "edge_ids" in spec:
            extracted["edges"] = spec["edge_ids"]
        for key in ("lanes", "nodes", "edges", "data", "credentials", "flow"):
            if key in spec:
                extracted[key] = spec[key]
        return extracted

    @staticmethod
    def _validate_optional_ids(
        name: str,
        element_ids: Dict[str, Any],
        key: str,
        valid_ids: Set[str],
        label: str,
    ) -> List[str]:
        values = element_ids.get(key, [])
        if values is None:
            return []
        if not isinstance(values, list):
            raise ValueError(
                f"Llamada rechazada: la selección '{name}' tiene {label} que no son un array."
            )
        values_str = [str(value) for value in values]
        unknown = [value for value in values_str if value not in valid_ids]
        if unknown:
            raise ValueError(
                f"Llamada rechazada: la selección '{name}' referencia {label} inexistentes: {unknown}."
            )
        return values_str

    @staticmethod
    def _merge_linked_ids(
        selected_ids: List[str],
        catalog: List[Dict[str, Any]],
        selected_nodes: Set[str],
        selected_lanes: Set[str],
    ) -> List[str]:
        merged = set(selected_ids)
        for item in catalog:
            for link in item.get("links", []) or []:
                if link.get("nodeId") in selected_nodes or link.get("laneId") in selected_lanes:
                    merged.add(str(item["id"]))
        return sorted(merged)

    def _validate_flow(
        self,
        name: str,
        flow: Dict[str, Any],
        selected_nodes: Set[str],
        selected_edges: Set[str],
    ) -> None:
        if not isinstance(flow, dict):
            raise ValueError(
                f"Llamada rechazada: flow de la selección '{name}' debe ser un objeto."
            )
        order = flow.get("order", [])
        if not isinstance(order, list):
            raise ValueError(
                f"Llamada rechazada: flow.order de la selección '{name}' debe ser un array."
            )

        def walk(items: List[Dict[str, Any]]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Llamada rechazada: flow.order de '{name}' contiene un item inválido."
                    )
                node_id = item.get("nodeId")
                edge_id = item.get("edgeId")
                if node_id and str(node_id) not in selected_nodes:
                    raise ValueError(
                        f"Llamada rechazada: flow de '{name}' referencia nodeId no seleccionado: {node_id}."
                    )
                if edge_id and str(edge_id) not in selected_edges:
                    raise ValueError(
                        f"Llamada rechazada: flow de '{name}' referencia edgeId no seleccionado: {edge_id}."
                    )
                branches = item.get("branches", [])
                if branches:
                    if not isinstance(branches, list):
                        raise ValueError(
                            f"Llamada rechazada: branches de '{name}' debe ser un array."
                        )
                    for branch in branches:
                        if not isinstance(branch, dict):
                            raise ValueError(
                                f"Llamada rechazada: branch de '{name}' debe ser un objeto."
                            )
                        branch_edge = branch.get("edgeId")
                        if branch_edge and str(branch_edge) not in selected_edges:
                            raise ValueError(
                                f"Llamada rechazada: branch de '{name}' referencia edgeId no seleccionado: {branch_edge}."
                            )
                        branch_order = branch.get("order", [])
                        if not isinstance(branch_order, list):
                            raise ValueError(
                                f"Llamada rechazada: branch.order de '{name}' debe ser un array."
                            )
                        walk(branch_order)

        walk(order)

    # ──────────────────────────── selection persistence

    def _persist_selections(
        self,
        canvas_uuid: str,
        selections: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for name, spec in selections.items():
            element_ids = spec["element_ids"]
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
                "element_ids": element_ids,
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
    def _format_created(results: List[Dict[str, Any]]) -> str:
        lines = ["## Selecciones creadas", ""]
        for entry in results:
            name = entry["name"]
            description = entry["description"]
            n_nodes = len(entry["node_ids"])
            n_edges = len(entry["edge_ids"])
            element_ids = entry.get("element_ids", {})
            n_lanes = len(element_ids.get("lanes", []))
            n_data = len(element_ids.get("data", []))
            n_credentials = len(element_ids.get("credentials", []))

            bullet = f"- **{name}**"
            if description:
                bullet += f" — {description}"
            lines.append(bullet)

            if entry.get("selection_uuid"):
                lines.append(
                    f"  (Selección {entry['selection_uuid']} creada, "
                    f"{n_lanes} lanes, {n_nodes} nodos, {n_edges} edges, "
                    f"{n_data} datos, {n_credentials} credenciales.)"
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
