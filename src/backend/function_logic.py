"""
Business logic for IdentifyCanvasWorkflowsFn.

Analyzes a canvas to identify isolated business workflows and (optionally)
creates a named CanvasSelection for each one.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional  # noqa: F401

from api.canvas_requests import canvas_api_manager
from chask_foundation.backend.models import OrchestrationEvent
from chask_foundation.llm import LLMClient

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAX_NODE_DATA_CHARS = 400
LLM_MODEL = "gpt-5.2"


SYSTEM_PROMPT = (
    "Eres un analista de procesos que recibe un canvas visual de nodos y "
    "conexiones. Un canvas puede contener uno o varios flujos de negocio "
    "independientes (por ejemplo: envio de cobranza + registro de pago, "
    "o confirmacion + respuesta automatica). Tu tarea es agrupar los nodos "
    "en flujos coherentes y devolver EXCLUSIVAMENTE un objeto JSON con el "
    "formato especificado. No agregues texto fuera del JSON."
)


class FunctionBackend:
    """Backend for IdentifyCanvasWorkflowsFn."""

    def __init__(
        self,
        orchestration_event: OrchestrationEvent,
        openai_api_key: Optional[str] = None,
    ):
        self.orchestration_event = orchestration_event
        self.openai_api_key = openai_api_key
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

        create_selections = tool_args.get("create_selections", True)
        if isinstance(create_selections, str):
            create_selections = create_selections.lower() not in ("false", "0", "no")

        canvas = self._fetch_canvas_detail(canvas_uuid)
        nodes = self._extract_nodes(canvas)
        edges = self._extract_edges(canvas)

        if not nodes:
            return "El canvas no tiene nodos, no hay flujos que identificar."

        workflows = self._identify_workflows(canvas, nodes, edges)
        workflows = self._clean_workflows(workflows, nodes, edges)

        if not workflows:
            return (
                "No se pudo identificar ningun flujo valido a partir del canvas. "
                "Revisa que el canvas tenga nodos y conexiones."
            )

        if len(workflows) == 1:
            return (
                "Este canvas representa un solo flujo coherente. "
                "No se crearon selecciones adicionales."
            )

        if not create_selections:
            return self._format_preview(workflows)

        created = self._persist_selections(canvas_uuid, workflows)
        return self._format_creation_summary(created)

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

    # ──────────────────────────── LLM call

    def _identify_workflows(
        self,
        canvas: Dict[str, Any],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        canvas_title = canvas.get("title") or "Canvas sin titulo"
        canvas_description = canvas.get("description") or ""

        node_lines = [self._summarize_node(n) for n in nodes]
        edge_lines = [self._summarize_edge(e) for e in edges]

        user_message = (
            f"## Canvas\n"
            f"Titulo: {canvas_title}\n"
            f"Descripcion: {canvas_description or '(sin descripcion)'}\n\n"
            f"## Nodos ({len(nodes)})\n"
            + "\n".join(node_lines)
            + "\n\n"
            f"## Conexiones ({len(edges)})\n"
            + ("\n".join(edge_lines) if edge_lines else "(sin conexiones)")
            + "\n\n"
            "## Tarea\n"
            "Agrupa los nodos en flujos de trabajo independientes. "
            "Dos nodos pertenecen al mismo flujo si forman parte del mismo "
            "proceso de negocio (misma cadena de causa y efecto). "
            "Los nodos de instrucciones o contexto compartidos pueden "
            "aparecer en varios flujos.\n\n"
            "Si el canvas es un unico flujo coherente, devuelve un arreglo "
            "con exactamente un elemento.\n\n"
            "Responde ESTRICTAMENTE con un JSON valido de la forma:\n"
            "{\n"
            '  "workflows": [\n'
            "    {\n"
            '      "name": "<nombre corto en espanol, ej. \\"Envio de cobranza\\">",\n'
            '      "description": "<resumen en una frase, en espanol>",\n'
            '      "node_ids": ["<uuid de nodo>", ...]\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        llm_client = LLMClient(
            access_token=self.orchestration_event.access_token,
            organization_id=self.orchestration_event.organization.organization_id,
            orchestration_session_uuid=self.orchestration_event.orchestration_session_uuid,
            internal_orchestration_session_uuid=self.orchestration_event.internal_orchestration_session_uuid,
            orchestration_event_uuid=str(self.orchestration_event.event_id),
            default_model=LLM_MODEL,
            openai_api_key=self.openai_api_key or os.environ.get("OPENAI_API_KEY"),
        )

        try:
            response = llm_client.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=1,
                response_format={"type": "json_object"},
                caller_function="IdentifyCanvasWorkflowsFn.process_request.workflow_grouper",
            )
        finally:
            llm_client.shutdown()

        if not response.get("success"):
            raise RuntimeError(
                f"LLM call failed: {response.get('error', 'Unknown error')}"
            )

        content = response.get("content") or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON: %s", content[:500])
            raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc

        workflows = parsed.get("workflows") or []
        if not isinstance(workflows, list):
            raise RuntimeError(
                "LLM response did not contain a 'workflows' array"
            )
        return workflows

    def _summarize_node(self, node: Dict[str, Any]) -> str:
        node_id = node.get("id", "?")
        node_type = node.get("type") or node.get("node_type") or ""
        data = node.get("data") or {}
        label = (
            data.get("label")
            or data.get("title")
            or node.get("label")
            or node.get("title")
            or ""
        )
        snippet = self._data_snippet(data)
        parts = [f"- [{node_id}] ({node_type or 'sin tipo'})"]
        if label:
            parts[0] += f" {label}"
        if snippet:
            parts.append(f"  datos: {snippet}")
        return "\n".join(parts)

    @staticmethod
    def _summarize_edge(edge: Dict[str, Any]) -> str:
        edge_id = edge.get("id", "?")
        source = edge.get("source") or edge.get("from") or "?"
        target = edge.get("target") or edge.get("to") or "?"
        return f"- [{edge_id}] {source} -> {target}"

    @staticmethod
    def _data_snippet(data: Dict[str, Any]) -> str:
        if not data:
            return ""
        try:
            snippet = json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError):
            snippet = str(data)
        if len(snippet) > MAX_NODE_DATA_CHARS:
            snippet = snippet[:MAX_NODE_DATA_CHARS] + "..."
        return snippet

    # ──────────────────────────── post-processing

    def _clean_workflows(
        self,
        workflows: List[Dict[str, Any]],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        valid_node_ids = {str(n.get("id")) for n in nodes}
        cleaned: List[Dict[str, Any]] = []

        for idx, wf in enumerate(workflows):
            if not isinstance(wf, dict):
                logger.warning("Skipping non-dict workflow at index %d", idx)
                continue
            raw_ids = wf.get("node_ids") or []
            if not isinstance(raw_ids, list):
                logger.warning(
                    "Workflow %s has non-list node_ids, skipping", wf.get("name")
                )
                continue

            kept_nodes: List[str] = []
            seen: set = set()
            for nid in raw_ids:
                nid_str = str(nid)
                if nid_str not in valid_node_ids:
                    logger.warning(
                        "Workflow %s references unknown node_id %s, dropping",
                        wf.get("name"), nid_str,
                    )
                    continue
                if nid_str in seen:
                    continue
                seen.add(nid_str)
                kept_nodes.append(nid_str)

            if not kept_nodes:
                logger.warning(
                    "Workflow %s has no valid node_ids, skipping", wf.get("name")
                )
                continue

            edge_ids = self._derive_edge_ids(kept_nodes, edges)
            name = (wf.get("name") or f"Flujo {idx + 1}").strip()
            description = (wf.get("description") or "").strip()

            cleaned.append({
                "name": name,
                "description": description,
                "node_ids": kept_nodes,
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

    # ──────────────────────────── selection creation

    def _persist_selections(
        self, canvas_uuid: str, workflows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for wf in workflows:
            element_ids = {"nodes": wf["node_ids"], "edges": wf["edge_ids"]}
            # `name` is passed positionally because ApiManager.call(self, name, ...)
            # reserves the first kwarg slot for the endpoint name.
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
                results.append({
                    **wf,
                    "selection_uuid": None,
                    "error": self._api_error_detail(response),
                })
                continue

            results.append({
                **wf,
                "selection_uuid": response.get("uuid"),
            })

        return results

    # ──────────────────────────── response formatters

    @staticmethod
    def _format_preview(workflows: List[Dict[str, Any]]) -> str:
        lines = ["## Flujos identificados (preview)\n"]
        for wf in workflows:
            lines.append(f"**{wf['name']}**")
            if wf["description"]:
                lines.append(wf["description"])
            lines.append(
                f"({len(wf['node_ids'])} nodos, {len(wf['edge_ids'])} conexiones)"
            )
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _format_creation_summary(created: List[Dict[str, Any]]) -> str:
        lines = ["## Flujos identificados\n"]
        for wf in created:
            lines.append(f"**{wf['name']}**")
            if wf["description"]:
                lines.append(wf["description"])

            if wf.get("selection_uuid"):
                lines.append(
                    f"(Seleccion: {wf['selection_uuid']}, "
                    f"{len(wf['node_ids'])} nodos, "
                    f"{len(wf['edge_ids'])} conexiones)"
                )
            else:
                err = wf.get("error") or "error desconocido"
                lines.append(
                    f"(No se pudo crear la seleccion: {err})"
                )
            lines.append("")
        return "\n".join(lines).strip()

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
