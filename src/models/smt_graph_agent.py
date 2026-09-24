"""Agente de Causa-Raiz SMT com LangGraph & Knowledge Graph de Engenharia de Processos.

Implementa um grafo de estados direcionado (StateGraph) utilizando LangGraph para orquestrar:
1. Extração de telemetria visual e coordenadas XAI (Grad-CAM).
2. Consulta determinística a um Grafo de Conhecimento de Processos SMT (norma IPC-A-610 Classe 3).
3. Roteamento condicional de severidade e risco fabril.
4. Síntese do Laudo Técnico Industrial com LLM (DeepSeek via OpenCode Go ou Fallback Estruturado).
"""

from typing import Any, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from src.core.config import get_settings

logger = structlog.get_logger("positivo_vision.graph_agent")


# Base de Conhecimento Estruturada de Processos SMT (Norma IPC-A-610 Classe 3)
SMT_KNOWLEDGE_GRAPH: dict[str, dict[str, Any]] = {
    "DEFECT_SHORT": {
        "critical_station": "Impressora de Pasta (Stencil) & Forno de Refluxo (Zonas 4 a 6)",
        "suspected_root_cause": "Excesso de deposição de pasta de solda SAC305 (> 130% do nominal) ou escorrimento térmico por perfil de pico inadequado (> 245°C por tempo excessivo).",
        "recommended_action": "1. Reduzir a espessura do stencil para 100µm ou aplicar redução de abertura de 10% nos pads finos.\n2. Aumentar a frequência de limpeza automática com vácuo na esteira de stencil para cada 3 ciclos.\n3. Ajustar o tempo acima de liquidus (TAL) no forno para 45-60 segundos.",
        "ipc_standard": "IPC-A-610 Class 3 - Seção 5.2.7.2 (Solder Bridging / Short Violation).",
        "default_severity": "ALTA",
    },
    "DEFECT_OPEN": {
        "critical_station": "Impressora de Pasta & Deposição de Fluxo",
        "suspected_root_cause": "Abertura de stencil entupida por pasta ressecada, pressão insuficiente do squeegee (< 0.2 kg/cm) ou desalinhamento mecânico no Pick & Place.",
        "recommended_action": "1. Realizar limpeza manual ultrassônica no estêncil com solvente aprovado.\n2. Verificar a viscosidade da pasta (deve estar entre 180 e 210 Pa.s a 25°C).\n3. Inspecionar coplanaridade dos terminais do componente antes da soldagem.",
        "ipc_standard": "IPC-A-610 Class 3 - Seção 5.2.7.1 (Non-Wetting / Open Circuit).",
        "default_severity": "CRÍTICA",
    },
    "DEFECT_MISSING_HOLE": {
        "critical_station": "Furação Mecânica CNC & Fotolitografia de Máscara",
        "suspected_root_cause": "Quebra de broca diamantada na máquina de furação CNC (diâmetro < 0.3mm) ou obstrução por resíduo de máscara de solda (solder mask plugging).",
        "recommended_action": "1. Paralisar a etapa de montagem do lote afetado para inspeção dos lotes do fornecedor de PCB.\n2. Checar sensor acústico de quebra de broca na estação CNC.\n3. Rejeitar painel antes do estágio SMT para evitar desperdício de componentes de valor.",
        "ipc_standard": "IPC-A-610 Class 3 - Seção 10.2 (Through-hole / Via Hole Obstruction).",
        "default_severity": "CRÍTICA",
    },
    "DEFECT_MOUSEBITE": {
        "critical_station": "Fresagem e Destacamento de Painel (Depaneling)",
        "suspected_root_cause": "Desgaste excessivo da fresa diamantada no corte de separação ou vibração mecânica descalibrada no destacamento dos tabs de sustentação da placa.",
        "recommended_action": "1. Substituir a fresa de corte da router de depaneling (vida útil máxima 500 metros lineares).\n2. Ajustar velocidade de avanço do spindle para diminuir a tração nas bordas.\n3. Inspecionar integridade da camada de terra (GND) adjacente.",
        "ipc_standard": "IPC-A-610 Class 3 - Seção 10.3 (Edge Delamination & Trace Nibbling).",
        "default_severity": "MODERADA",
    },
    "DEFECT_SPUR": {
        "critical_station": "Fotogravação Química e Banho de Corrosão de Cobre",
        "suspected_root_cause": "Microfissura no filme de photoresist ou sub-corrosão na borda da trilha devido a velocidade irregular da esteira de spray de cloreto cúprico.",
        "recommended_action": "1. Monitorar densidade e potencial redox (ORP) do banho químico de corrosão.\n2. Inspecionar alinhamento ótico na exposição UV da máscara fotoquímica.\n3. Realizar retrabalho manual com lâmina de precisão sob microscópio para remoção da rebarba.",
        "ipc_standard": "IPC-A-610 Class 3 - Seção 10.2.3 (Conductor Spacing Reduction).",
        "default_severity": "MODERADA",
    },
    "DEFECT_SPURIOUS_COPPER": {
        "critical_station": "Lavagem Pós-Gravação e Decapagem",
        "suspected_root_cause": "Remoção incompleta de cobre por tempo insuficiente de ataque químico ou resíduo metálico de suspensão redepositado no substrato FR4.",
        "recommended_action": "1. Efetuar limpeza e substituição dos bicos de aspersão da câmara de gravação química.\n2. Aumentar a vazão de enxágue de água desionizada após a corrosão.\n3. Isolar ilhas espúrias via fresa manual caso não afetem a capacitância parasita.",
        "ipc_standard": "IPC-A-610 Class 3 - Seção 10.2.1 (Extraneous Copper / Foreign Metal).",
        "default_severity": "MODERADA",
    },
    "NORMAL": {
        "critical_station": "Linha SMT Completa (Homologada)",
        "suspected_root_cause": "Nenhuma anomalia detectada. Placa em perfeita conformidade geométrica e funcional.",
        "recommended_action": "Liberar painel para a estação de Teste In-Circuit (ICT) e montagem mecânica final no chassi do equipamento Positivo.",
        "ipc_standard": "IPC-A-610 Class 3 - Conforme (Golden Unit Reference).",
        "default_severity": "CONFORME",
    },
    "UNKNOWN_ANOMALY": {
        "critical_station": "Quarentena de Engenharia de Qualidade Industrial",
        "suspected_root_cause": "Vetor latente fora do domínio estatístico homologado. Possível contaminação por corpo estranho, oxidação severa, pad queimado ou novo lote de PCB fora de especificação.",
        "recommended_action": "1. Reter a placa imediatamente na esteira para inspeção manual sob microscópio 40x.\n2. Realizar análise por Raio-X (AXI) para checagem de camadas internas.\n3. Caso confirmada falha de montagem, acionar engenharia de qualidade para re-homologação.",
        "ipc_standard": "IPC-A-610 Class 3 - Defeito Inédito / Fora de Especificação.",
        "default_severity": "CRÍTICA",
    },
}


class SMTState(TypedDict):
    """Estado tipado que transita entre os nós do LangGraph."""

    inference_id: str
    prediction: str
    confidence: float
    is_unknown_anomaly: bool
    gradcam_bbox: dict[str, Any] | None
    knowledge_record: dict[str, Any]
    severity_level: str
    routing_decision: str
    report_markdown: str
    model_source: str
    execution_trace: list[str]


def node_extract_xai(state: SMTState) -> dict[str, Any]:
    """Nó 1: Analisa os metadados visuais, confiança e bounding box do Grad-CAM."""
    pred = state["prediction"]
    conf = state["confidence"]
    is_unk = state.get("is_unknown_anomaly", False)

    trace = list(state.get("execution_trace", []))
    trace.append(f"extract_xai[pred={pred}, conf={conf:.2f}, unk={is_unk}]")

    return {
        "execution_trace": trace,
    }


def node_query_knowledge_graph(state: SMTState) -> dict[str, Any]:
    """Nó 2: Consulta determinística ao Knowledge Graph de regras IPC-A-610 e SMT."""
    pred = state["prediction"]
    is_unk = state.get("is_unknown_anomaly", False)

    if is_unk:
        rule_key = "UNKNOWN_ANOMALY"
    elif pred in SMT_KNOWLEDGE_GRAPH:
        rule_key = pred
    elif "SHORT" in pred:
        rule_key = "DEFECT_SHORT"
    elif "OPEN" in pred:
        rule_key = "DEFECT_OPEN"
    elif "HOLE" in pred:
        rule_key = "DEFECT_MISSING_HOLE"
    elif "MOUSEBITE" in pred:
        rule_key = "DEFECT_MOUSEBITE"
    elif "SPURIOUS" in pred:
        rule_key = "DEFECT_SPURIOUS_COPPER"
    elif "SPUR" in pred:
        rule_key = "DEFECT_SPUR"
    else:
        rule_key = "NORMAL"

    knowledge = SMT_KNOWLEDGE_GRAPH.get(rule_key, SMT_KNOWLEDGE_GRAPH["UNKNOWN_ANOMALY"])
    trace = list(state.get("execution_trace", []))
    trace.append(f"query_knowledge_graph[rule={rule_key}]")

    return {
        "knowledge_record": knowledge,
        "execution_trace": trace,
    }


def node_evaluate_risk(state: SMTState) -> dict[str, Any]:
    """Nó 3: Avalia o risco e define o roteamento de intervenção fabril."""
    conf = state["confidence"]
    is_unk = state.get("is_unknown_anomaly", False)
    knowledge = state["knowledge_record"]

    if is_unk or conf < 0.60:
        severity = "CRÍTICA"
        decision = "QUARENTENA_ENGENHARIA_IMEDIATA"
    elif state["prediction"] == "NORMAL":
        severity = "CONFORME"
        decision = "LIBERADO_TESTE_FUNCIONAL"
    elif knowledge["default_severity"] == "CRÍTICA":
        severity = "CRÍTICA"
        decision = "INTERVENCAO_ESTENCOL_CNC"
    else:
        severity = knowledge["default_severity"]
        decision = "INTERVENCAO_FORNO_STENCIL"

    trace = list(state.get("execution_trace", []))
    trace.append(f"evaluate_risk[severity={severity}, route={decision}]")

    return {
        "severity_level": severity,
        "routing_decision": decision,
        "execution_trace": trace,
    }


def node_generate_laudo_llm(state: SMTState) -> dict[str, Any]:
    """Nó 4: Sintetiza o laudo técnico utilizando o LLM ou fallback estruturado."""
    settings = get_settings()
    pred = state["prediction"]
    conf = state["confidence"]
    knowledge = state["knowledge_record"]
    severity = state["severity_level"]
    decision = state["routing_decision"]
    inf_id = state["inference_id"]

    # Tentativa de chamada ao LLM DeepSeek via OpenCode Go se chave configurada
    if settings.OPENCODE_API_KEY and settings.OPENCODE_API_KEY.strip():
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=settings.OPENCODE_BASE_URL,
                api_key=settings.OPENCODE_API_KEY,
            )

            prompt = f"""Você é o Engenheiro Especialista em Processos SMT da Positivo Tecnologia (Curitiba/Manaus).
Gere um Laudo Técnico de Causa-Raiz e Ação Corretiva para a esteira fabril com base nas seguintes evidências industriais:
- ID da Inspeção: {inf_id}
- Diagnóstico da IA: {pred} (Confiança: {conf*100:.1f}%)
- Severidade Fabril: {severity}
- Roteamento de Intervenção: {decision}
- Norma de Referência: {knowledge['ipc_standard']}
- Estação Crítica Mapeada: {knowledge['critical_station']}
- Causa-Raiz Preliminar: {knowledge['suspected_root_cause']}
- Ação Corretiva Recomendada: {knowledge['recommended_action']}

Formate seu laudo de forma executiva e profissional em Markdown, com as seções:
### 1. Diagnóstico de Linha SMT
### 2. Análise de Causa-Raiz (Processo & Equipamentos)
### 3. Ações Corretivas Recomendadas
### 4. Roteamento de Qualidade & SLA
"""

            response = client.chat.completions.create(
                model=settings.OPENCODE_MODEL,
                messages=[
                    {"role": "system", "content": "Você é um Engenheiro de Processos SMT Sênior da Positivo Tecnologia."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            report_text = response.choices[0].message.content or ""
            model_source = f"{settings.OPENCODE_MODEL} (LangGraph via OpenCode Go)"
        except Exception as exc:
            logger.warning("Falha na chamada LLM externa, gerando síntese via LangGraph Knowledge Engine", error=str(exc))
            report_text = _build_structured_fallback_report(state)
            model_source = "LangGraph SMT Knowledge Engine (Deterministic Fallback)"
    else:
        report_text = _build_structured_fallback_report(state)
        model_source = "LangGraph SMT Knowledge Engine (Deterministic Fallback)"

    trace = list(state.get("execution_trace", []))
    trace.append("generate_laudo[complete]")

    return {
        "report_markdown": report_text,
        "model_source": model_source,
        "execution_trace": trace,
    }


def _build_structured_fallback_report(state: SMTState) -> str:
    """Gera laudo técnico determinístico rico estruturado pelo LangGraph."""
    pred = state["prediction"]
    conf = state["confidence"]
    knowledge = state["knowledge_record"]
    severity = state["severity_level"]
    decision = state["routing_decision"]
    inf_id = state["inference_id"]

    return f"""### 🏭 Laudo Técnico de Processos SMT — Positivo Tecnologia
**Identificador da Inspeção:** `{inf_id}`
**Norma Aplicada:** {knowledge['ipc_standard']}
**Severidade Fabril:** `{severity}` | **Roteamento:** `{decision}`

---

#### 1. Diagnóstico Óptico Automatizado (AOI)
* **Classificação:** **{pred}**
* **Índice de Certeza:** {conf * 100:.1f}%
* **Estação Crítica:** {knowledge['critical_station']}

#### 2. Investigação de Causa-Raiz de Processo
{knowledge['suspected_root_cause']}

#### 3. Plano de Ação Imediato na Linha SMT
{knowledge['recommended_action']}

#### 4. Decisão de Roteamento de Qualidade
* **Direcionamento da Placa:** `{decision}`
* **Risco de Devolução (RMA):** {'Alto' if severity in ('CRÍTICA', 'ALTA') else 'Baixo / Nulo'}
* **Governança:** Gravação de evidências no stream Parquet com validação Human-in-the-Loop na bancada de retrabalho.
"""


class SMTGraphAgent:
    """Orquestrador de IA Generativa & Decisão Industrial baseado em LangGraph."""

    def __init__(self) -> None:
        self.workflow = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(SMTState)

        # Adiciona os nós
        graph.add_node("extract_xai", node_extract_xai)
        graph.add_node("query_knowledge_graph", node_query_knowledge_graph)
        graph.add_node("evaluate_risk", node_evaluate_risk)
        graph.add_node("generate_laudo", node_generate_laudo_llm)

        # Conecta as bordas do pipeline
        graph.add_edge(START, "extract_xai")
        graph.add_edge("extract_xai", "query_knowledge_graph")
        graph.add_edge("query_knowledge_graph", "evaluate_risk")
        graph.add_edge("evaluate_risk", "generate_laudo")
        graph.add_edge("generate_laudo", END)

        return graph.compile()

    def generate_report(
        self,
        inference_id: str,
        prediction: str,
        confidence: float,
        is_unknown_anomaly: bool = False,
        gradcam_bbox: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Executa o grafo de ponta a ponta e retorna o laudo e rastro de execução."""
        initial_state: SMTState = {
            "inference_id": inference_id,
            "prediction": prediction,
            "confidence": confidence,
            "is_unknown_anomaly": is_unknown_anomaly,
            "gradcam_bbox": gradcam_bbox,
            "knowledge_record": {},
            "severity_level": "MODERADA",
            "routing_decision": "EM_AVALIACAO",
            "report_markdown": "",
            "model_source": "",
            "execution_trace": [],
        }

        final_state = self.workflow.invoke(initial_state)
        return {
            "report_markdown": final_state["report_markdown"],
            "model_source": final_state["model_source"],
            "severity_level": final_state["severity_level"],
            "routing_decision": final_state["routing_decision"],
            "execution_trace": final_state["execution_trace"],
        }
