"""Testes unitários para o Agente de Causa-Raiz SMT baseado em LangGraph."""

import pytest

from src.models.smt_graph_agent import SMTGraphAgent


@pytest.fixture
def graph_agent():
    return SMTGraphAgent()


def test_langgraph_defect_short_execution(graph_agent: SMTGraphAgent):
    """Verifica a execução de ponta a ponta do StateGraph para curto-circuito."""
    result = graph_agent.generate_report(
        inference_id="inf-test-01",
        prediction="DEFECT_SHORT",
        confidence=0.94,
        is_unknown_anomaly=False,
    )

    assert "report_markdown" in result
    assert "### 🏭 Laudo Técnico" in result["report_markdown"] or "Diagnóstico" in result["report_markdown"]
    assert result["severity_level"] in ("ALTA", "CRÍTICA")
    assert "execution_trace" in result
    assert len(result["execution_trace"]) == 4
    assert "extract_xai" in result["execution_trace"][0]
    assert "query_knowledge_graph" in result["execution_trace"][1]
    assert "evaluate_risk" in result["execution_trace"][2]
    assert "generate_laudo" in result["execution_trace"][3]


def test_langgraph_unknown_anomaly_quarantine(graph_agent: SMTGraphAgent):
    """Verifica se uma anomalia inédita (Open-Set) é roteada para Quarentena Crítica."""
    result = graph_agent.generate_report(
        inference_id="inf-test-02",
        prediction="UNKNOWN_ANOMALY",
        confidence=0.45,
        is_unknown_anomaly=True,
    )

    assert result["severity_level"] == "CRÍTICA"
    assert "QUARENTENA" in result["routing_decision"]
    assert "IPC-A-610" in result["report_markdown"]


def test_langgraph_normal_board_approval(graph_agent: SMTGraphAgent):
    """Verifica se uma placa normal é liberada para a próxima etapa produtiva."""
    result = graph_agent.generate_report(
        inference_id="inf-test-03",
        prediction="NORMAL",
        confidence=0.98,
        is_unknown_anomaly=False,
    )

    assert result["severity_level"] == "CONFORME"
    assert result["routing_decision"] == "LIBERADO_TESTE_FUNCIONAL"
