"""Gerador de Laudos Técnicos de Causa-Raiz (GenAI Industrial).

Integra com OpenCode Go (DeepSeek V4.1 Flash) via protocolo compatível com OpenAI,
com fallback heurístico determinístico baseado na norma IPC-A-610.
"""

from typing import Any

import structlog
from openai import AsyncOpenAI

from src.core.config import get_settings

logger = structlog.get_logger("positivo_vision.genai")

# Base de conhecimento de causas-raiz industriais e normas IPC-A-610
SMT_KNOWLEDGE_BASE: dict[str, dict[str, str]] = {
    "DEFECT_SHORT": {
        "title": "Curto-Circuito por Ponte de Solda (Solder Bridging)",
        "ipc_clause": "IPC-A-610G Seção 5.2.7.2 - Pontes e Filetes Condutores",
        "root_cause": (
            "Excesso de deposição de pasta de solda na serigrafia (stencil com abertura desgastada "
            "ou lâmina de raspagem desalinhada). Possível temperatura de pré-aquecimento insuficiente no forno de refluxo."
        ),
        "corrective_action": (
            "1. Limpar e inspecionar o stencil com álcool isopropílico (IPA).\n"
            "2. Ajustar a pressão da lâmina da serigráfica em -0.5 bar.\n"
            "3. Na bancada de retrabalho, aplicar fluxo ROL0 e remover o filete de solda com malha de dessoldagem."
        ),
        "rma_risk": "Alto. Curto-circuito em linhas de alimentação (+5V/VCC) causa desligamento imediato ou queima de CIs.",
    },
    "DEFECT_OPEN": {
        "title": "Circuito Aberto / Trilha Rompida (Open Circuit)",
        "ipc_clause": "IPC-A-610G Seção 10.2.1 - Descontinuidade em Trilhas e Pads",
        "root_cause": (
            "Trinca mecânica por flexão excessiva da placa na despanelização ou corrosão localizada "
            "durante a etapa de corrosão química do cobre (etching)."
        ),
        "corrective_action": (
            "1. Inspecionar as facas de corte da despanelizadora mecânica.\n"
            "2. Verificar a concentração química do banho de percloreto de ferro.\n"
            "3. Na bancada, efetuar jumper com fio de cobre envernizado AWG 30 e selar com máscara UV."
        ),
        "rma_risk": "Crítico. Causa falhas intermitentes no cliente final dependendo da dilatação térmica.",
    },
    "DEFECT_MISSING_HOLE": {
        "title": "Furo Ausente em Via de Passagem (Missing Through-Hole)",
        "ipc_clause": "IPC-A-610G Seção 10.3.3 - Furos Passantes e Metalização",
        "root_cause": (
            "Quebra de micro-broca na furadeira CNC multieixos ou entupimento por resíduo de verniz "
            "fotossensível durante a cura da máscara de solda."
        ),
        "corrective_action": (
            "1. Notificar a estação de furação mecânica CNC para recalibração do sensor de broca partida.\n"
            "2. Inspecionar o processo de aplicação de máscara para evitar sangramento nos furos de via."
        ),
        "rma_risk": "Médio-Alto. Impede o contato elétrico entre as camadas interna e externa da PCB.",
    },
    "DEFECT_SPURIOUS": {
        "title": "Cobre / Solda Espúria (Spurious Copper / Solder Splatter)",
        "ipc_clause": "IPC-A-610G Seção 5.2.7.1 - Esferas e Respingos de Solda",
        "root_cause": (
            "Fervura violenta do fluxo por umidade residual na pasta de solda ou taxa de rampa de subida "
            "de temperatura muito rápida na zona 1 do forno de refluxo."
        ),
        "corrective_action": (
            "1. Reduzir a rampa de aquecimento inicial para < 2°C/segundo.\n"
            "2. Verificar o controle de umidade e validade da pasta de solda no armazém SMT.\n"
            "3. Limpar a área com escova antiestática (ESD) e solvente apropriado."
        ),
        "rma_risk": "Médio. Pode se soltar durante o transporte e provocar curtos aleatórios com vibração.",
    },
    "UNKNOWN_ANOMALY": {
        "title": "Anomalia Inédita / Fora de Catálogo Padrão (Quarantine SMT)",
        "ipc_clause": "Critério de Engenharia Positivo - Rejeição por Variação Geométrica",
        "root_cause": (
            "Assinatura visual anômala não compatível com o catálogo supervisionado. "
            "Provável risco mecânico profundo no substrato FR-4, queimadura térmica ou corpo estranho na placa."
        ),
        "corrective_action": (
            "1. Encaminhar lote imediatamente para a Quarentena da Engenharia de Processos.\n"
            "2. Coletar imagens em alta resolução com microscópio óptico para inclusão no próximo re-treinamento."
        ),
        "rma_risk": "Desconhecido / Alto. Requer validação humana obrigatória.",
    },
    "NORMAL": {
        "title": "Placa Conforme - Homologada (Pass)",
        "ipc_clause": "IPC-A-610G Classe 2 - Produtos Eletrônicos Dedicados",
        "root_cause": "Nenhuma não-conformidade detectada. Padrão de solda e integridade de trilhas aprovados.",
        "corrective_action": "Liberar placa para montagem do módulo de hardware e esteira final de testes elétricos.",
        "rma_risk": "Nulo.",
    },
}


async def generate_technical_report(
    prediction: str,
    confidence: float,
    inference_id: str,
    latency_ms: float,
    is_unknown_anomaly: bool = False,
) -> dict[str, Any]:
    """Gera laudo técnico formatado para engenharia SMT via LLM ou fallback IPC."""
    settings = get_settings()
    target_class = "UNKNOWN_ANOMALY" if is_unknown_anomaly else prediction
    kb = SMT_KNOWLEDGE_BASE.get(target_class, SMT_KNOWLEDGE_BASE["NORMAL"])

    # Se a chave do OpenCode Go estiver presente, consulta o LLM assincronamente
    if settings.OPENCODE_API_KEY and settings.OPENCODE_API_KEY.strip():
        try:
            client = AsyncOpenAI(
                base_url=settings.OPENCODE_BASE_URL,
                api_key=settings.OPENCODE_API_KEY,
            )
            prompt = (
                f"Você é o Engenheiro Especialista SMT de Processos e Qualidade da Positivo Tecnologia (POSI3).\n"
                f"Gere um laudo técnico conciso e extremamente profissional para a seguinte não-conformidade fabril:\n"
                f"- ID da Inspeção: {inference_id}\n"
                f"- Diagnóstico Visual: {prediction}\n"
                f"- Confiança do Modelo: {confidence * 100:.1f}%\n"
                f"- Anomalia Inédita: {'Sim' if is_unknown_anomaly else 'Não'}\n"
                f"- Norma de Referência: {kb['ipc_clause']}\n\n"
                f"Estruture em:\n"
                f"1. **Identificação da Não-Conformidade**\n"
                f"2. **Causa-Raiz no Processo SMT** (forno, stencil, pasta)\n"
                f"3. **Ação Corretiva Imediata na Bancada**\n"
                f"4. **Mitigação de Custo / RMA Positivo**\n"
            )

            response = await client.chat.completions.create(
                model=settings.OPENCODE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.2,
            )
            report_text = response.choices[0].message.content or ""
            return {
                "source": "opencode_go",
                "model": settings.OPENCODE_MODEL,
                "report_markdown": report_text,
                "ipc_standard": kb["ipc_clause"],
            }
        except Exception as exc:
            logger.warning("Falha na chamada ao OpenCode Go LLM; utilizando fallback IPC", error=str(exc))

    # Fallback estruturado de alta fidelidade
    report_markdown = (
        f"### 📋 Laudo Técnico de Manufatura SMT — Positivo Tecnologia\n\n"
        f"**ID da Inspeção:** `{inference_id}` | **Status:** `{target_class}` | **Confiança:** `{confidence * 100:.1f}%`  \n"
        f"**Norma de Conformidade:** *{kb['ipc_clause']}*\n\n"
        f"#### 1. Diagnóstico Técnico\n"
        f"{kb['title']}\n\n"
        f"#### 2. Investigação de Causa-Raiz (Processo SMT)\n"
        f"{kb['root_cause']}\n\n"
        f"#### 3. Ação Corretiva para o Técnico de Bancada\n"
        f"{kb['corrective_action']}\n\n"
        f"#### 4. Avaliação de Impacto & Blindagem de RMA\n"
        f"**Risco de Devolução:** {kb['rma_risk']}\n"
        f"*Ação preventiva protege a Positivo contra custos de frete reverso e substituição de placas em garantia.*"
    )

    return {
        "source": "industrial_kb_fallback",
        "model": "SMT-IPC-A-610-Expert",
        "report_markdown": report_markdown,
        "ipc_standard": kb["ipc_clause"],
    }
