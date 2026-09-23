"""Middleware e Guardrail de Rate Limiting em Memória (FinOps).

Controla o volume de requisições por IP na demonstração pública:
- Inferência de Esteira: 25 requisições a cada 2 horas por IP.
- Laudos Técnicos LLM: 10 requisições a cada 2 horas por IP.
- Chave de contornamento (Bypass Key) para entrevista sem restrições.
"""

import time
from collections import defaultdict, deque

import structlog
from fastapi import HTTPException, Request, status

from src.core.config import get_settings

logger = structlog.get_logger("positivo_vision.ratelimit")


class InMemoryRateLimiter:
    """Implementa controle de taxa por janela deslizante estritamente em memória."""

    def __init__(self) -> None:
        self.settings = get_settings()
        # Mapeia ip -> deque de timestamps
        self.inference_hits: dict[str, deque[float]] = defaultdict(deque)
        self.report_hits: dict[str, deque[float]] = defaultdict(deque)

        # 2 horas em segundos
        self.window_seconds = 2 * 3600
        self.max_inference_requests = 25
        self.max_report_requests = 10

    def _get_client_ip(self, request: Request) -> str:
        """Extrai o IP real do visitante considerando proxies reversos do Cloud Run."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = request.client
        return client.host if client else "127.0.0.1"

    def _check_admin_bypass(self, request: Request) -> bool:
        """Verifica se a requisição possui a chave mestra de administrador."""
        bypass_header = request.headers.get("X-Admin-Bypass")
        bypass_query = request.query_params.get("bypass_key")
        expected_key = self.settings.ADMIN_BYPASS_KEY

        if expected_key and (bypass_header == expected_key or bypass_query == expected_key):
            return True
        return False

    def check_inference_limit(self, request: Request) -> dict[str, int]:
        """Aplica o rate limit na rota de inferência."""
        if self._check_admin_bypass(request):
            return {"remaining": 999, "limit": self.max_inference_requests}

        ip = self._get_client_ip(request)
        now = time.time()
        queue = self.inference_hits[ip]

        # Limpa timestamps mais antigos que 2 horas
        while queue and (now - queue[0] > self.window_seconds):
            queue.popleft()

        if len(queue) >= self.max_inference_requests:
            retry_after = int(self.window_seconds - (now - queue[0])) + 1
            logger.warning("Rate limit de inferencia atingido", ip=ip, retry_after=retry_after)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "status": "rate_limited",
                    "message": "Limite de testes na demonstração pública atingido (25 requisições a cada 2 horas).",
                    "retry_after_seconds": retry_after,
                    "tip": "Para avaliações técnicas da Positivo Tecnologia, utilize a chave de demonstração.",
                },
                headers={"Retry-After": str(retry_after)},
            )

        queue.append(now)
        remaining = self.max_inference_requests - len(queue)
        return {"remaining": remaining, "limit": self.max_inference_requests}

    def check_report_limit(self, request: Request) -> dict[str, int]:
        """Aplica o rate limit na rota de geração de laudos LLM."""
        if self._check_admin_bypass(request):
            return {"remaining": 999, "limit": self.max_report_requests}

        ip = self._get_client_ip(request)
        now = time.time()
        queue = self.report_hits[ip]

        while queue and (now - queue[0] > self.window_seconds):
            queue.popleft()

        if len(queue) >= self.max_report_requests:
            retry_after = int(self.window_seconds - (now - queue[0])) + 1
            logger.warning("Rate limit de laudos GenAI atingido", ip=ip, retry_after=retry_after)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "status": "rate_limited",
                    "message": "Limite de laudos de auditoria SMT atingido (10 laudos a cada 2 horas).",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        queue.append(now)
        remaining = self.max_report_requests - len(queue)
        return {"remaining": remaining, "limit": self.max_report_requests}


rate_limiter = InMemoryRateLimiter()
