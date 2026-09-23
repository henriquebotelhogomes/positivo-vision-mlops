# 🎯 INTERVIEW_PLAYBOOK: Guia de Posicionamento & Roteiro Tático para a Positivo Tecnologia

> **Vaga:** Desenvolvedor de IA Sênior (100% Remoto / CLT)  
> **Empresa:** Positivo Tecnologia (POSI3) via consultoria Rehva Tech  
> **Recrutador Inicial:** Janazio Freitas | **Ponto Focal de CV/Docs:** sophia@rehva.com.br  
> **Acordo Salarial Alinhado:** **R$ 18.000,00 CLT + Benefícios Robustos** (PPR até 1 salário, SulAmérica/Unimed, Swile, Gympass, Previdência Privada, etc.)  
> **Diferencial Competitivo:** Sistema em produção online no **Google Cloud Run** com ONNX Runtime, Grad-CAM, Open-Set Anomaly Detection e Laudo Multimodal GenAI.

---

## 1. O Alinhamento Salarial (Como se portar)

Durante a triagem no LinkedIn, o valor inicial discutido foi de R$ 14.000 CLT, mas após as suas respostas técnicas e apresentação do seu projeto em produção, o próprio recrutador (Janazio) confirmou:
> *"Ah, importante: conseguimos chegar nos 18K CLT + benefícios, sem problemas. :)"*

### Quando o RH perguntar: *"Henrique, qual a sua pretensão salarial atual?"*
* **Sua resposta (firme, serena e natural):**  
  > *"Conforme já validado diretamente com o Janazio na triagem técnica inicial, acertamos o patamar de **R$ 18.000 CLT**, que atende plenamente às minhas expectativas e reflete a senioridade, a autonomia técnica e o impacto do papel."*
* **Atenção:** Não reabra negociação, não diga "entre 14 e 18" e não justifique o valor. Apenas confirme com naturalidade.

---

## 2. Seu Pitch de Apresentação (2 Minutos)

Quando pedirem: *"Henrique, conte um pouco sobre sua trajetória profissional."*

> *"Sou Desenvolvedor de Inteligência Artificial e Engenheiro de Software com foco em levar modelos de Machine Learning, Deep Learning e Visão Computacional para ambientes reais de produção industrial e alta escala.*
> 
> *Minha trajetória combina o ciclo completo de engenharia de dados e modelagem com práticas rigorosas de MLOps: desde a concepção de arquiteturas neurais convolucionais (PyTorch/TensorFlow) até a otimização de grafos para borda fabril com ONNX Runtime, explicabilidade visual com Grad-CAM e empacotamento em microsserviços conteinerizados em nuvem (GCP Cloud Run e Azure).*
> 
> *Recentemente, além de atuar com sistemas de alta disponibilidade e observabilidade avançada, desenvolvi duas soluções de alto impacto:*
> 1. *O **Departamento Médico ML** (utilizando ResNet50 para diagnóstico por imagem em produção hospitalar).*
> 2. *O **Positivo Vision MLOps**, projetado sob medida para as fábricas da Positivo Tecnologia (Curitiba e Manaus): uma plataforma completa de inspeção visual de placas de circuito impresso (PCBs), com inferência de borda em menos de 25 milissegundos via ONNX, governança no MLflow sob a política Champion vs. Challenger, detecção de defeitos inéditos no espaço latente (Open-Set Recognition) e geração automática de laudos técnicos de causa-raiz para a linha SMT utilizando IA Multimodal.*
> 
> *Minha mentalidade é sempre orientada ao valor de negócio, latência na esteira e engenharia de software resiliente."*

---

## 3. Respostas Blindadas para as Perguntas do Tech Lead da Positivo

### Pergunta 1: *"Como você lida com a latência de inferência e restrições de hardware no chão de fábrica (Edge Computing)?"*
* **Sua resposta sênior:**
  > *"Em uma linha de montagem SMT, exigir GPUs dedicadas de alto custo em cada câmera é inviável, e o PyTorch bruto em CPU pode levar mais de 100ms. No **Positivo Vision MLOps**, estruturei o pipeline com exportação automática para **ONNX Runtime** com quantização dinâmica INT8/FP16.*
  > 
  > *Isso reduziu o grafo computacional em mais de 60% e derrubou a latência média de inferência para **menos de 25 milissegundos** em processadores x86 comuns de computadores industriais. O MLflow registra esse benchmark comparativo a cada treino para garantir que nenhum modelo seja promovido sem cumprir a meta de SLA fabril."*

---

### Pergunta 2: *"E se entrar um lote de placas com um defeito totalmente novo que nunca esteve no dataset de treino?"*
* **Sua resposta Staff (Diferencial Absoluto):**
  > *"Esse é o calcanhar de Aquiles dos classificadores convencionais: eles operam em 'Closed-Set' e forçam uma predição em uma das classes conhecidas, muitas vezes rotulando um defeito crítico como 'NORMAL' com 95% de confiança.*
  > 
  > *Para blindar a Positivo contra isso, acoplei ao backbone um **Open-Set Anomaly Head**. Nós extraímos os embeddings latentes e calculamos a distância em relação ao cluster de normalidade da PCB de referência (Golden Sample). Se a imagem apresenta uma assinatura geométrica anômala mas não se encaixa com alta densidade em nenhuma classe conhecida, o sistema aciona a flag **`UNKNOWN_ANOMALY`** e encaminha a peça diretamente para a quarentena da engenharia, impedindo que falhas inéditas passem para o cliente final."*

---

### Pergunta 3: *"Como o operador de bancada ou o técnico de engenharia sabe onde está o defeito identificado pela IA?"*
* **Sua resposta sênior:**
  > *"Classificação caixa-preta não serve para a indústria. Implementei no motor de inferência o **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Quando a rede detecta uma classe anômala (como curto-circuito ou circuito aberto), o sistema calcula os gradientes nas últimas camadas convolucionais e gera um mapa de calor térmico sobreposto na PCB exata.*
  > 
  > *Além disso, integrei um módulo de **IA Multimodal que gera o Laudo Técnico de Causa-Raiz SMT**: ao cruzar as coordenadas do Grad-CAM com a classe predita, o sistema sugere a causa provável de processo (ex.: excesso de pasta de solda no stencil ou pico térmico no forno de refusão) e a ação corretiva recomendada para o técnico."*

---

### Pergunta 4: *"Como você garante a melhoria contínua do modelo após o deploy?"*
* **Sua resposta sênior:**
  > *"Implementei uma esteira de **Human-in-the-Loop (Active Learning)**. Na tela de inspeção, o operador pode validar ou corrigir um diagnóstico duvidoso. Esse feedback é gravado assincronamente em background em arquivos colunares **`.parquet`** com a flag `requires_retrain=True`.*
  > 
  > *No MLflow, adotamos a política estrita de **Champion vs. Challenger**: novas rodadas de re-treinamento só recebem o alias de produção (`@champion`) após superar o F1-score ponderado em dados de teste out-of-sample, garantindo um ciclo virtuoso de aprendizado sem risco de regressão."*

---

### Pergunta 5: *"Como você estruturou a conteinerização, o deploy em nuvem e a proteção de custos (FinOps)?"*
* **Sua resposta sênior:**
  > *"Construí o projeto com Dockerfile multi-stage, separando as ferramentas de build das dependências de runtime e executando tudo sob um usuário dedicado sem privilégios (`appuser`).*
  > 
  > *O projeto está **publicado online no Google Cloud Run**, utilizando arquitetura serverless com política de **Scale-to-Zero**, o que garante custo zero quando ocioso e cold-start abaixo de 4 segundos graças ao ONNX Runtime.*
  > 
  > *Além disso, apliquei guardrails rigorosos de **FinOps**: implementei um middleware de **Rate Limiting por IP (25 testes a cada 2 horas)** para proteger nossa cota de tokens do modelo **DeepSeek V4.1 Flash (OpenCode Go)** contra abusos ou scrapers na demonstração pública, com bypass key exclusiva para a entrevista. A API expõe endpoints de saúde (`/healthz` e `/ready`) prontos para Kubernetes e a documentação interativa é servida via Scalar OpenAPI."*

---

## 4. O Roteiro da Demonstração ao Vivo (Live Demo Script)

Se na entrevista técnica você puder compartilhar a tela (ou enviar o link público para eles acessarem no celular):

1. **Abertura do Sistema:**
   * Abra a interface da Live Demo (`/demo` no Cloud Run).
2. **Apresentação em 30 Segundos:**
   > *"Esta interface está conectada diretamente ao nosso microsserviço assíncrono em FastAPI, consumindo o modelo homologado com o alias `@champion` no MLflow Model Registry e acelerado por ONNX Runtime."*
3. **O Botão Mágico (Amostra Aleatória de Esteira):**
   * Clique em: **"🎲 Simular Leitura da Câmera na Esteira (Amostra Aleatória)"**.
   * Destaque:
     > *"Ao clicar aqui, simulamos o disparo de uma câmera industrial de esteira. Notem que em apenas **18 milissegundos**, o modelo classificou a peça como `REPROVADA: CURTO-CIRCUITO` com 98% de confiança."*
4. **O Efeito Grad-CAM (O Diferencial Visual):**
   * Alterne a chave de visualização: **"Ver Mapa de Calor Grad-CAM"**.
   * Explique:
     > *"E aqui está o mapa de ativação Grad-CAM: ele destaca em vermelho a solda exata que causou o curto-circuito entre as trilhas condutoras."*
5. **O Laudo Técnico GenAI (O Efeito Uau):**
   * Clique no botão: **"📄 Gerar Laudo Técnico SMT"**.
   * Mostre o modal gerado:
     > *"O sistema analisa a falha e entrega ao engenheiro de processo a causa-raiz provável (excesso de pasta no stencil) e a ação corretiva imediata."*
6. **O Feedback HITL:**
   * Clique em **"✅ Validar Diagnóstico"**:
     > *"O operador confirma o diagnóstico na esteira e isso alimenta imediatamente nosso stream Parquet para a governança de retreinamento contínuo."*

---

## 5. Perguntas Estratégicas para VOCÊ fazer no final da entrevista

1. *"Na linha de montagem da Positivo em Curitiba e Manaus, as câmeras de esteira hoje enviam imagens para servidores industriais on-premise na borda ou vocês já possuem algum pipeline híbrido conectando com Azure/GCP?"*
2. *"Como o time de manufatura lida hoje com a detecção de anomalias inéditas quando entram novos SKUs de placas em produção?"*
3. *"Quais são os principais desafios técnicos de MLOps que a liderança de engenharia de IA planeja endereçar nos primeiros 90 dias dessa contratação?"*
