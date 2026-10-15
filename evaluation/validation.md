# Validação executada — versão 2 — 15/10/2026

Ambiente real: Windows, Python 3.12, CPU; dependências principais fixadas e ambiente transitivo em `requirements-lock.txt`. Ferramentas de vídeo são opcionais, em `requirements-media.txt`. Este registro distingue execução real, testes controlados e limitações.

| Verificação | Resultado confirmado |
|---|---|
| Suíte automatizada | **89 testes passaram**, `pytest -q -p no:cacheprovider --basetemp=<pasta gravável nova>`, em 3,19 s |
| Dependências | `pip check`: nenhuma dependência quebrada |
| Sintaxe | `compileall` de `voc/`, `scripts/` e `app.py` concluído |
| Proveniência | SHA-256 dos cinco CSVs oficiais em `data-provenance.json` e notebook |
| Corpus completo | 99.224 linhas; 56.537 sem texto; 42.687 documentos; 42.370 review_ids textuais únicos; 307 reviews com múltiplos pedidos |
| Índice | Schema 2, 42.687×384, MiniLM com camadas lineares int8 e vetores float32 normalizados |
| Atualização de metadados | IDs/chunks/textos/ordem idênticos; bytes dos vetores preservados |
| Notebook | 9 células Python executadas sem erro, incluindo checagem do índice schema 2 |
| Recuperação real | Comparação pareada com mesmas perguntas/filtros/vetores: 16 desenvolvimento e 8 reservadas |
| Cenários reais | **19/19 aprovados**: consultas, filtros, analytics, assunto não coberto, intervalo entre meses e abstenções em `scenarios.json` |
| Interface | Streamlit AppTest: renderização, busca real, analytics no corpus completo, filtros em português e recusa passaram |
| LLM local | Nova inferência completa aceita: Qwen2.5-0.5B-Instruct, greedy, sem quantização; um achado sobre atraso com fonte e trecho literal |
| Vídeo | 12 telas, H.264/AAC 1280×720; **5min16,76s**, decodificação integral FFmpeg sem erro e revisão visual das telas |

## O que a comparação demonstra

| Precisão silver@6 | Legado denso+MMR | Híbrido RRF padrão |
|---|---:|---:|
| Desenvolvimento, 16 perguntas | 58,33% | 67,71% |
| Reservadas, 8 perguntas | 77,08% | 81,25% |

No conjunto reservado, o MRR cai de 0,9375 para 0,9167. Há regressões em consultas específicas e ganhos em outras. A seleção do padrão sem MMR ocorreu no desenvolvimento, antes das perguntas reservadas. As regex de relevância silver, IDs, textos e tempos estão nos JSONs/CSVs dos benchmarks. O proxy pode favorecer BM25 e não substitui julgamento humano. Não se comparam esses percentuais com métricas de outro repositório, nem se chama esse resultado de precisão geral do RAG.

`results.json` contém o smoke test atualizado das quatro perguntas temáticas e duas recusas. Seu denominador é o número retornado; o benchmark divide pelo k solicitado. As medidas não são intercambiáveis.

## Evidência da geração

`llm-demo.json` preserva prompts, revisão do modelo, configuração, manifesto, fontes, saída bruta e resultado. A execução final levou **51,97 s** incluindo carregamento do LLM. O modelo selecionou `atraso | E1 | Gostaria de esclarecimentos acerca da entrega que está em atraso.`. A fonte não tinha o ponto final. O verificador removeu somente essa pontuação terminal, exigiu novamente substring literal e registrou `quote_normalization=removed_terminal_punctuation`. Nenhuma palavra ou número foi alterado.

O achado cita o review `7c4c2d10a57d4aab58a570a5a8ac9f2c`, pedido `e269b2472c0546324acf4d09dd7acfab`. A aplicação usa rótulo neutro e o trecho validado; não publica prosa factual livre do modelo. É um fluxo gerativo restrito, próximo de uma síntese extrativa; não comprova qualidade avançada nem desempenho geral em várias perguntas.

As tentativas rejeitadas estão preservadas: JSON inválido do modelo pequeno, repetição no modelo 1,5B quantizado e as duas primeiras tentativas do contrato v2. O teste 1,5B sem quantização foi interrompido pela carga de memória/I/O, sem conclusão sobre sua qualidade. Falhas permanecem identificadas como falhas. O endpoint HTTP tem testes controlados; não há declaração de serviço remoto/Ollama real executado.

## Regressões cobertas

Os testes verificam identidade, ausência de texto/datas, múltiplos pedidos e notas conflitantes; filtros conjuntivos; intervalo entre meses e rejeição de ambiguidades; escopo e resultados vazios; documentos e textos únicos; adulteração de arquivos do índice; orçamento de contexto; números/causas inventados, citações desconhecidas, fonte trocada, trecho alterado, recusa contraditória e falha de API.

Na análise agregada, verificam denominador/unicidade, sobreposição de temas, percentuais em grupos pequenos e troca de contagens/percentuais/rótulos/exemplos. Perguntas de atendimento restringem os temas àquele assunto; foco não coberto, como cashback, provoca abstenção, sem substituir por ranking de outros temas. Categorias informadas como filtros não são tratadas como assuntos analíticos desconhecidos.

## Limites práticos

- Testes com encoders/reranker controlados verificam implementação, não qualidade de modelos reais.
- O limiar 0,30, o mínimo de duas evidências e a base analítica mínima de 30 são políticas heurísticas.
- Não há auditoria humana independente, avaliação de entailment universal, recall do corpus completo ou cálculo de incerteza estatística.
- Temas lexicais podem errar por ironia, negação, variantes e vocabulário. Cobertura de regras não é recall.
- AppTest verifica comportamento da interface; não se alega captura de navegador ou avaliação visual completa da aplicação por automação.
- Docker, instalação limpa em outro computador, servidor HTTP real, reranker real e teste de carga não foram executados.
- A submissão acadêmica e o bônus Databricks não são realizados nem comprovados pelo código.

As revisões padrão dos modelos são fixadas no código: MiniLM `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, Qwen 0,5B `7ae557604adf67be50417f59c2c2f167def9a775`. Cada novo experimento deve preservar dados, configuração e resultados próprios.
