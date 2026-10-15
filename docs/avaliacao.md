# Avaliação: resultados, unidades e limites

A avaliação separa **correção da implementação**, **recuperação de evidências**, **análise agregada** e **geração por LLM**. Uma métrica lexical de busca não é uma nota do RAG inteiro. Um teste de contrato com resposta simulada não comprova uma execução real de modelo, e uma referência existente não demonstra, sozinha, apoio semântico.

Os comandos e as verificações efetivamente concluídos ficam em `evaluation/validation.md`. O mapa dos requisitos do enunciado está em `docs/criterios-aceite.md`. Este relatório não apresenta anotação ou auditoria humana como se tivesse sido realizada.

## 1. Corpus e unidades de contagem

| Unidade | Quantidade no corpus usado | Uso |
|---|---:|---|
| Linhas do CSV de reviews | 99.224 | Ponto de partida da ingestão |
| Linhas sem título e mensagem | 56.537 | Excluídas do corpus textual, sem imputação |
| Documentos preparados | 42.687 | Identidade review + pedido + texto; preservam vínculos |
| Fragmentos indexados nesta execução | 42.687 | Linhas da matriz vetorial |
| `review_id` textuais distintos | 42.370 | Unidade da análise de frequência |
| Reviews textuais ligados a múltiplos pedidos | 307 | Contam uma vez em analytics; todos os vínculos são preservados |

O notebook `notebooks/01_exploracao_e_decisoes.ipynb` contém a exploração executada, hashes de cinco tabelas, ausentes, notas, vínculos, estados de entrega e conferência estrutural do índice. Ele não executa benchmark de relevância nem revisão humana.

O índice final é esquema 2, MiniLM multilíngue, 384 dimensões, com encoder de camadas lineares quantizadas em int8 e vetores float32 normalizados. Os hashes de corpus, documentos e vetores estão no manifesto e nos relatórios de benchmark. Trocar modelo, variante, corpus ou filtros muda a condição do experimento.

## 2. Recuperação: comparação pareada real

O script `scripts/evaluate_retrieval.py` executa consultas reais contra os mesmos vetores, perguntas e filtros em quatro configurações:

- `legacy_dense_mmr`: reprodução do ranking denso + MMR do commit inicial `a39930b`, usando os filtros atuais para manter a comparação de ranking.
- `dense_dedup_mmr`: caminho denso novo, com deduplicação de texto e MMR.
- `hybrid_rrf`: busca densa + BM25 com RRF e sem MMR; é o padrão final.
- `hybrid_rrf_mmr`: busca híbrida com MMR.

São **16 perguntas de desenvolvimento** e **8 perguntas holdout**. O padrão `hybrid_rrf` foi escolhido no desenvolvimento e mantido no holdout, sem ajustar a configuração depois de observar suas respostas. Os arquivos de perguntas e seus hashes são registrados em `retrieval-benchmark.json` e `retrieval-holdout.json`. Holdout aqui significa perguntas reservadas para essa execução; não implica elaboração ou anotação humana independente.

### Como as métricas são calculadas

Cada pergunta publica uma regex que identifica sinais textuais do assunto. O rótulo resultante é chamado de **silver**: é automático e heurístico, não um julgamento humano.

- **Precisão silver@6:** número de evidências que satisfazem a regex dividido por 6, mesmo quando a busca retorna menos resultados.
- **MRR silver:** média do inverso da posição da primeira evidência que satisfaz a regex; zero quando nenhuma satisfaz.
- **Textos únicos:** proporção de textos normalizados distintos entre as evidências retornadas por pergunta. Não mede diversidade semântica nem cobertura de temas.
- **Latência:** tempo de uma passagem por consulta após aquecimento do encoder, incluindo codificação, filtros e ranking. A inicialização do processo/modelo é registrada separadamente e não entra na mediana.

Os resultados seguintes vêm dos JSONs versionados, com `k=6`, cosseno mínimo 0,30, RRF `k=60` e MMR `lambda=0.8` quando habilitado. O pool legado é 48 candidatos; o novo é 100. Assim, comparam-se configurações completas: a diferença não isola exclusivamente o efeito de BM25.

| Variante | Desenvolvimento: precisão silver@6 | Desenvolvimento: MRR silver | Holdout: precisão silver@6 | Holdout: MRR silver |
|---|---:|---:|---:|---:|
| Legado denso + MMR | 0,5833 | 0,6875 | 0,7708 | 0,9375 |
| Denso novo + dedup + MMR | 0,5833 | 0,6875 | 0,7917 | 0,9375 |
| **Híbrido RRF, sem MMR** | **0,6771** | 0,7750 | **0,8125** | 0,9167 |
| Híbrido RRF + MMR | 0,6354 | **0,7937** | 0,7292 | 0,9167 |

O híbrido sem MMR teve a maior precisão silver média entre essas variantes nos dois conjuntos. Isso sustenta uma escolha local do padrão, **não superioridade em toda pergunta ou métrica**. No holdout, seu MRR foi menor que o do legado; a precisão caiu em H01 e H03, aumentou em H05 e H08 e ficou igual nos outros quatro casos. Os resultados por pergunta estão nos relatórios `.md` e nos JSONs.

| Variante | Mediana por consulta no desenvolvimento (ms) | Mediana por consulta no holdout (ms) |
|---|---:|---:|
| Legado denso + MMR | 491,26 | 286,81 |
| Denso novo + dedup + MMR | 345,57 | 296,30 |
| Híbrido RRF, sem MMR | 320,28 | 250,50 |
| Híbrido RRF + MMR | 310,60 | 249,86 |

São medições de uma passagem em CPU, sem intervalo estatístico nem isolamento de carga concorrente. Não são SLA, throughput de produção ou benchmark de LLM. A proporção de textos únicos foi 1,0 em todas as variantes nos dois conjuntos; portanto, esses dados não demonstram ganho observado de diversidade pela deduplicação, embora os testes verifiquem a barreira contra repetições.

### Limites da comparação

As regex podem favorecer coincidência lexical e BM25, ignorar paráfrases ou confundir negação e polaridade. Há apenas 24 perguntas escolhidas, com temas conhecidos da base e regras criadas no mesmo contexto do projeto. O limiar 0,30 e o mínimo de duas fontes permanecem heurísticos, sem calibração como detector universal de relevância ou insuficiência.

Não foram medidos recall sobre todo o corpus, nDCG com julgamentos graduados, entailment universal, satisfação de usuários ou qualidade global do RAG. Não há rótulos humanos independentes suficientes para essas conclusões. Tampouco se conclui superioridade sobre outro projeto: uma comparação exigiria o mesmo corpus, perguntas, filtros, k, modelos, protocolo e rótulos.

O reranker cross-encoder é opcional e não foi executado no benchmark real registrado (`reranker_model=null`). Seu funcionamento de integração foi testado de forma controlada; não há melhora, latência ou qualidade real atribuída ao modelo opcional.

## 3. Testes de implementação e cenários de integração

A suíte automatizada cobre preparação, valores ausentes, identidade de reviews, vínculos múltiplos, filtros conjuntivos, notas conflitantes, datas, UF, roteamento, ranking, deduplicação, integridade do índice, análise agregada e contrato de geração. Encoders e respostas controladas deixam esses testes determinísticos; eles verificam lógica, não a qualidade semântica de um modelo real. A contagem final de testes e o comando realmente executado constam no relatório de validação.

`tests/test_safety_v2.py` exercita trechos alterados, números e causas fabricados, fonte trocada, referências desconhecidas, tema incompatível, prosa/campos extras, recusa contraditória, sucesso sem achados e instrução não confiável mantida dentro de uma citação. Números realmente presentes no comentário são preservados como relato. Esses testes são regressões de barreiras específicas, sem alegação de resistência a todas as injeções ou de julgamento semântico perfeito.

`evaluation/scenarios.json` registra **19 cenários reais** na execução salva: busca temática com embeddings reais, filtros, consultas agregadas, recortes vazios, limiar alto, fora de escopo e base analítica pequena. Esses cenários usam modos extrativo/analítico e não substituem a demonstração de geração real. Os casos de escopo incluem expressões conhecidas pelas regras, portanto não comprovam generalização a toda consulta fora do domínio.

`evaluation/results.json` é o smoke test lexical das perguntas de exemplo. Sua proxy usa o número de evidências retornadas como denominador, enquanto o benchmark pareado divide pelo k solicitado. As métricas têm definições diferentes e não devem ser misturadas. Confira o manifesto de cada arquivo antes de compará-los; um resultado de esquema/versão anterior é histórico, não valida automaticamente o código atual.

## 4. Análise agregada: números verificáveis e alcance limitado

A rota analítica conta **reviews textuais únicos no corpus completo filtrado**, sem usar top-k como população. Todas as variantes de texto e os pedidos ligados ao review são preservados; os filtros de pedido precisam coincidir no mesmo vínculo. Um review conta uma vez por tema, e as regras podem atribuir vários temas ao mesmo review.

Os resultados incluem unidade, denominador, filtros, taxonomia `lexical-2026-10-v2`, contagens, IDs, regex e exemplos. Percentuais só são publicados com pelo menos 30 reviews elegíveis; em base menor ficam `null` e o status sinaliza insuficiência. Esse mínimo é uma política operacional, não resultado de um cálculo amostral ou garantia estatística.

Os testes verificam denominadores, unicidade, não multiplicação por chunks/pedidos, filtros, sobreposição, base pequena e consistência entre fatos estruturados, exemplos e texto renderizado. Nenhum LLM escreve ou estima esses números. Isso demonstra consistência determinística, **não acurácia humana da taxonomia**. As regras têm cobertura parcial; negações complexas, ironia, erros de escrita e sinônimos podem produzir erros. A cobertura lexical publicada não é recall.

Uma taxa refere-se somente ao conjunto textual filtrado, excluindo reviews sem texto. Temas se sobrepõem e seus percentuais não devem ser somados. “Atraso mencionado no comentário”, “entrega posterior à estimativa” e “nota baixa” permanecem medidas diferentes; não se deduz causa nem prevalência entre todos os clientes.

## 5. Geração real e contrato restrito

O contrato v2 permite que o LLM selecione até três achados `{theme, citation, quote}`. A aplicação verifica a existência da referência, a cópia literal do trecho e uma regra lexical do tema e renderiza rótulos neutros. Se o modelo apenas acrescentar pontuação terminal (`.!?…`), ela pode ser removida, desde que todo o conteúdo restante exista literalmente na fonte. A normalização é declarada em `quote_normalization`; palavras, números e pontuação interna não são corrigidos. Prosa factual livre, novos números e campos extras não são publicados. Trata-se de geração restrita orientada a evidências, com menor expressividade; não é um verificador universal de entailment.

O arquivo `evaluation/llm-demo.json` registra uma **nova inferência completa aceita no contrato v2**, com Qwen2.5-0.5B-Instruct local, sem quantização do LLM, geração greedy e limite de 180 tokens. O modelo selecionou um achado `atraso`, ligado a E1, dentre três avaliações recuperadas. O trecho publicado é “Gostaria de esclarecimentos acerca da entrega que está em atraso”, do review `7c4c2d10a57d4aab58a570a5a8ac9f2c`. A saída bruta acrescentou apenas um ponto final; a validação retirou essa pontuação e registrou `quote_normalization=removed_terminal_punctuation`, sem alterar palavras ou números.

A consulta levou **51,97 segundos**, incluindo carregamento do LLM nessa execução; não é um benchmark isolado de geração nem um tempo esperado para toda consulta. O JSON preserva prompts, saída bruta, modelo, configuração, manifesto, fontes e resultado validado. Essa execução comprova seleção de um achado com LLM real e resposta rastreável de escopo restrito. Não comprova síntese analítica avançada, desempenho geral do modelo pequeno, qualidade em muitas perguntas ou resistência universal a injeções.

As tentativas v2 rejeitadas foram preservadas em `llm-v2-first-attempt.json` e `llm-v2-punctuation-attempt.json`: a primeira não obedeceu ao contrato; a segunda produziu trecho com ponto final que a política estritamente literal anterior rejeitava. A política final só normaliza pontuação terminal com nova checagem literal e expõe a normalização. O sucesso final vem de nova inferência completa, não da substituição manual de fatos ou do reaproveitamento de uma resposta de contrato anterior.

Os arquivos de tentativas anteriores preservam falhas reais de JSON do modelo pequeno e repetição no experimento quantizado de 1,5B. Esses resultados são limitados ao prompt, modelo, quantização e hardware utilizados; não permitem classificar genericamente um modelo como melhor ou pior.

O backend HTTP foi exercitado com respostas controladas e falhas de conexão. Serviço remoto/Ollama real, qualidade de síntese livre e muitas respostas geradas não são considerados validados sem seus próprios registros. Somente o modo extrativo não cumpre a etapa de geração por LLM prevista no enunciado.

## 6. Reproduzir e interpretar os artefatos

Com o ambiente ativado, os CSVs em `data/raw` e o índice construído:

```bash
python scripts/run_notebook.py
python -m pytest -q
python scripts/evaluate.py --index data/index
python scripts/evaluate_retrieval.py --index data/index --split development --questions evaluation/retrieval-questions.json --output evaluation/retrieval-benchmark.json
python scripts/evaluate_retrieval.py --index data/index --split holdout --questions evaluation/retrieval-holdout-questions.json --output evaluation/retrieval-holdout.json
python scripts/demo_llm.py
python scripts/check_ui.py
```

Para reproduzir o encoder do benchmark, a ingestão deve usar a variante `--quantize` registrada no manifesto. `scripts/run_notebook.py` aceita `--raw` e `--output` e executa Python comum, sem exigir Jupyter; não implementa magics nem um kernel IPython. O script de LLM exige os pesos/serviço e variáveis documentados. Reranker só participa do benchmark se `--reranker` for fornecido explicitamente.

Em ambientes onde o temporário padrão do pytest está sem permissão, use um `--basetemp` novo dentro de uma pasta gravável do projeto; isso trata uma restrição do ambiente, sem alterar o resultado esperado do teste. Não compare execuções que falharam na preparação de fixtures como se fossem resultados funcionais do sistema.

Os JSONs preservam as evidências por pergunta, configurações, hashes e tempos; os CSVs facilitam comparar as mesmas consultas; os Markdown resumem esses registros. Docker, instalação limpa em outro sistema, teste de carga e revisão humana não são presumidos por esses artefatos. Uma avaliação humana independente de relevância e apoio dos achados seria a próxima evidência necessária para ampliar as conclusões além destas verificações automáticas.
