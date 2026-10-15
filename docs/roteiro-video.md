# Roteiro para gravação pessoal — até 7 minutos

O vídeo pronto está em [apresentacao.mp4](apresentacao.mp4). Este roteiro permite substituí-lo por uma gravação da interface com a própria voz, mantendo os resultados conferidos. Meta: 6min40s.

| Tempo | Conteúdo |
|---|---|
| 0:00–0:35 | Problema de negócio: investigar experiências de compra com evidências, além da nota. Base Olist histórica. |
| 0:35–1:15 | Mostrar arquitetura e duas rotas: RAG qualitativo e análise determinística de frequência. |
| 1:15–2:00 | Explicar texto vazio, limpeza, documentos/review_ids, metadados e pedidos múltiplos; abrir notebook e manifesto schema 2. |
| 2:00–2:45 | Explicar MiniLM, cosseno, BM25, RRF e deduplicação. Padrão sem MMR; filtros conjuntivos antes do ranking. |
| 2:45–3:45 | Rodar pergunta sobre atraso em modo LLM previamente aquecido. Conferir trecho, citação e review_id; mostrar saída bruta e contrato restrito. |
| 3:45–4:35 | Perguntar pelos temas frequentes em notas 1–2/SP. Mostrar denominador, contagens, exemplos e sobreposição; top-k não mede prevalência. |
| 4:35–5:10 | Perguntar por elogios à rapidez com notas 4–5 e abrir fontes. |
| 5:10–5:45 | Demonstrar 2099 sem evidências, previsão do tempo fora do escopo e assunto analítico não coberto, como cashback. |
| 5:45–6:20 | Mostrar testes e resultados dev/holdout, inclusive regressões/MRR e limites da relevância silver. |
| 6:20–6:40 | Abrir repositório e instruções de reprodução. Indicar avaliação humana como etapa futura. |

Antes de gravar, confirme corpus completo, modelo configurado e execução das consultas. Em CPU limitada, aqueça o LLM e reserve o resultado real exportado para explicar a latência sem ultrapassar o tempo. Identifique claramente qualquer resultado pré-executado. Não apresente mocks ou respostas escritas manualmente como geração real. Revise áudio, duração, fontes e conteúdo antes da submissão.
