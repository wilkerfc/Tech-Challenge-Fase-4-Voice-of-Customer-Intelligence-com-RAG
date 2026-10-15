# Recuperação: benchmark pareado real (holdout)

Corpus: 42,687 chunks; k=6; 8 perguntas. Mesmo MiniLM e vetores em todas as variantes.

| Variante | Precisão silver@k | MRR silver | Textos únicos | Latência mediana ms |
|---|---:|---:|---:|---:|
| legacy_dense_mmr | 0.7708 | 0.9375 | 1.0000 | 286.81 |
| dense_dedup_mmr | 0.7917 | 0.9375 | 1.0000 | 296.30 |
| hybrid_rrf | 0.8125 | 0.9167 | 1.0000 | 250.50 |
| hybrid_rrf_mmr | 0.7292 | 0.9167 | 1.0000 | 249.86 |

## Mesmas perguntas, resultados por caso

A média não significa melhoria em cada pergunta. Valores são precisão silver@k.

| ID | Pergunta | Legado denso+MMR | Híbrido RRF | Híbrido RRF+MMR |
|---|---|---:|---:|---:|
| H01 | O que dizem os compradores que esperaram muito para receber? | 0.833 | 0.667 | 0.500 |
| H02 | Há reclamações de equipamentos que pararam de funcionar? | 0.500 | 0.500 | 0.333 |
| H03 | Os consumidores gostaram de receber antes da data combinada? | 1.000 | 0.833 | 0.667 |
| H04 | Como foram as experiências com a cor ou o tamanho errado? | 1.000 | 1.000 | 1.000 |
| H05 | Quais relatos descrevem caixas amassadas ou produtos mal protegidos? | 0.500 | 1.000 | 0.833 |
| H06 | Clientes conseguiram contato e retorno da loja? | 1.000 | 1.000 | 1.000 |
| H07 | O que contam sobre solicitar o estorno depois de devolver a compra? | 1.000 | 1.000 | 1.000 |
| H08 | O que dizem sobre montar o móvel sem instruções? | 0.333 | 0.500 | 0.500 |

Cosseno mínimo 0,30 (heurístico, sem calibração nova); MMR λ=0,8; RRF k=60; pool legado=max(8k,40), novo=100. Normalização BM25 PT leve, não stemmer linguístico completo. A escolha de variante após consultar este conjunto é exploratória; exige validação em perguntas novas.

## Limites

- 8 perguntas no conjunto holdout; relevância silver por regex, sem rótulos humanos independentes.
- Regex podem favorecer BM25 e não reconhecem toda paráfrase ou negação: resultado não mede entailment.
- Mesmas perguntas/filtros/vetores nas variantes; baseline legado reproduz ranking anterior com filtros atuais.
- Precisão divide pelo k solicitado inclusive quando retorna menos; MRR usa primeira evidência que casa regex.
- Latência após aquecimento inclui encoder/busca/filtros/MMR, exclui carga inicial; uma passagem, sem intervalo estatístico.
- Nenhuma métrica permite comparar diretamente com B: perguntas, k, corpus, modelos e rótulos diferentes.
- Sem chamadas a LLM e sem auditoria humana nova. Reranker só avaliado se --reranker for fornecido.
- Padrão hybrid_rrf sem MMR escolhido nas 16 perguntas de desenvolvimento e congelado antes das 8 holdout; sem ajuste posterior ao holdout.

IDs, textos, regex, filtros e tempos por pergunta estão em retrieval-benchmark.json e no CSV. Hashes do índice registram os dados realmente usados.
