# Recuperação: benchmark pareado real (development)

Corpus: 42,687 chunks; k=6; 16 perguntas. Mesmo MiniLM e vetores em todas as variantes.

| Variante | Precisão silver@k | MRR silver | Textos únicos | Latência mediana ms |
|---|---:|---:|---:|---:|
| legacy_dense_mmr | 0.5833 | 0.6875 | 1.0000 | 491.26 |
| dense_dedup_mmr | 0.5833 | 0.6875 | 1.0000 | 345.57 |
| hybrid_rrf | 0.6771 | 0.7750 | 1.0000 | 320.28 |
| hybrid_rrf_mmr | 0.6354 | 0.7937 | 1.0000 | 310.60 |

## Mesmas perguntas, resultados por caso

A média não significa melhoria em cada pergunta. Valores são precisão silver@k.

| ID | Pergunta | Legado denso+MMR | Híbrido RRF | Híbrido RRF+MMR |
|---|---|---:|---:|---:|
| R01 | Por que compradores relatam que ficaram esperando a encomenda? | 1.000 | 0.667 | 0.167 |
| R02 | O aparelho deixou de funcionar depois da compra? | 0.500 | 0.000 | 0.000 |
| R03 | O que agradou quem recebeu a compra cedo? | 0.167 | 0.167 | 0.167 |
| R04 | Há relatos de mercadoria diferente da escolhida? | 0.333 | 0.833 | 0.833 |
| R05 | O pacote chegou com peças ou unidades faltando? | 0.833 | 0.833 | 0.833 |
| R06 | Quais problemas aparecem na embalagem da mercadoria? | 0.333 | 0.333 | 0.333 |
| R07 | As compras vieram sem documento fiscal? | 1.000 | 1.000 | 1.000 |
| R08 | O vendedor responde quando o comprador precisa de ajuda? | 0.167 | 0.500 | 0.333 |
| R09 | Quais experiências envolvem devolução e dinheiro de volta? | 0.833 | 1.000 | 1.000 |
| R10 | O preço compensou para compradores satisfeitos? | 0.833 | 0.667 | 0.500 |
| R11 | Há satisfação com o material e o acabamento? | 0.667 | 1.000 | 1.000 |
| R12 | Quais avaliações descrevem cheiro ou aroma? | 0.667 | 1.000 | 1.000 |
| R13 | O que dizem os compradores de cartuchos? | 0.000 | 0.167 | 0.500 |
| R14 | Quais relatos citam pagamento por boleto? | 1.000 | 1.000 | 1.000 |
| R15 | Os produtos vieram com instruções ou manual? | 0.667 | 1.000 | 0.833 |
| R16 | O que reclamam de móveis comprados em 2018? | 0.333 | 0.667 | 0.667 |

Cosseno mínimo 0,30 (heurístico, sem calibração nova); MMR λ=0,8; RRF k=60; pool legado=max(8k,40), novo=100. Normalização BM25 PT leve, não stemmer linguístico completo. A escolha de variante após consultar este conjunto é exploratória; exige validação em perguntas novas.

## Limites

- 16 perguntas no conjunto development; relevância silver por regex, sem rótulos humanos independentes.
- Regex podem favorecer BM25 e não reconhecem toda paráfrase ou negação: resultado não mede entailment.
- Mesmas perguntas/filtros/vetores nas variantes; baseline legado reproduz ranking anterior com filtros atuais.
- Precisão divide pelo k solicitado inclusive quando retorna menos; MRR usa primeira evidência que casa regex.
- Latência após aquecimento inclui encoder/busca/filtros/MMR, exclui carga inicial; uma passagem, sem intervalo estatístico.
- Nenhuma métrica permite comparar diretamente com B: perguntas, k, corpus, modelos e rótulos diferentes.
- Sem chamadas a LLM e sem auditoria humana nova. Reranker só avaliado se --reranker for fornecido.
- Padrão hybrid_rrf sem MMR escolhido nas 16 perguntas de desenvolvimento e congelado antes das 8 holdout; sem ajuste posterior ao holdout.

IDs, textos, regex, filtros e tempos por pergunta estão em retrieval-benchmark.json e no CSV. Hashes do índice registram os dados realmente usados.
