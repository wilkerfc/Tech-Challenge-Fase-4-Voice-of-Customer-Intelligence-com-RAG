# Validação executada - 14/10/2026

Ambiente: Windows, Python 3.12, CPU. Dependências principais fixadas e versões transitivas registradas em `requirements-lock.txt`.

## Resultados confirmados

| Verificação | Resultado |
|---|---|
| `python -m pytest -q` | 12 testes passaram |
| Verificação de dependências (`pip check`) | Nenhuma dependência quebrada |
| Download oficial com `scripts/download_data.py` | CSVs reproduzidos com os mesmos hashes |
| Ingestão completa | 99.224 linhas lidas, 56.537 sem texto descartadas, 42.687 documentos válidos |
| Embeddings | MiniLM multilíngue, 384 dimensões, camadas lineares quantizadas em int8; vetores float32 normalizados |
| Indexação | Toda a base textual válida, sem `--limit`; 42.687 fragmentos |
| Interface (`scripts/check_ui.py`) | Renderização, consulta real e recusa passaram |
| Geração local | Qwen2.5-0.5B-Instruct, sem quantização, saída textual; citações E1/E2/E3 validadas |

## Recuperação real: resultados exploratórios

| Tema / filtro de nota | Proxy lexical nos 6 resultados |
|---|---|
| Atrasos / 1-2 | 6/6 |
| Produtos quebrados ou com defeito / 1-2 | 5/6 |
| Rapidez da entrega / 4-5 | 6/6 |
| Produto diferente / 1-2 | 4/6 |

As duas perguntas fora do escopo foram recusadas antes da geração. Esses resultados vêm de `results.json`. A proxy não é uma avaliação humana de relevância, não mede recall e não permite afirmar 87,5% de qualidade global do RAG.

## Geração: evidência e limitações observadas

`llm-demo.json` preserva a pergunta, o modelo, a saída bruta, o tempo e as evidências. A execução válida retornou três avaliações citadas, com pequena normalização de pontuação. É uma resposta gerada por um LLM real, mas seu comportamento foi **majoritariamente extrativo**, sem análise aprofundada. A latência registrada inclui carregar o modelo no processo. Citações consistentes não demonstram qualidade generalizada; esta é uma demonstração pontual.

O modelo 0,5B não conseguiu obedecer ao contrato JSON inicial (`llm-small-model-failure.json` e `llm-first-attempt.json`). O experimento 1,5B com quantização dinâmica apresentou repetição e foi rejeitado (`llm-quantized-failure.json`). Depois dessas observações, o backend local passou a gerar texto curto com citações e a deixar a estruturação para a aplicação. A geração local não usa quantização por padrão. Não há manipulação manual de afirmações no resultado aceito.

O teste de falha mostra que o sistema não apresenta uma geração inválida como uma resposta confiável. Não se declara que o modelo maior seja pior em geral: o resultado é limitado a esse experimento, hardware, quantização e prompt.

## Reproduzir os principais caminhos

```bash
python scripts/download_data.py
python -m voc.cli ingest --raw data/raw --quantize
python -m pytest -q
python scripts/evaluate.py
python scripts/demo_llm.py
python scripts/check_ui.py
```

## Limites desta validação

- Docker foi fornecido, mas não executado neste ambiente.
- O backend HTTP remoto foi validado em testes controlados, incluindo falha de conexão e referências inválidas; não foi executado contra Ollama ou provedor remoto real nesta entrega.
- Uma tentativa de inspeção no navegador integrado não abriu o servidor local. A confirmação da interface vem do AppTest do Streamlit, não de inspeção visual pelo navegador.
- Não houve benchmark humano, avaliação de muitas respostas geradas ou teste de concorrência.
- Os pesos dos modelos e a base bruta não estão no Git; a reprodução exige baixar esses recursos.

Revisões do Hugging Face observadas no download: MiniLM `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`; Qwen 0,5B `7ae557604adf67be50417f59c2c2f167def9a775`. O script usa o identificador público do modelo; para reprodução estritamente imutável, mantenha a cópia dos pesos dessa revisão e use o ambiente registrado.
