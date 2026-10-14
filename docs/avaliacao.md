# Avaliação e limites da evidência

## Testes de comportamento

`python -m pytest -q` verifica ingestão com duplicatas e vazios, validação de nota, preservação de IDs repetidos em pedidos distintos, seleção por metadados, documentos únicos, recusa e contrato de geração. O encoder fake é deliberado: torna determinístico o teste da lógica; ele não demonstra qualidade semântica.

## Experimento com embeddings reais

`python scripts/evaluate.py` usa a base indexada e o modelo do manifesto. Há quatro perguntas temáticas com filtros de nota e duas consultas fora do escopo. Para cada pergunta temática, mede-se a proporção de trechos top-k contendo alguma raiz lexical predefinida. Essa proxy indica se a recuperação encontra sinais do tema; pode produzir falso positivo e falso negativo. As recusas fora de escopo testam expressões conhecidas pela barreira, portanto não demonstram generalização.

O relatório preserva os textos, review_id, order_id, similaridades e configurações. Não há rótulos humanos suficientes para declarar recall, nDCG ou fidelidade do LLM. Não se usa a proxy lexical como nota de qualidade do sistema inteiro. O número de avaliações indexadas e o uso de amostra são registrados no manifesto.

## Validação recomendada para a apresentação

Prepare pelo menos 30 perguntas com diversidade de categorias e notas. Dois avaliadores devem marcar relevância dos documentos em escala 0-2 e apoio de cada afirmação gerada nas evidências. Compare busca simples, MMR e eventualmente busca híbrida; reporte precision@k, nDCG@k, taxa de recusa correta, cobertura de citações e proporção de afirmações sustentadas. Divida consultas de calibração e teste para escolher o limiar sem adaptar tudo ao mesmo conjunto.

Para o modo LLM, execute o modelo escolhido com temperatura zero, guarde versão/configuração e revise as respostas. Teste pedidos por fatos ausentes e comentários com instruções maliciosas. Referência sintaticamente válida não comprova fidelidade.

## Critérios de aceitação do protótipo

- Instalação documentada em Python 3.12.
- Evidências reais rastreáveis após a indexação.
- Filtros obedecidos antes do ranking.
- Recusa sem inventar resposta quando não há resultados suficientes.
- Falha do serviço LLM explícita.
- Relatórios distinguem modo extrativo e geração.

O arquivo `evaluation/validation.md` descreve os comandos efetivamente executados nesta entrega; Docker e geração real só devem ser considerados validados quando houver evidência de execução.
