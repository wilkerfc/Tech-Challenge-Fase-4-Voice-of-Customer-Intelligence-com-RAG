# Roteiro de apresentação - até 7 minutos

Grave sua tela e narração após executar a aplicação. Duração alvo: 6min40s. Não inclua chaves, tokens ou dados pessoais visíveis. Este documento é um roteiro, não um vídeo pronto.

| Tempo | Conteúdo e demonstração |
|---|---|
| 0:00-0:40 | Problema: gestores recebem avaliações e precisam entender relatos com evidências; apresentar a base Olist histórica. |
| 0:40-1:25 | Mostrar diagrama do README: preparação, embeddings, recuperação, contexto e geração. |
| 1:25-2:10 | Explicar vazios, deduplicação composta, título+texto e categorias por pedido. Mostrar manifesto e tamanho real do índice. |
| 2:10-2:55 | Mostrar busca por cosseno, limiar, MMR e filtros. Explicar que top-k não mede frequência na base inteira. |
| 2:55-4:10 | Selecionar modo LLM. Perguntar sobre atrasos com notas 1-2. Ler a síntese e abrir duas evidências, mostrando IDs e nota. |
| 4:10-5:00 | Perguntar por elogios à rapidez com notas 4-5. Mostrar os relatos que sustentam a resposta. |
| 5:00-5:40 | Perguntar pela previsão do tempo e mostrar recusa; usar filtro sem resultados para demonstrar falta de evidências. |
| 5:40-6:20 | Mostrar testes e relatório real; explicar proxy lexical, limitações e necessidade de avaliação humana do LLM. |
| 6:20-6:40 | Apresentar repositório e instruções de execução. Encerrar com os próximos passos: avaliação anotada e busca híbrida. |

## Preparação

1. Indexar toda a base (sem `--limit`) ou identificar verbalmente a amostra utilizada.
2. Disponibilizar o LLM local ou remoto e fazer uma pergunta de teste antes de gravar.
3. Manter o README e a interface abertos.
4. Escolher evidências reais, conferir referências e não ler uma resposta inventada como resultado da aplicação.
5. Gravar, revisar duração e qualidade de áudio, e publicar o vídeo conforme o canal exigido pelo curso.
