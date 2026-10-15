# Checklist de entrega

| Requisito do enunciado | Implementação / evidência |
|---|---|
| Código-fonte e reprodução | README, requirements, CLI, app, Docker |
| Seleção e preparação | voc/data.py; estatísticas da ingestão |
| Embeddings e armazenamento | voc/retrieval.py; manifesto local |
| Recuperação semântica | cosseno + BM25 + RRF; deduplicação; MMR opcional |
| Contexto e geração | voc/pipeline.py; modo llm configurável |
| Evidências rastreáveis | texto, review_id, order_id e citações |
| Insuficiência / fora de escopo | limiar, mínimo de documentos, barreira e prompt |
| Metadados opcionais | notas, datas, categorias, UF, situação da entrega e vínculos múltiplos |
| Justificativas e arquitetura | docs/arquitetura.md |
| Perguntas e respostas | evaluation/scenarios.json, results.json e llm-demo.json |
| Avaliação | 89 testes, notebook executado, comparação pareada dev/holdout e limites documentados |
| Vídeo até 7 minutos | apresentação narrada de 5min16,76s em docs/apresentacao.mp4; roteiro também fornecido |
| Bônus Databricks opcional | depende de realização e comprovação pelo aluno |

Para entregar: revise o projeto e a apresentação, confira a publicação do repositório e envie o vídeo e o link do repositório no ambiente da instituição. O vídeo utiliza narração sintética identificada e resultados reais registrados; você pode substituí-lo por uma gravação pessoal seguindo o roteiro. Não há acesso ao ambiente acadêmico nesta solução nem comprovação de curso extra. A publicação depende de autenticação GitHub com permissão de escrita.
