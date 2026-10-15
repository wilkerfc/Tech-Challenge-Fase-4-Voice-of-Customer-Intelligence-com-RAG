# Critérios de aceite e mapa de evidências

Este documento transforma os requisitos do enunciado **[IAST] — Tech Challenge — Fase 4**, de 12 páginas, em verificações concretas. As páginas citadas são as páginas do PDF, incluindo a capa. O documento serve para revisar a entrega; não representa uma nota atribuída pela instituição nem uma auditoria humana de relevância ou fidelidade.

Os registros de execução são a fonte para saber **o que foi realmente testado**, com qual corpus e modelo. A existência de código, de um teste ou de um vídeo não prova, por si só, que o requisito foi atendido em todos os cenários. Após mudar código, dados, índice, modelo ou configuração, reexecute os caminhos afetados e atualize os artefatos antes de apresentar resultados como atuais.

## 1. Base de conhecimento e decisões técnicas

| ID | Exigência do PDF | Critério verificável | Evidência e reprodução |
|---|---|---|---|
| B01 | Usar avaliações textuais da Olist, pp. 4–6 | O corpus deve vir dos CSVs da base indicada; demonstrar seleção de título/mensagem, sem fabricar texto ausente | `scripts/download_data.py`; `evaluation/data-provenance.json`; notebook `notebooks/01_exploracao_e_decisoes.ipynb`, seções 1–2 |
| B02 | Definir e justificar seleção, p. 6 | Informar tabelas utilizadas, unidade documental e quantidade efetiva após preparação; diferenciar pedidos de reviews | `voc/data.py`; `docs/arquitetura.md`; saídas reais do notebook |
| B03 | Limpeza e preparação, p. 6 | Remover ruído de HTML/espaços preservando acentos e negações; explicar mascaramento parcial e limites | `voc/data.py`; notebook, seção 6; testes de limpeza |
| B04 | Tratamento de ausentes, p. 6 | Excluir texto vazio sem imputação sintética; não converter datas de entrega desconhecidas em pontualidade | Estatísticas de `prepare`; notebook, seções 2 e 5; testes de preparação |
| B05 | Organização de documentos, p. 6 | Documentos devem ter texto, ID estável e vínculo de origem; joins de itens não podem multiplicar reviews | `voc/data.py`; notebook, seção 3; metadados exportados nas respostas |
| B06 | Criação de embeddings, p. 6 | Informar modelo, dimensões e variante de execução; codificar pergunta e documentos de forma compatível | `voc/retrieval.py`; `data/index/manifest.json` após ingestão; manifesto de `evaluation/results.json` |
| B07 | Armazenamento/indexação, p. 6 | Demonstrar persistência de vetores e documentos e carregamento consistente; explicar o custo da busca escolhida | `voc/retrieval.py`; `docs/arquitetura.md`; notebook, seção 7; testes de índice |
| B08 | Justificar escolhas, pp. 6–7 e 9 | Explicar seleção, limpeza, identidade, embeddings, recuperação, contexto e geração, com limites e alternativas | `docs/arquitetura.md`; notebook, seção 8; `README.md` |

O notebook é uma exploração de dados executada, com contagens, hashes e verificações estruturais. Não mede qualidade de recuperação e não substitui a execução do RAG. Os CSVs e índices são artefatos locais, reproduzidos pelos scripts; não devem ser confundidos com arquivos versionados no Git.

Para reproduzir suas saídas sem instalar Jupyter: `python scripts/run_notebook.py`. O script executa células Python comuns, em ordem, com o mesmo interpretador do ambiente, e salva stdout/stderr no `.ipynb`. Ele não implementa magics do IPython nem um kernel Jupyter. Use `--raw CAMINHO` para outra cópia dos CSVs e `--output CAMINHO.ipynb` para preservar o arquivo original.

## 2. Pipeline RAG completo

| ID | Exigência do PDF | Critério verificável | Evidência e reprodução |
|---|---|---|---|
| P01 | Processar/transformar a pergunta, pp. 5–7 | Normalizar a consulta; rejeitar entradas inválidas e explicitar qualquer classificação/roteamento adotado | `voc/retrieval.py`; `voc/pipeline.py`; testes de entrada e escopo |
| P02 | Recuperação semântica, pp. 6–7 | Codificar uma pergunta real e recuperar avaliações por seus vetores, e não somente por regras de exemplo | `scripts/evaluate.py`; perguntas e resultados em `evaluation/`; teste com encoder real registrado no relatório |
| P03 | Seleção dos conteúdos mais relevantes, p. 7 | Informar top-k, limiar, diversidade e momento dos filtros; preservar IDs e scores para inspeção | `voc/retrieval.py`; `docs/arquitetura.md`; resultados reais com evidências |
| P04 | Construção do contexto, p. 7 | Contexto deve conter somente evidências selecionadas, dentro do orçamento, com referências que o modelo possa usar | `voc/pipeline.py`; testes de orçamento e consistência de referências |
| P05 | Geração final por modelo de linguagem, pp. 6–7 | Executar um LLM real usando a pergunta e os trechos recuperados; guardar modelo, entrada, saída bruta e resultado validado | `scripts/demo_llm.py`; `evaluation/llm-demo.json`; modo LLM da interface e CLI |
| P06 | Diferenciar caminho de diagnóstico do RAG gerativo | Identificar claramente modo extrativo e geração; erro de LLM não pode aparecer como geração bem-sucedida | `app.py`; `voc/pipeline.py`; testes de falha; documentação do backend |

**Limite de aceite:** uma resposta apenas extrativa prova recuperação e rastreabilidade de trechos, mas não atende sozinha ao requisito P05. Um mock de API permite testar o contrato e as falhas, mas não comprova uma execução real de LLM. Uma demonstração válida de um modelo pequeno também não comprova qualidade geral, desempenho de outro modelo ou um serviço remoto que não tenha sido executado.

## 3. Coerência, rastreabilidade e insuficiência

| ID | Exigência do PDF | Critério verificável | Evidência e reprodução |
|---|---|---|---|
| E01 | Respostas fundamentadas nos comentários, p. 7 | Cada achado exibido deve apontar a uma avaliação recuperada e a um trecho da própria fonte; rejeitar citações inexistentes e trechos alterados | `voc/pipeline.py`; `tests/test_safety_v2.py`; evidências de `evaluation/llm-demo.json` |
| E02 | Permitir identificar as avaliações, pp. 7 e 10 | Exibir texto e identificadores de origem junto da resposta e incluí-los na exportação | `app.py`; resposta JSON da CLI; `review_id`, `order_id` e marcadores de evidência |
| E03 | Recuperar conteúdo relevante e produzir coerência, pp. 7 e 10 | Comparar perguntas reais, evidências e saída; distinguir testes sintáticos de apoio semântico | `docs/avaliacao.md`; `evaluation/results.json`; `evaluation/retrieval-benchmark.*`; resultados de geração e limites explícitos no relatório |
| E04 | Ausência de avaliações relevantes, pp. 7–8 | Filtros sem correspondência ou resultados abaixo do limiar devem provocar recusa antes de gerar | Testes de filtros e limiar; caso reproduzível com categoria inexistente |
| E05 | Evidências insuficientes, pp. 7–8 | Aplicar a política mínima de evidências e aceitar abstenção; não exibir achados contraditórios com recusa | `voc/pipeline.py`; testes de uma evidência, recusa e contrato inválido |
| E06 | Perguntas fora do escopo, p. 8 | Recusar casos sem suporte na base e documentar limites do mecanismo; não alegar detector universal | Perguntas fora do domínio em `evaluation/`; testes de escopo e `docs/arquitetura.md` |
| E07 | Evitar informações sem suporte, pp. 7–8 e 10 | Não confiar em números, causas ou prosa livre vindos do LLM; validar o contrato adotado e renderizar apenas o que ele autoriza | `voc/pipeline.py`; `tests/test_safety_v2.py`; casos de números/causas inventados, fonte trocada e campos extras |
| E08 | Explorações sobre problemas, elogios e padrões, pp. 4–5 e 8 | Demonstrar consultas de reclamações, tema específico, experiências positivas e insatisfação, preservando o vínculo com fontes | Exemplos do `README.md`; perguntas e saídas em `evaluation/`; demonstração da interface |

### Interpretação correta das barreiras

- Uma referência válida não prova que toda interpretação é correta. O contrato restrito reduz a superfície de prosa livre: o modelo seleciona tema, citação e trecho literal, e a aplicação renderiza o achado. A compatibilidade lexical do tema é uma verificação conservadora e não equivale a entailment universal.
- Um trecho literal pode conter uma instrução maliciosa ou uma alegação incorreta do próprio consumidor. Ele permanece dado citado, não uma instrução confiável nem um fato operacional comprovado.
- Top-k não estima frequência na população. Uma análise de frequência precisa de corpus/filtros, unidade de contagem, definição de tema, denominador e limitações explícitos. Regras lexicais são proxies automáticos; não são anotações humanas.
- Atraso confirmado por datas, relato textual de demora e nota baixa são conceitos distintos. Não inferir causalidade nem afirmar que todos os comentários de um pedido se referem a uma categoria específica.
- Revisão humana de relevância, fidelidade e utilidade permanece uma etapa recomendada. Só marque essa etapa como executada se existir procedimento, amostra, responsáveis e resultados reais.

## 4. Metadados opcionais

O desafio adicional de exploração contextual é **não obrigatório**, conforme pp. 8–9. Se apresentado, deve ser justificado e demonstrado.

| Metadado | Aceite | Evidência |
|---|---|---|
| Nota | Filtro de 1 a 5 antes do ranking; não tratar nota como causa/sentimento validado | `voc/retrieval.py`; testes de filtro; notebook, seção 4 |
| Período de compra | Filtrar pela data da compra e indicar o significado da data; validar intervalo | `app.py`; testes de data; notebook, seção 5 |
| Pedido/entrega | Preservar vínculo do pedido e diferenciar atraso, pontualidade, não entrega e desconhecido conforme os campos disponíveis | `voc/data.py`; documentação dos metadados; testes de estados ausentes |
| Categorias | Agregar por pedido sem multiplicar reviews; avisar sobre pedidos multicategoria | `voc/data.py`; filtros; notebook, seção 5 |

## 5. Repositório e reprodução

O PDF exige um **repositório público no GitHub**, com os itens abaixo, nas pp. 9–10. A publicação e a visibilidade devem ser confirmadas no destino; um commit somente local não atende a esse requisito.

| Item exigido | Evidência esperada |
|---|---|
| Código-fonte | `voc/`, `app.py`, `scripts/` e testes no repositório remoto |
| Instalação e execução | Comandos do `README.md`; `requirements.txt` e registro do ambiente validado |
| README explicando o projeto | Problema, comportamento, preparação, execução, exemplos e limitações |
| Arquitetura desenvolvida | Diagrama e `docs/arquitetura.md` |
| Tecnologias e bibliotecas | Dependências e escolhas descritas no README/documentação |
| Estratégia RAG | Embeddings, ranking, filtros, contexto e geração descritos e demonstrados |
| Preparação da base | `voc/data.py`, notebook executado e proveniência dos arquivos |
| Justificativas técnicas | Arquitetura, avaliação e notebook |
| Perguntas e respostas | Exemplos reais exportados, com distinção entre extrativo e LLM |
| Informações de reprodução | Fonte de dados, comandos, variáveis de ambiente, configuração/modelos, resultados e limites do ambiente testado |
| Organização e clareza | Separação entre código, dados locais, avaliação, documentação e apresentação |

Comandos-base, a partir da raiz do projeto e com o ambiente ativado:

```bash
python scripts/download_data.py
python scripts/run_notebook.py
python -m voc.cli ingest --raw data/raw
python -m pytest -q
python scripts/evaluate.py --index data/index
python scripts/demo_llm.py
python scripts/check_ui.py
streamlit run app.py
```

Os scripts de LLM precisam das variáveis e dos pesos/serviço documentados. Se a avaliação tiver opções adicionais, consulte `--help` e o procedimento em `docs/avaliacao.md`. Os resultados devem informar a variante de encoder, a configuração do LLM e se a validação usou todo o corpus ou uma amostra.

O lock registra uma configuração de dependências; não é prova de instalação limpa em todo sistema. Docker/Compose são alternativas de reprodução e só devem ser descritos como testados se uma execução real tiver ocorrido. Não publique tokens, chaves, ambientes virtuais, pesos ou arquivos brutos de dados por acidente.

## 6. Vídeo de até sete minutos

O requisito da p. 10 combina **duração máxima de 7 minutos** e explicação do raciocínio técnico. Os arquivos `docs/apresentacao.mp4`, `docs/video.md`, `docs/video-narration.json` e o roteiro devem se referir à mesma versão da solução.

| Conteúdo obrigatório | O que conferir na apresentação final |
|---|---|
| Problema de negócio | Explica por que gestores precisam explorar as falas de clientes |
| Arquitetura proposta | Mostra o caminho dos CSVs até a resposta e suas evidências |
| Estratégia de construção do RAG | Justifica preparação, embeddings, recuperação, contexto e geração |
| Recuperação de avaliações | Mostra uma consulta e trechos realmente recuperados |
| Demonstração prática | Mostra comportamento executado da solução, com clareza sobre gravação de interface ou resultados exportados |
| Exemplos de perguntas | Inclui consultas pertinentes e um caso de insuficiência/escopo |
| Evidências das respostas | Exibe referências e comentários que sustentam os achados |
| Desafios e decisões | Explica limites observados e escolhas, sem apresentar uma tentativa falha como sucesso |

A narração sintética deve ser identificada quando usada. A duração deve ser medida no arquivo final. Uma apresentação com resultados exportados deve ser descrita dessa forma, sem alegar captura contínua da interface ou revisão humana que não aconteceu.

## 7. Checklist final honesto

Antes de submeter, confira as evidências da versão final:

- [ ] Revisar os resultados atuais de testes e dependências no relatório de validação.
- [ ] Confirmar que os CSVs usados correspondem à proveniência e que o índice usa o código/configuração descritos.
- [ ] Abrir o notebook e conferir que todas as células têm saídas reais, sem erros e sem alegação de auditoria humana.
- [ ] Executar consultas representativas e recusa; verificar fontes e exportação na interface.
- [ ] Conferir a demonstração de LLM real e suas limitações, separada dos mocks e do modo extrativo.
- [ ] Verificar se README, arquitetura, avaliação e vídeo descrevem o comportamento final.
- [ ] Confirmar duração de até 420 segundos e todos os conteúdos obrigatórios no vídeo final.
- [ ] Confirmar commit publicado, arquivos acessíveis e visibilidade pública no GitHub.
- [ ] Enviar link do repositório e vídeo no ambiente acadêmico pelo responsável pela entrega.

Esses itens exigem conferência final; não estão marcados automaticamente por existir este documento. O relatório `evaluation/validation.md` deve registrar apenas verificações realmente concluídas, com resultados e limitações.

O bônus **Databricks Generative AI Engineering Pathway**, pp. 2–3, é opcional e depende da conclusão real da trilha pelo aluno. Certificado, badge ou captura que identifique a conta e os cursos concluídos deve ser obtido pelo próprio responsável. Nenhum arquivo deste projeto comprova a conclusão da trilha ou a submissão no ambiente acadêmico.
