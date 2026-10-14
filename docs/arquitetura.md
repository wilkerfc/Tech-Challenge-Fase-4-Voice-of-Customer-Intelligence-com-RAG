# Arquitetura e decisões técnicas

## Construção da base

1. Download do arquivo oficial Olist pelo endpoint Kaggle; SHA-256 dos CSVs registrado no download.
2. Leitura de reviews com UTF-8; seleção de avaliações com título ou mensagem e nota de 1 a 5.
3. HTML, whitespace, emails e telefones tratados sem apagar acentos ou negações.
4. Deduplicação da tripla review, pedido e texto; mensagens iguais em pedidos distintos permanecem.
5. Join opcional com orders; categorias agregadas de products e order_items. Um pedido com vários produtos possui várias categorias, sem afirmar a qual produto o comentário se refere.
6. Fragmentação por palavras (90, overlap 20). O corpus Olist possui comentários curtos; para outros corpora recomenda-se chunking pelo tokenizer e monitoramento do truncamento.
7. Embeddings float32 normalizados, índice exato NumPy, documentos JSON e manifesto de modelo/contagens.

A opção `--quantize` aplica quantização dinâmica int8 às camadas lineares do encoder em CPU. Os embeddings resultantes continuam em float32 normalizado. O manifesto registra essa variante e a recuperação aplica o mesmo encoder quantizado à pergunta, evitando misturar configurações. A quantização pode alterar similaridades; o relatório deve ser interpretado para a variante realmente executada. `VOC_CPU_THREADS` permite controlar o paralelismo conforme os recursos disponíveis.

Não há inferência de atraso para entregas faltantes; `null` significa desconhecido. A filtragem por período utiliza compra, não data de avaliação. Os IDs e os trechos utilizados acompanham as respostas.

## Recuperação

A pergunta recebe NFKC e normalização de espaços; não é traduzida nem reescrita por LLM. Embeddings do mesmo modelo e dimensão garantem compatibilidade entre consulta e documentos. Filtros são aplicados antes da similaridade. Produto escalar de vetores unitários equivale ao cosseno. Recuperam-se até `max(8*k,40)` candidatos e selecionam-se k documentos distintos com MMR:

`0.8 * similaridade(query, candidato) - 0.2 * maior similaridade(candidato, selecionados)`.

O limiar é aplicado à similaridade original, não ao score MMR. Similaridade não é probabilidade de relevância. Este método equilibra relevância e diversidade, mas pode deixar passar casos relevantes em linguagem muito diferente. Uma versão futura deve comparar BM25 + embeddings, reranking e um benchmark humano, antes de ampliar a complexidade.

## Contexto e geração

No máximo 10 mil caracteres de evidências entram no contexto. Uma seleção conservadora interrompe a inclusão ao atingir o orçamento. Citações são atribuídas às avaliações recuperadas; o prompt exige português, apoio nos dados, abstinência de inferências populacionais e JSON estruturado. Comentários e perguntas ficam serializados como dados no user message; o system message explicita que textos do corpus não são instruções.

O endpoint de geração é configurável por ambiente. Não se executam ferramentas, código ou links sugeridos pelos reviews. Timeout e erro de API mantêm as evidências e informam falha. O parser rejeita JSON inválido, IDs desconhecidos e inconsistência entre referências inline e lista de citações. Isso reduz falhas estruturais; não elimina hallucination nem prompt injection. Um LLM pode citar um documento válido que não sustenta a afirmação.

Há também geração local com Transformers, sem servidor externo. Esse backend carrega pesos em float32, usa geração greedy com limite de 180 tokens e contexto textual com identificadores de evidência. O LLM produz uma síntese textual com marcadores; a aplicação extrai esses marcadores e aplica a validação de referências. Para recusa, o modelo pode retornar `SEM_EVIDENCIAS`. A geração remota mantém o protocolo JSON. O protocolo local foi simplificado após falhas reais de formato; as saídas rejeitadas constam no diretório de avaliação. A quantização do LLM fica desativada por padrão depois de apresentar repetição no experimento 1,5B. `LLM_QUANTIZE=1` é uma opção experimental, e `LLM_MODEL_PATH` permite carregar pesos já baixados. Modelos maiores podem exigir vários GB de RAM. Os pesos não são publicados no repositório. O cache mantém um modelo por processo.

## Insuficiência

- Consulta vazia ou muito longa: erro de validação.
- Exemplos de temas alheios à base: recusa na barreira de escopo.
- Filtros sem correspondência ou menos de duas evidências acima do limiar: recusa sem chamar o LLM.
- O LLM também pode declarar `insufficient=true` ao não encontrar suporte.
- Falha do serviço ou contrato inválido: falha explícita e exibição de evidências.

A barreira de escopo é uma heurística por expressões, não um detector universal. Duas avaliações relevantes podem apoiar a existência de um relato, mas não sua prevalência. Perguntas sobre preços atuais, causas internas, faturamento ou dados ausentes devem ser recusadas pelo LLM, mesmo quando há alguma correspondência semântica.

## Operação e escala

`streamlit.cache_resource` evita recarregar índice/modelo a cada interação. Recriar o índice exige limpar o cache ou reiniciar a aplicação. Para aproximadamente 42 mil reviews e 384 dimensões, vetores ocupam cerca de 65 MB, além de chunks, JSON, modelo e runtime. A implantação deve dimensionar memória pelo índice efetivo. NumPy faz varredura exata O(n*d); é adequado ao protótipo, e ANN passa a ser necessário se volume/concurrência aumentarem.

O índice é um artefato local confiável; não deve ser substituído por arquivos de terceiros. A interface é um protótipo para uso local, sem autenticação e sem isolamento por usuário. Para uma implantação pública, implementar controle de acesso, proteção de endpoint e limites de consumo.
