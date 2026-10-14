# Dados de entrada

Execute `python scripts/download_data.py` na raiz do projeto. O script baixa os CSVs oficiais e registra seus hashes em `provenance.json`. Os dados brutos e índices não devem ser commitados.

`olist_order_reviews_dataset.csv` é obrigatório. Os arquivos `olist_orders_dataset.csv`, `olist_products_dataset.csv` e `olist_order_items_dataset.csv` habilitam os metadados. Se não forem fornecidos, notas continuam disponíveis, mas os demais filtros podem não encontrar resultados.
