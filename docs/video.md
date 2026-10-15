# Apresentação da versão 2

[Abrir o vídeo](apresentacao.mp4) — **5 min 16,76 s**, medidos no MP4 final; abaixo do limite de 7 minutos. H.264, 1280×720, aproximadamente 10 fps, áudio AAC em português. O arquivo foi decodificado integralmente com FFmpeg, sem erro, e as 12 telas foram inspecionadas visualmente.

São apresentados problema de negócio, preparação, arquitetura, recuperação híbrida, geração real com Qwen local, fontes, análise de frequência, elogios, recusas, comparação experimental e reprodução. Os exemplos vêm de `evaluation/scenarios.json` e `evaluation/llm-demo.json`; as métricas vêm dos benchmarks pareados. O vídeo mostra **resultados exportados de execuções reais**, sem alegar captura contínua da interface. A interface possui validação separada por Streamlit AppTest.

A **narração sintética** está identificada em todas as telas. Não é a voz do aluno. A geração pequena demonstrada é limitada a achados com trechos verificáveis; não comprova síntese livre ou qualidade geral. Os temas analíticos são regras lexicais e as métricas de recuperação não são auditoria humana.

Texto completo: [video-narration.json](video-narration.json). Tempos por tela: [video-timings.json](video-timings.json). Para reconstruir, com o ambiente ativo e os resultados já gerados:

```text
python -m pip install -r requirements-media.txt
python scripts/build_video.py --work work/video
```

O script exige internet para a narração e baixa um FFmpeg pelo pacote de mídia quando necessário. Usa Segoe UI no Windows ou DejaVu Sans no Linux. `--slides-only` gera apenas as telas e o texto. A criação confere o limite de duração e recusa demonstração LLM com falha. Revalide duração, integridade e layout após qualquer reconstrução.

O [roteiro para gravação pessoal](roteiro-video.md) permite apresentar a interface aberta. A submissão na plataforma acadêmica é responsabilidade do aluno.
