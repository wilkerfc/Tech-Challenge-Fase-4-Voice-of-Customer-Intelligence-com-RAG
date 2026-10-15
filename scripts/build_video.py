"""Apresentação com resultados reais exportados e narração sintética declarada.

Requer requirements-media.txt. --work guarda intermediários fora dos entregáveis.
Os textos de demonstração vêm exclusivamente dos JSONs de avaliação executados.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import wave
from PIL import Image, ImageDraw, ImageFont
import edge_tts
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def slides():
    scenarios = read("evaluation/scenarios.json")
    cases = {c["id"]: c["result"] for c in scenarios["cases"]}
    llm = read("evaluation/llm-demo.json")
    if llm["result"]["status"] != "ok":
        raise ValueError("O vídeo exige uma demonstração LLM real aceita no contrato atual.")
    analytics = cases["S04"]["analytics"]
    source = llm["result"]["evidence"][0]
    items = []

    def add(title, body, narration, label="TECH CHALLENGE · FASE 4"):
        items.append(dict(title=title, body=body, narration=narration, label=label))

    add("Escutar. Investigar. Conferir.", ["Voz do Cliente · Olist", "RAG com recuperação híbrida e fontes verificáveis", "Wilker · Tech Challenge Fase 4"],
        "Este projeto transforma avaliações de compra em uma ferramenta de investigação da voz do cliente. O problema de negócio é encontrar reclamações e elogios em milhares de comentários sem perder a ligação com o relato original. Vamos percorrer a arquitetura e demonstrar resultados realmente executados. A apresentação utiliza narração sintética identificada e saídas exportadas da aplicação.")
    add("Da base oficial ao conhecimento", ["99.224 linhas de avaliações lidas", "56.537 sem título nem comentário", "42.687 documentos · 42.370 avaliações únicas", "Texto, nota e vínculos de pedido preservados"],
        "Usamos os arquivos oficiais da Olist. Das noventa e nove mil duzentas e vinte e quatro linhas, cinquenta e seis mil quinhentas e trinta e sete não têm texto. A preparação preserva quarenta e dois mil seiscentos e oitenta e sete documentos, correspondentes a quarenta e dois mil trezentos e setenta identificadores de avaliação. A limpeza remove HTML e espaços e mascara padrões de email e telefone, preservando acentos e negações. O notebook registra essas contagens e os hashes dos arquivos.")
    add("Arquitetura com duas rotas", ["CSV → limpeza → corpus completo + metadados", "Relatos → MiniLM + BM25 → RRF → contexto → LLM", "Achados → conferência do trecho e da fonte → resposta", "Frequência → corpus filtrado → contagens → resposta"],
        "A pergunta é normalizada e seus filtros são identificados. Investigações qualitativas seguem o RAG: recuperação, seleção, contexto, geração e validação. Perguntas de frequência seguem uma camada analítica que percorre o corpus completo. Essa separação evita estimar prevalência a partir dos poucos resultados mais parecidos. As duas rotas exportam resposta, filtros e avaliações de origem.")
    add("Recuperação e integridade", ["MiniLM multilíngue · 384 dimensões · cosseno exato", "BM25 + fusão RRF; textos repetidos removidos", "Nota, compra, categoria, UF e situação da entrega", "Corpus, documentos e vetores com SHA-256"],
        "O encoder multilíngue representa perguntas e documentos no mesmo espaço de trezentas e oitenta e quatro dimensões. A busca exata em NumPy é suficiente para este corpus. BM25 acrescenta sinais lexicais e a fusão RRF combina os rankings. O padrão usa busca híbrida sem MMR, escolhido no desenvolvimento. O índice valida hashes e rejeita inconsistências. Metadados podem ser atualizados sem reindexar apenas se textos, identificadores e ordem continuarem idênticos.")
    evidence = cases["S01"]["evidence"]
    add("Demonstração · investigar atrasos", [cases["S01"]["question"], *[f'[{e["citation"]}] {e["text"][:190]}' for e in evidence[:2]], "Resultado real · avaliações e scores exportados"],
        "Nesta investigação real, perguntamos o que os clientes relatam sobre demora na entrega. Os comentários exibidos foram recuperados pelo sistema. Cada um recebe um marcador de evidência, a nota e o identificador de origem. O modo extrativo permite inspecionar a busca antes de ativar a geração. Não se conclui que esses sejam os problemas mais frequentes apenas por estarem no topo.", "DEMONSTRAÇÃO · RECUPERAÇÃO EXECUTADA")
    add("Demonstração · geração com LLM", [llm["question"], "Modelo local: Qwen2.5 0,5B · greedy", llm["result"]["answer"].split("\n\nOs relatos")[0], "Saída bruta e achados validados em llm-demo.json"],
        "A mesma pergunta foi executada com o Qwen local. O modelo recebe as fontes e gera um achado com tema, citação e trecho. O sistema confere a fonte e publica somente texto comprovado literalmente, com uma descrição neutra. O exemplo pequeno ficou próximo de uma síntese extrativa, limitação registrada no relatório. Uma tentativa anterior falhou no formato e foi bloqueada, com o registro preservado. Essa execução comprova o fluxo gerativo, sem presumir qualidade geral do modelo.", "DEMONSTRAÇÃO · LLM REAL")
    add("Rastreabilidade até o pedido", [f'[{source["citation"]}] {source["text"][:180]}', f'review_id: {source["review_id"]}', f'order_id: {source["order_id"]}', "O texto relata uma experiência; metadados descrevem o pedido"],
        "O marcador da resposta leva ao texto, ao identificador da avaliação e aos pedidos associados. Uma avaliação pode estar ligada a mais de um pedido. Por isso todos os filtros precisam coincidir no mesmo pedido, e o resultado mostra quais satisfazem o recorte. A nota não comprova a causa do problema. Uma reclamação textual de atraso também não é igual a atraso calculado pelas datas de entrega.", "DEMONSTRAÇÃO · FONTES")
    rows = analytics["rows"][:3]
    add("Demonstração · frequência com base", ["Pergunta: temas frequentes · notas 1–2 · SP", f'Base: {analytics["denominator"]:,} avaliações textuais únicas'.replace(",", "."), *[f'{r["label"]}: {r["count"]} (' + f'{r["percentage"]:.2f}'.replace(".", ",") + '%)' for r in rows], "Temas lexicais podem se sobrepor; percentuais não se somam"],
        "Agora perguntamos pelos temas frequentes nas avaliações de notas um e dois em São Paulo. O sistema identifica os filtros e conta todas as avaliações textuais únicas do recorte. Os números mostrados são calculados, sem estimativa livre do modelo de linguagem. As regras de tema são transparentes e podem falhar ou se sobrepor. O denominador exclui avaliações sem texto e não representa todos os clientes. Grupos menores que trinta não publicam percentuais.", "DEMONSTRAÇÃO · ANÁLISE EXECUTADA")
    positive = cases["S03"]["evidence"]
    add("Demonstração · experiências positivas", [cases["S03"]["question"], "Recorte desta execução: notas 4–5", *[f'[{e["citation"]}] {e["text"][:170]}' for e in positive[:2]]],
        "A ferramenta também investiga elogios. Na pergunta sobre rapidez, esta execução filtra notas quatro e cinco. As evidências positivas vêm do mesmo corpus e preservam suas fontes. A interface permite refinar categoria, período e estado. Filtros da pergunta e dos controles são combinados por interseção; se entrarem em conflito, a aplicação pede correção em vez de ampliar o recorte.", "DEMONSTRAÇÃO · ELOGIOS")
    add("Demonstração · quando se abster", ["Entrega em 2099 → nenhuma evidência no recorte", "Previsão do tempo → fora do escopo", "Grupo pequeno → evidência insuficiente para percentuais", "Fonte ou trecho inválido → geração bloqueada"],
        "Sem avaliações relevantes, a solução se abstém. O período de dois mil e noventa e nove não tem suporte na base histórica. Previsão do tempo fica fora do escopo. Grupos analíticos pequenos não sustentam os percentuais adotados pelo protótipo. Falhas de modelo ou de contrato aparecem como falha de geração, mantendo as fontes disponíveis. A barreira de escopo é uma heurística limitada, não um detector universal.", "DEMONSTRAÇÃO · RECUSAS EXECUTADAS")
    add("Comparação e limites medidos", ["Desenvolvimento · 16 perguntas: 58,33% → 67,71%", "Reservadas · 8 perguntas: 77,08% → 81,25%", "Precisão silver@6 · legado denso → híbrido", "MRR reservado caiu: 0,9375 → 0,9167", "Regex automáticas; sem auditoria humana independente"],
        "Comparamos estratégias nas mesmas perguntas, filtros e vetores. A precisão lexical no desenvolvimento passou de cinquenta e oito vírgula trinta e três para sessenta e sete vírgula setenta e um por cento. Em oito perguntas reservadas, o híbrido alcançou oitenta e um vírgula vinte e cinco por cento, contra setenta e sete vírgula zero oito do legado. Algumas consultas pioraram e o MRR caiu. A relevância vem de regras lexicais, não de juízes humanos, portanto esses números não são precisão geral nem provam superioridade em outro conjunto.")
    add("Reprodução e decisões", ["Instalar → baixar CSVs → construir índice → abrir Streamlit", "Testes de filtros, contagens, integridade e geração", "Notebook, benchmark e cenários reais versionados", "Código, documentação e vídeo no repositório"],
        "A entrega reúne código, versões fixadas, notebook executado, testes e evidências reais. Para reproduzir, instale o ambiente, baixe os arquivos oficiais, construa o índice e abra o Streamlit. Configure o LLM para a etapa gerativa. As decisões priorizam fontes verificáveis e contagens com denominador claro. Avaliação humana independente, conjuntos maiores e revisão de dados pessoais continuam necessários para uso operacional. Os materiais estão preparados para a revisão e a apresentação acadêmica.")
    return items


def font(size, bold=False):
    windows = "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"
    linux = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(windows if Path(windows).exists() else linux, size)


def render(items, work):
    for i, slide in enumerate(items):
        img = Image.new("RGB", (1280, 720), "#0c1525")
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 14, 720), fill="#38d9b6")
        draw.text((60, 42), slide["label"], font=font(18, True), fill="#38d9b6")
        draw.text((60, 100), slide["title"], font=font(40, True), fill="white")
        y = 190
        body_font = font(26)
        for entry in slide["body"]:
            line, lines = "", []
            for word in entry.split():
                candidate = (line + " " + word).strip()
                if draw.textlength(candidate, font=body_font) > 1120:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
            for line in lines:
                draw.text((64, y), line, font=body_font, fill="#d9e4f5")
                y += 37
            y += 17
        if y > 650:
            raise ValueError(f"Slide {i+1} excede a área útil: {y}.")
        draw.text((60, 674), "OLIST · RESULTADOS EXPORTADOS · NARRAÇÃO SINTÉTICA", font=font(16), fill="#8294af")
        draw.text((1160, 674), f"{i+1:02}/{len(items):02}", font=font(16), fill="#8294af")
        img.save(work / f"slide-{i:02}.png")


async def build(items, work, output):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    durations = []
    for i, slide in enumerate(items):
        mp3, wav = work / f"audio-{i:02}.mp3", work / f"audio-{i:02}.wav"
        await edge_tts.Communicate(slide["narration"], "pt-BR-FranciscaNeural", rate="+8%").save(str(mp3))
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3), str(wav)], check=True)
        with wave.open(str(wav)) as audio:
            duration = audio.getnframes() / audio.getframerate() + .5
        durations.append(duration)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-framerate", "1", "-i", str(work / f"slide-{i:02}.png"),
                        "-i", str(wav), "-t", str(duration), "-c:v", "libx264", "-tune", "stillimage", "-preset", "veryfast", "-crf", "32", "-r", "10", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "48k", "-af", "apad", str(work / f"part-{i:02}.mp4")], check=True)
        print(f"Slide {i+1}/{len(items)}: {duration:.2f}s", flush=True)
    if sum(durations) > 419:
        raise ValueError("Apresentação ultrapassa o limite de 7 minutos; encurte a narração.")
    concat = work / "concat.txt"
    concat.write_text("\n".join(f"file 'part-{i:02}.mp4'" for i in range(len(items))), encoding="utf-8")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(output)], check=True)
    (ROOT / "docs/video-timings.json").write_text(json.dumps({"duration_seconds_estimated_from_audio": round(sum(durations), 2), "slides": durations}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--slides-only", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    items = slides()
    render(items, work)
    (ROOT / "docs/video-narration.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.slides_only:
        asyncio.run(build(items, work, ROOT / "docs/apresentacao.mp4"))


if __name__ == "__main__":
    main()
