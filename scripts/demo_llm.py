"""Salva uma execução real de geração local, sem credenciais."""
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voc.retrieval import Retriever
from voc.pipeline import answer_question
import voc.pipeline as pipeline


def main():
    os.environ.setdefault("LLM_BASE_URL", "local")
    os.environ.setdefault("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    retriever = Retriever("data/index")
    raw_generation = []
    original_generate = pipeline.generate_local
    def capture(model, messages):
        content = original_generate(model, messages)
        raw_generation.append(content)
        return content
    pipeline.generate_local = capture
    question = "O que os clientes relatam sobre atrasos na entrega?"
    start = time.monotonic()
    result = answer_question(retriever, question, mode="llm", score_max=2, k=3)
    output = {"question": question, "model": os.environ["LLM_MODEL"], "backend": os.environ["LLM_BASE_URL"],
              "quantized_linear_layers": os.environ.get("LLM_QUANTIZE", "0") == "1" if os.environ["LLM_BASE_URL"] == "local" else None,
              "elapsed_seconds": round(time.monotonic() - start, 2), "raw_generation": raw_generation, "result": result}
    Path("evaluation/llm-demo.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
