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
    prompts = []
    original_generate = pipeline.generate_local
    def capture(model, messages):
        prompts.append(messages)
        content = original_generate(model, messages)
        raw_generation.append(content)
        return content
    pipeline.generate_local = capture
    question = "O que os clientes relatam sobre atrasos na entrega?"
    start = time.monotonic()
    result = answer_question(retriever, question, mode="llm", score_max=2, k=3)
    output = {"question": question, "model": os.environ["LLM_MODEL"], "backend": os.environ["LLM_BASE_URL"],
              "model_revision": os.environ.get("LLM_REVISION") or ("7ae557604adf67be50417f59c2c2f167def9a775" if os.environ["LLM_MODEL"] == "Qwen/Qwen2.5-0.5B-Instruct" else None),
              "quantized_linear_layers": os.environ.get("LLM_QUANTIZE", "0") == "1" if os.environ["LLM_BASE_URL"] == "local" else None,
              "elapsed_seconds": round(time.monotonic() - start, 2), "prompts": prompts,
              "generation_config": {"do_sample": False, "max_new_tokens": 180, "cpu_threads": int(os.environ.get("VOC_CPU_THREADS", "1"))},
              "manifest": retriever.manifest, "raw_generation": raw_generation, "result": result}
    Path("evaluation/llm-demo.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
