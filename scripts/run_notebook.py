"""Run this project's Python-only notebook without Jupyter dependencies.

This is a sequential Python executor, not a Jupyter kernel. It supports ordinary
Python cells and stream/error outputs; IPython magics and rich displays are not
implemented. Use the same interpreter as the application environment.
"""
import argparse
import contextlib
import io
import json
import os
import platform
import sys
import traceback
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path,
                        default=project / "notebooks" / "01_exploracao_e_decisoes.ipynb")
    parser.add_argument("--raw", type=Path,
                        help="Override the notebook's VOC_RAW_DIR (default: project data/raw).")
    parser.add_argument("--output", type=Path,
                        help="Save to another notebook; by default update the input in place.")
    args = parser.parse_args()
    path = args.notebook.resolve()
    output = args.output.resolve() if args.output else path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        parser.error("Expected a version 4 .ipynb notebook with a cells list.")
    if args.raw:
        os.environ["VOC_RAW_DIR"] = str(args.raw.resolve())
    os.chdir(project)
    namespace = {"__name__": "__main__"}
    count, failed = 0, False
    # A failed rerun must not retain old successful outputs in later cells.
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        count += 1
        cell["execution_count"] = count
        stdout, stderr = io.StringIO(), io.StringIO()
        error_output = None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                source = "".join(cell["source"])
                exec(compile(source, f"{path.name}:cell-{count}", "exec"), namespace)
        except Exception as error:
            error_output = {"output_type": "error", "ename": type(error).__name__,
                            "evalue": str(error), "traceback": traceback.format_exc().splitlines()}
            failed = True
        for name, stream in (("stdout", stdout), ("stderr", stderr)):
            if stream.getvalue():
                cell["outputs"].append({"output_type": "stream", "name": name,
                                        "text": stream.getvalue().splitlines(keepends=True)})
        if error_output:
            cell["outputs"].append(error_output)
        print(f"Cell {count}: {'ERROR' if failed else 'OK'}", flush=True)
        if failed:
            print("\n".join(error_output["traceback"]), file=sys.stderr)
            break
    metadata = notebook.setdefault("metadata", {})
    metadata.setdefault("language_info", {})["version"] = platform.python_version()
    metadata["execution_note"] = (
        "Células Python executadas sequencialmente por scripts/run_notebook.py "
        "com compile/exec e o interpretador do ambiente do projeto; sem magics/IPython. "
        "Saídas reais de stdout/stderr; não representa anotação humana nem avaliação de relevância."
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Saved {output.name}; executed {count} code cells.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
