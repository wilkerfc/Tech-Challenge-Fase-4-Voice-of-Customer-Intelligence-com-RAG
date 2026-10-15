"""Smoke test da interface com índice real já construído."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest


def main():
    app = AppTest.from_file("app.py", default_timeout=180).run()
    assert not app.exception, str(app.exception)
    app.button[0].click().run()
    assert not app.exception, str(app.exception)
    assert len(app.session_state["result"]["evidence"]) >= 2
    app.text_area[0].set_value("Quais os temas mais frequentes nas avaliações com notas 1 e 2 em SP?")
    app.button[0].click().run()
    assert not app.exception, str(app.exception)
    result = app.session_state["result"]
    assert result["mode"] == "analytics" and result["analytics"]["denominator"] > 30
    assert result["resolved_query"]["filters"]["state"] == "SP"
    app.text_area[0].set_value("Qual a previsão do tempo amanhã?")
    app.button[0].click().run()
    assert app.session_state["result"]["insufficient"]
    Path("evaluation/ui-smoke.json").write_text(json.dumps({"initial_render": "passed", "real_search": "passed", "analytics_real_corpus": "passed", "natural_language_filters": "passed", "out_of_scope": "passed", "method": "Streamlit AppTest, sem captura visual de navegador"}, indent=2), encoding="utf-8")
    print("Interface: renderização, busca real e recusa verificadas.")


if __name__ == "__main__":
    main()
