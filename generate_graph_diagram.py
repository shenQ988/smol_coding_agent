"""Generate a visual representation of the agent's LangGraph structure."""

from agent.graph import build_graph


def main(graph):
    compiled_graph = graph.get_graph()

    # ASCII — quick terminal check
    print("=== ASCII ===")
    print(compiled_graph.draw_ascii())

    # Mermaid — paste into README.md in a ```mermaid block, GitHub renders it natively
    print("\n=== Mermaid ===")
    mermaid_code = compiled_graph.draw_mermaid()
    print(mermaid_code)

    with open("docs/graph.mmd", "w") as f:
        f.write(mermaid_code)
    print("\nSaved Mermaid source to docs/graph.mmd")

    # PNG — actual image file for embedding in README or docs
    try:
        png_bytes = compiled_graph.draw_mermaid_png()
        with open("docs/graph.png", "wb") as f:
            f.write(png_bytes)
        print("Saved PNG to docs/graph.png")
    except Exception as e:
        print(f"PNG generation failed: {e}")
        print("Try: pip install pyppeteer  (draw_mermaid_png needs a headless browser)")


if __name__ == "__main__":
    graph, db_conn = build_graph(provider="anthropic", model="claude-sonnet-4-6")
    try:
        main(graph)
    finally:
        db_conn.close()