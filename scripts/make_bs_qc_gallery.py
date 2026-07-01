# scripts/make_bs_qc_gallery.py
"""Build a static HTML gallery of the corrected Black Sea frames for the G1 gate:
gradual vs catastrophic vs wrong, per epoch, side by side."""
from pathlib import Path

BASE = Path(r"D:/pasha/RESEARCH/renders/black_sea_corrected/topo")
EPOCHS = [22, 18, 16.5, 14.5, 12, 11, 10, 9.5, 9, 7.5, 5, 2, 0]
OUT = BASE / "bs_qc_gallery.html"


def _img(p: Path) -> str:
    return f'<img src="{p.as_uri()}" style="width:32%;border:1px solid #ccc">' if p.exists() else \
           f'<div style="width:32%;color:#a11">missing {p.name}</div>'


def main() -> None:
    rows = []
    for e in EPOCHS:
        g = BASE / "correct" / "gradual" / f"paleo_render_{e}ka.png"
        c = BASE / "correct" / "catastrophic" / f"paleo_render_{e}ka.png"
        w = BASE / "wrong" / f"paleo_render_{e}ka.png"
        rows.append(
            f"<h3>{e} ka BP &nbsp; (left: WRONG global &middot; mid: gradual &middot; right: catastrophic)</h3>"
            f'<div style="display:flex;gap:6px">{_img(w)}{_img(g)}{_img(c)}</div>'
        )
    OUT.write_text(
        "<html><body style='font-family:sans-serif'>"
        "<h1>Black Sea (Pontic) — corrected paleo-topography QC</h1>" + "".join(rows) +
        "</body></html>",
        encoding="utf-8",
    )
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
