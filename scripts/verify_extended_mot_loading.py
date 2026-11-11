"""Verify the merged data and execute all three delivered loading notebooks."""

from base64 import b64decode
from hashlib import sha256
from html.parser import HTMLParser

import nbformat
from nbclient import NotebookClient

import report_extended_mot_loading as report


class EmbeddedImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag != "img":
            return
        src = dict(attrs).get("src", "")
        if not src.startswith("data:image/png;base64,"):
            raise ValueError("Report references an external figure")
        self.images.append(b64decode(src.split(",", 1)[1], validate=True))


def notebook_images(book):
    images = []
    for cell in book.cells:
        if cell.cell_type == "markdown":
            for attachment in cell.get("attachments", {}).values():
                if "image/png" in attachment:
                    images.append(sha256(b64decode(attachment["image/png"])).hexdigest())
        else:
            for output in cell.get("outputs", []):
                if "image/png" in output.get("data", {}):
                    images.append(sha256(b64decode(output["data"]["image/png"])).hexdigest())
    return images


def main():
    extension = report.extension
    document = report.read(report.OUTPUT / "manifest.json")
    for name, expected in document["source_sha256"].items():
        if extension.checksum(report.REPO / name) != expected:
            raise ValueError(f"Original frozen source changed: {name}")
    original = report.read(extension.ORIGINAL / "final_loading_results.json")
    merged = report.read(report.OUTPUT / "merged/final_loading_results.json")
    if any(merged[key] != row for key, row in original.items()) or len(merged) != 89:
        raise ValueError("Original records were not retained exactly")
    manifest = report.read(report.OUTPUT / "merged/pilot_manifest.json")
    for key in ("trap_power_isotropic", "trap_power_axial"):
        gradients = [abs(manifest["points"][i]["origin_surrogate_field_jacobian_g_per_cm"][2][2])
                     for i in manifest["scans"][key]]
        if len(gradients) != 32 or abs(max(gradients)-40) > 1e-7:
            raise ValueError("A trapping scan does not include all old and new settings")
    diagnosis = report.read(report.OUTPUT / "retention_diagnosis.json")
    physics = report.read(report.OUTPUT / "physical_validation.json")
    if not physics["passed"] or len(physics["core_stability_checks"]) != 20:
        raise ValueError("Extension force and stability validation is incomplete")
    if diagnosis["status"] != "passed":
        raise ValueError("Retention diagnosis failed")
    for row in diagnosis["results"].values():
        if (max(row["spectral_abscissae_s_inv"]) >= 0
                or any(item["agreed_category"] != 1 or max(item["final_distance_m"]) > 1e-4
                       for item in row["perturbed_returns"])):
            raise ValueError("Low-gradient equilibrium did not pass stability and return checks")
    notebooks = (
        (report.REPO / f"notebooks/mot_strange/{extension.ORIGINAL.name}.ipynb", 4),
        (report.NOTEBOOKS / report.COMPARISON_NOTEBOOK_NAMES["loading_three"], 4),
        (report.NOTEBOOKS / report.COMPARISON_NOTEBOOK_NAMES["loading_four"], 12),
    )
    results = []
    for path, count in notebooks:
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        saved = notebook_images(book)
        if len(saved) != count:
            raise ValueError(f"Unexpected embedded figure count in {path.name}")
        executed = NotebookClient(book, timeout=300, resources={"metadata": {"path": str(report.REPO)}}).execute()
        images = notebook_images(executed)
        # The strange-MOT notebook retains four static attachments and regenerates four figures.
        regenerated = images[-count:]
        if sorted(regenerated) != sorted(saved):
            raise ValueError(f"Clean notebook execution did not reproduce every figure in {path.name}")
        results.append({"path": str(path.relative_to(report.REPO)), "figures": count,
                        "sha256": extension.checksum(path)})
        print(f"Verified clean notebook {path.name}", flush=True)
    html_path = report.REPO / f"visualizations/mot_loading/{extension.ORIGINAL.name}.html"
    html = EmbeddedImages()
    html.feed(html_path.read_text(encoding="utf-8"))
    if len(html.images) != 4:
        raise ValueError("Strange-MOT HTML does not embed four figures")
    book = nbformat.read(notebooks[0][0], as_version=4)
    if sorted(sha256(img).hexdigest() for img in html.images) != sorted(notebook_images(book)):
        raise ValueError("Strange-MOT HTML and notebook disagree")
    pngs = sorted(report.FIGURES.glob("*.png"))
    if len(pngs) != 18:
        raise ValueError("Expected two full strange-MOT scans and sixteen comparison figures")
    extension.write_json(report.OUTPUT / "report_verification.json", {
        "passed": True, "original_records_unchanged": 69, "new_settings": 20,
        "notebooks": results, "html_sha256": extension.checksum(html_path),
        "figures": {str(p.relative_to(report.REPO)): extension.checksum(p) for p in pngs},
    })
    print("All extended loading artifacts verified", flush=True)


if __name__ == "__main__":
    main()
