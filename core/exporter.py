from __future__ import annotations

import io
from xml.sax.saxutils import escape
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.colors import Color

from core.engine import _bridge_kernel, color_name, floating_islands, project_px_per_mm, weak_bridges


def safe_name(value: str) -> str:
    import re
    value = re.sub(r"[^A-Za-z0-9_\-\u0600-\u06FF]+", "_", str(value or "Project").strip())
    return value.strip("_") or "Project"


def _smooth_closed_points(points: np.ndarray, passes: int = 4) -> np.ndarray:
    """Smooth a closed contour without changing its topology.

    The exporter now works from a supersampled mask and applies a circular
    Gaussian-like filter plus a Catmull-Rom-to-Bézier conversion. This removes
    stair-stepping from both SVG and PNG while keeping meaningful details.
    """
    pts = np.asarray(points, dtype=np.float64)
    count = len(pts)
    if count < 5:
        return pts
    passes = max(1, min(int(passes), 1 if count < 14 else 2 if count < 28 else 3 if count < 64 else 4))
    for _ in range(passes):
        pts = (
            np.roll(pts, 2, axis=0)
            + 4.0 * np.roll(pts, 1, axis=0)
            + 6.0 * pts
            + 4.0 * np.roll(pts, -1, axis=0)
            + np.roll(pts, -2, axis=0)
        ) / 16.0
    return pts


def _supersampled_binary(mask: np.ndarray, supersample: int = 4) -> np.ndarray:
    """Build a higher-resolution binary mask with smoother boundaries.

    This keeps the layer logic untouched and only improves the raster used for
    vector tracing and PNG alpha generation.
    """
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if supersample <= 1:
        return binary
    h, w = binary.shape[:2]
    up = cv2.resize(binary, (w * supersample, h * supersample), interpolation=cv2.INTER_CUBIC)
    # Blur in supersampled space so the threshold falls between pixel steps
    # instead of following every staircase edge from the low-resolution mask.
    up = cv2.GaussianBlur(
        up,
        (0, 0),
        sigmaX=0.58 * supersample,
        sigmaY=0.58 * supersample,
        borderType=cv2.BORDER_REPLICATE,
    )
    _t, up = cv2.threshold(up, 127, 255, cv2.THRESH_BINARY)
    return up


def _vectorize_mask(mask: np.ndarray, epsilon: float = 0.80, supersample: int = 4) -> tuple[list[np.ndarray], int, int]:
    """Extract smooth closed contours and map them back to the original mask space."""
    binary = _supersampled_binary(mask, supersample=supersample)
    found, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    result: list[np.ndarray] = []
    for contour in found:
        if len(contour) < 3:
            continue
        points = contour.reshape(-1, 2).astype(np.float64)
        points = _smooth_closed_points(points, passes=4)
        if len(points) >= 8:
            simplified = cv2.approxPolyDP(
                points.astype(np.float32).reshape(-1, 1, 2),
                float(epsilon * supersample),
                True,
            )
            points = simplified.reshape(-1, 2).astype(np.float64)
            points = _smooth_closed_points(points, passes=2)
        if len(points) >= 3:
            # Return the contour in the original low-resolution mask coordinate
            # space so SVG/PDF page dimensions remain exact A4.
            result.append(points / float(supersample))
    orig_h, orig_w = mask.shape[:2]
    return result, int(orig_w), int(orig_h)


def _vectorize_holes(mask: np.ndarray, epsilon: float = 0.80, supersample: int = 4) -> tuple[list[np.ndarray], int, int]:
    """Vectorize only the internal openings (holes), excluding the outside page border."""
    inv = (np.asarray(mask, dtype=np.uint8) == 0).astype(np.uint8)
    binary = _supersampled_binary(inv, supersample=supersample)
    found, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    H, W = binary.shape[:2]
    result: list[np.ndarray] = []
    for contour in found:
        if len(contour) < 3:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            # This is the outside-of-page background, not a cut opening.
            continue
        points = contour.reshape(-1, 2).astype(np.float64)
        points = _smooth_closed_points(points, passes=4)
        if len(points) >= 8:
            simplified = cv2.approxPolyDP(
                points.astype(np.float32).reshape(-1, 1, 2),
                float(epsilon * supersample),
                True,
            )
            points = simplified.reshape(-1, 2).astype(np.float64)
            points = _smooth_closed_points(points, passes=2)
        if len(points) >= 3:
            result.append(points / float(supersample))
    orig_h, orig_w = mask.shape[:2]
    return result, int(orig_w), int(orig_h)


def _curve_commands(contour: np.ndarray, sx: float, sy: float, invert_y: float | None = None) -> list[str]:
    """Convert one closed contour to smooth cubic Bézier commands."""
    pts = np.asarray(contour, dtype=np.float64).copy()
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    if invert_y is not None:
        pts[:, 1] = float(invert_y) - pts[:, 1]
    count = len(pts)
    first = pts[0]
    commands = [f"M {first[0]:.4f} {first[1]:.4f}"]
    if count < 4:
        commands.extend(f"L {point[0]:.4f} {point[1]:.4f}" for point in pts[1:])
        commands.append("Z")
        return commands
    for index in range(count):
        p0 = pts[(index - 1) % count]
        p1 = pts[index]
        p2 = pts[(index + 1) % count]
        p3 = pts[(index + 2) % count]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        commands.append(
            f"C {c1[0]:.4f} {c1[1]:.4f} {c2[0]:.4f} {c2[1]:.4f} {p2[0]:.4f} {p2[1]:.4f}"
        )
    commands.append("Z")
    return commands


def _svg_path(contours: list[np.ndarray], width_px: int, height_px: int, width_mm: float, height_mm: float) -> str:
    sx = float(width_mm) / max(1.0, float(width_px))
    sy = float(height_mm) / max(1.0, float(height_px))
    return " ".join(" ".join(_curve_commands(contour, sx, sy)) for contour in contours)


def _antialiased_alpha(mask: np.ndarray, width_px: int, height_px: int) -> np.ndarray:
    """Create a very smooth antialiased alpha channel from a binary sheet.

    PNG now uses a soft high-resolution coverage map instead of a thresholded
    binary map. This specifically improves the jagged inner cut edges.
    """
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    h, w = binary.shape[:2]
    # Blur by half a mask pixel, then scale straight to the output size. The
    # old path went through a 12x-supersampled copy of the whole page first,
    # which needed several GB of memory for A3 at 1200 DPI.
    soft = cv2.GaussianBlur(binary, (0, 0), sigmaX=0.50, sigmaY=0.50, borderType=cv2.BORDER_REPLICATE)
    alpha = cv2.resize(soft, (int(width_px), int(height_px)), interpolation=cv2.INTER_CUBIC)
    output_per_mask = max(1.0, float(width_px) / max(1, w))
    cv2.GaussianBlur(
        alpha, (0, 0), sigmaX=0.32 * output_per_mask, sigmaY=0.32 * output_per_mask,
        dst=alpha, borderType=cv2.BORDER_REPLICATE,
    )
    alpha[alpha < 2] = 0
    alpha[alpha > 253] = 255
    return alpha


def _rasterize_cut_sheet(hole_contours: list[np.ndarray], width_px: int, height_px: int, scale: int = 4) -> np.ndarray:
    """Rasterize the traced sheet (page minus holes, even-odd) at ``scale``x."""
    sheet = np.ones((height_px * scale, width_px * scale), np.uint8)
    for contour in hole_contours:
        points = np.round(np.asarray(contour, dtype=np.float64) * scale).astype(np.int32)
        x, y, w, h = cv2.boundingRect(points)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(sheet.shape[1], x + w + 1), min(sheet.shape[0], y + h + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        stamp = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(stamp, [points - np.array([x0, y0], np.int32)], 1)
        sheet[y0:y1, x0:x1] ^= stamp
    return sheet


def check_traced_sheet(mask: np.ndarray, px_per_mm: float | None = None) -> dict:
    """Check the paper the cutter will actually produce from the SVG path.

    Validation runs on the pixel sheet, but the exported outline is smoothed
    and simplified first, which can pinch or close a narrow bridge. This
    re-checks the traced shape for floating pieces and for bridges narrower
    than the minimum width.
    """
    scale = 4
    hole_contours, width_px, height_px = _vectorize_holes(mask)
    sheet = _rasterize_cut_sheet(hole_contours, width_px, height_px, scale)
    kernel = _bridge_kernel(px_per_mm * scale if px_per_mm else None)
    return {
        "floating": len(floating_islands(sheet)),
        "weak": len(weak_bridges(sheet, kernel)),
    }


def write_svg(path: Path, mask: np.ndarray, rgb, width_mm: float, height_mm: float, title: str) -> None:
    hole_contours, vector_w, vector_h = _vectorize_holes(mask)
    hole_path = _svg_path(hole_contours, vector_w, vector_h, width_mm, height_mm)
    outer_rect = f"M 0 0 L {width_mm:.4f} 0 L {width_mm:.4f} {height_mm:.4f} L 0 {height_mm:.4f} Z"
    path_data = f"{outer_rect} {hole_path}".strip()
    color = "#%02x%02x%02x" % tuple(int(v) for v in rgb)
    text = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm:g}mm" height="{height_mm:g}mm" '
        f'viewBox="0 0 {width_mm:g} {height_mm:g}">\n'
        f'  <title>{escape(str(title))}</title>\n'
        f'  <path d="{path_data}" fill="{color}" fill-rule="evenodd"/>\n'
        '</svg>\n'
    )
    path.write_text(text, encoding="utf-8")


def write_png(path: Path, mask: np.ndarray, rgb, width_mm: float, height_mm: float, dpi: int = 1200) -> None:
    width_px = max(1, int(round(float(width_mm) / 25.4 * dpi)))
    height_px = max(1, int(round(float(height_mm) / 25.4 * dpi)))
    alpha = _antialiased_alpha(mask, width_px, height_px)
    # One color at 256 opacity levels: an indexed PNG whose palette entries
    # all share the layer color and differ only in transparency. It displays
    # exactly like the RGBA version at a quarter of the memory.
    alpha = np.ascontiguousarray(alpha, dtype=np.uint8)
    image = Image.frombuffer("P", (alpha.shape[1], alpha.shape[0]), alpha, "raw", "P", 0, 1)
    color = [int(v) for v in np.asarray(rgb, dtype=np.uint8)[:3]]
    image.putpalette(color * 256)
    image.save(path, dpi=(dpi, dpi), transparency=bytes(range(256)))
    del alpha


def write_composite_png(path: Path, composite: np.ndarray, width_mm: float, height_mm: float, dpi: int = 1200) -> None:
    width_px = max(1, int(round(float(width_mm) / 25.4 * dpi)))
    height_px = max(1, int(round(float(height_mm) / 25.4 * dpi)))
    resized = cv2.resize(composite.astype(np.uint8), (width_px, height_px), interpolation=cv2.INTER_LANCZOS4)
    Image.fromarray(resized, "RGB").save(path, dpi=(dpi, dpi))


def write_pdf(path: Path, project: dict) -> None:
    width_mm = float(project["page_width_mm"])
    height_mm = float(project["page_height_mm"])
    points_per_mm = 72.0 / 25.4
    width_pt, height_pt = width_mm * points_per_mm, height_mm * points_per_mm
    palette = np.asarray(project["palette"], dtype=np.uint8)
    pdf = pdf_canvas.Canvas(str(path), pagesize=(width_pt, height_pt), pageCompression=1)
    for layer, mask in zip(project["layers"], project["sheets"]):
        rgb = palette[int(layer["color_index"])] / 255.0
        pdf.setFillColor(Color(float(rgb[0]), float(rgb[1]), float(rgb[2])))
        path_obj = pdf.beginPath()
        contours, vector_w, vector_h = _vectorize_mask(mask)
        sx = width_pt / vector_w
        sy = height_pt / vector_h
        for contour in contours:
            pts = np.asarray(contour, dtype=np.float64).copy()
            pts[:, 0] *= sx
            pts[:, 1] = height_pt - pts[:, 1] * sy
            count = len(pts)
            first = pts[0]
            path_obj.moveTo(float(first[0]), float(first[1]))
            if count < 4:
                for point in pts[1:]:
                    path_obj.lineTo(float(point[0]), float(point[1]))
            else:
                for index in range(count):
                    p0 = pts[(index - 1) % count]
                    p1 = pts[index]
                    p2 = pts[(index + 1) % count]
                    p3 = pts[(index + 2) % count]
                    c1 = p1 + (p2 - p0) / 6.0
                    c2 = p2 - (p3 - p1) / 6.0
                    path_obj.curveTo(
                        float(c1[0]), float(c1[1]),
                        float(c2[0]), float(c2[1]),
                        float(p2[0]), float(p2[1]),
                    )
            path_obj.close()
        pdf.drawPath(path_obj, fill=1, stroke=0, fillMode=0)
        pdf.showPage()
    pdf.save()


def save_project(path: str | Path, project: dict, project_name: str, source_path: str | Path | None = None) -> Path:
    path = Path(path)
    arrays_buffer = io.BytesIO()
    np.savez_compressed(
        arrays_buffer,
        palette=np.asarray(project["palette"], dtype=np.uint8),
        labels=np.asarray(project["labels"], dtype=np.uint8),
        component_map=np.asarray(project["component_map"], dtype=np.int32),
    )
    metadata = {
        "format_version": 1,
        "project_name": str(project_name),
        "source_name": Path(source_path or project.get("source_path", "source.png")).name,
        "page_width_mm": float(project["page_width_mm"]),
        "page_height_mm": float(project["page_height_mm"]),
        "margin_mm": float(project.get("margin_mm", 8.0)),
        "color_count": int(project.get("color_count", len(project["palette"]))),
        "working_width": int(project["working_width"]),
        "working_height": int(project["working_height"]),
        "fit_rect": list(project.get("fit_rect", [])),
        "components": project["components"],
        "layers": project["layers"],
        "root_component_id": int(project.get("root_component_id", 0)),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.writestr("arrays.npz", arrays_buffer.getvalue())
        source = Path(source_path or project.get("source_path", ""))
        if source.is_file():
            archive.write(source, f"source/{source.name}")
    return path


def load_project(path: str | Path) -> tuple[dict, str]:
    from core.engine import rebuild_project
    path = Path(path)
    with zipfile.ZipFile(path, "r") as archive:
        metadata = json.loads(archive.read("project.json").decode("utf-8"))
        arrays = np.load(io.BytesIO(archive.read("arrays.npz")))
        project = {
            "format_version": int(metadata.get("format_version", 1)),
            "source_path": "",
            "page_width_mm": float(metadata["page_width_mm"]),
            "page_height_mm": float(metadata["page_height_mm"]),
            "margin_mm": float(metadata.get("margin_mm", 8.0)),
            "color_count": int(metadata.get("color_count", len(arrays["palette"]))),
            "working_width": int(metadata["working_width"]),
            "working_height": int(metadata["working_height"]),
            "fit_rect": list(metadata.get("fit_rect", [])),
            "palette": arrays["palette"].copy(),
            "labels": arrays["labels"].copy(),
            "component_map": arrays["component_map"].copy(),
            "components": list(metadata["components"]),
            "layers": list(metadata["layers"]),
            "root_component_id": int(metadata.get("root_component_id", 0)),
            "adjacency": {},
        }
        source_members = [name for name in archive.namelist() if name.startswith("source/") and not name.endswith("/")]
        if source_members:
            project["embedded_source_member"] = source_members[0]
    return rebuild_project(project), str(metadata.get("project_name", path.stem))


def export_package(project: dict, destination: str | Path, project_name: str, source_path: str | Path | None = None, progress=None) -> Path:
    destination = Path(destination)
    name = safe_name(project_name)
    root = destination / f"{name}_Export"
    serial = 2
    while root.exists():
        # Never delete a folder that is already there; write next to it.
        root = destination / f"{name}_Export_{serial}"
        serial += 1
    svg_dir = root / "SVG"
    png_dir = root / "PNG"
    pdf_dir = root / "PDF"
    preview_dir = root / "Preview"
    project_dir = root / "Project"
    zip_dir = root / "ZIP"
    for folder in (svg_dir, png_dir, pdf_dir, preview_dir, project_dir, zip_dir):
        folder.mkdir(parents=True, exist_ok=True)

    px_per_mm = project_px_per_mm(project)
    for index, (layer, mask) in enumerate(zip(project["layers"], project["sheets"]), start=1):
        traced = check_traced_sheet(mask, px_per_mm)
        if traced["floating"]:
            raise ValueError(
                f"Layer {index:02d}: the smoothed cut outline leaves {traced['floating']} "
                "loose paper piece(s). Move the involved part to another layer and export again."
            )

    palette = np.asarray(project["palette"], dtype=np.uint8)
    total = max(1, len(project["layers"]) * 2 + 4)
    step = 0
    def report(message):
        nonlocal step
        step += 1
        if progress:
            progress(message, step, total)

    for index, (layer, mask) in enumerate(zip(project["layers"], project["sheets"]), start=1):
        rgb = palette[int(layer["color_index"])]
        filename = f"{name}_Layer_{index:02d}"
        write_svg(svg_dir / f"{filename}.svg", mask, rgb, project["page_width_mm"], project["page_height_mm"], layer["name"])
        report(f"SVG layer {index}/{len(project['layers'])}")
        write_png(png_dir / f"{filename}.png", mask, rgb, project["page_width_mm"], project["page_height_mm"], 1200)
        report(f"PNG layer {index}/{len(project['layers'])}")

    write_pdf(pdf_dir / f"{name}_All_Layers.pdf", project)
    report("PDF created")
    write_composite_png(preview_dir / f"{name}_Final_Preview.png", np.asarray(project["composite"], dtype=np.uint8), project["page_width_mm"], project["page_height_mm"], 1200)
    report("Final preview created")
    save_project(project_dir / f"{name}.colorbox", project, project_name, source_path)
    report("Editable project saved")

    archive_path = zip_dir / f"{name}_Export.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in root.rglob("*"):
            if file.is_file() and file != archive_path:
                archive.write(file, file.relative_to(root))
    report("ZIP package created")
    return root
