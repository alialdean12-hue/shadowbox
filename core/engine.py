from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

Progress = Callable[[str, int, int], None] | None


def _progress(callback: Progress, message: str, current: int, total: int) -> None:
    if callback is not None:
        callback(str(message), int(current), max(1, int(total)))


def _safe_rgb(value) -> tuple[int, int, int]:
    data = [int(round(float(v))) for v in value]
    return tuple(max(0, min(255, v)) for v in data[:3])


def color_name(rgb: Iterable[int]) -> str:
    r, g, b = _safe_rgb(rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    chroma = mx - mn
    if lum < 38:
        return "Black"
    if chroma < 14:
        if lum < 92:
            return "Dark Gray"
        if lum < 178:
            return "Gray"
        if lum < 238:
            return "Light Gray"
        return "White"
    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
    hue = int(hsv[0]) * 2
    sat = int(hsv[1])
    if lum > 220 and sat < 65:
        return "Off White"
    if 10 <= hue < 42 and r > g and g > b:
        if lum > 185:
            return "Skin"
        if lum > 115:
            return "Tan"
        return "Brown"
    if hue < 18 or hue >= 345:
        return "Red"
    if hue < 48:
        return "Orange"
    if hue < 72:
        return "Yellow"
    if hue < 165:
        return "Green"
    if hue < 255:
        return "Blue"
    if hue < 315:
        return "Purple"
    return "Magenta"


def _load_rgb(path: str | Path) -> np.ndarray:
    """Load common raster formats safely, including Arabic/Unicode paths on Windows.

    OpenCV's ``imread`` can fail on Windows when the filename contains Arabic or
    other non-ASCII characters.  Reading the bytes first and using ``imdecode``
    avoids that problem.  Pillow is used as a second loader for TIFF, GIF,
    CMYK JPEG, EXIF rotation, and formats supported by installed Pillow plugins.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError("The selected image file does not exist.")
    if source.stat().st_size <= 0:
        raise ValueError("The selected image file is empty.")

    image = None
    errors = []

    # Unicode-safe OpenCV loading. This supports PNG, JPEG/JFIF, WebP, BMP,
    # TIFF and other codecs included in the installed OpenCV build.
    try:
        encoded = np.fromfile(str(source), dtype=np.uint8)
        if encoded.size:
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        errors.append(f"OpenCV: {exc}")

    if image is not None:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            bgr, alpha = image[:, :, :3], image[:, :, 3:4].astype(np.float32) / 255.0
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            corner_samples = []
            h, w = alpha.shape[:2]
            for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
                if alpha[y, x, 0] >= 0.1:
                    corner_samples.append(rgb[y, x])
            background = np.mean(corner_samples, axis=0) if corner_samples else np.array([255, 255, 255], np.float32)
            rgb = rgb * alpha + background.reshape(1, 1, 3) * (1.0 - alpha)
            return np.clip(rgb, 0, 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[2] >= 3:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)

    # Pillow fallback handles EXIF orientation, animated images (first frame),
    # CMYK JPEG, palette images, TIFF, and optional HEIC/HEIF/AVIF plugins.
    try:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass
        from PIL import Image, ImageOps
        with Image.open(source) as pil_image:
            try:
                pil_image.seek(0)
            except Exception:
                pass
            pil_image = ImageOps.exif_transpose(pil_image)
            rgba = pil_image.convert("RGBA")
            array = np.asarray(rgba, dtype=np.uint8)
            rgb = array[:, :, :3].astype(np.float32)
            alpha = array[:, :, 3:4].astype(np.float32) / 255.0
            corner_samples = []
            h, w = alpha.shape[:2]
            for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
                if alpha[y, x, 0] >= 0.1:
                    corner_samples.append(rgb[y, x])
            background = np.mean(corner_samples, axis=0) if corner_samples else np.array([255, 255, 255], np.float32)
            return np.clip(rgb * alpha + background.reshape(1, 1, 3) * (1.0 - alpha), 0, 255).astype(np.uint8)
    except Exception as exc:
        errors.append(f"Pillow: {exc}")

    suffix = source.suffix.lower() or "unknown"
    extra = ""
    if suffix in {".heic", ".heif"}:
        extra = " HEIC/HEIF requires the optional pillow-heif package."
    elif suffix == ".avif":
        extra = " AVIF support depends on the installed Pillow codec."
    detail = " | ".join(errors[-2:])
    raise ValueError(
        f"The selected image ({suffix}) could not be decoded.{extra}"
        + (f" Details: {detail}" if detail else "")
    )


def _resize_for_page(
    rgb: np.ndarray,
    page_width_mm: float,
    page_height_mm: float,
    margin_mm: float,
    max_dimension: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    page_width_mm = max(1.0, float(page_width_mm))
    page_height_mm = max(1.0, float(page_height_mm))
    max_dimension = max(500, min(1800, int(max_dimension)))
    if page_height_mm >= page_width_mm:
        page_h = max_dimension
        page_w = max(1, int(round(page_h * page_width_mm / page_height_mm)))
    else:
        page_w = max_dimension
        page_h = max(1, int(round(page_w * page_height_mm / page_width_mm)))
    px_per_mm = min(page_w / page_width_mm, page_h / page_height_mm)
    margin_px = max(0, int(round(float(margin_mm) * px_per_mm)))
    inner_w = max(8, page_w - 2 * margin_px)
    inner_h = max(8, page_h - 2 * margin_px)
    h, w = rgb.shape[:2]
    scale = min(inner_w / max(1, w), inner_h / max(1, h))
    fit_w = max(1, int(round(w * scale)))
    fit_h = max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (fit_w, fit_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x = (page_w - fit_w) // 2
    y = (page_h - fit_h) // 2
    return resized, (x, y, fit_w, fit_h), px_per_mm


def _lab_distance_ciede2000(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """Return CIEDE2000 perceptual distance between LAB arrays.

    The inputs may be shape (3,), (N, 3), or broadcast-compatible arrays.
    """
    a = np.asarray(lab_a, dtype=np.float64)
    b = np.asarray(lab_b, dtype=np.float64)
    L1, a1, b1 = np.moveaxis(a, -1, 0)
    L2, a2, b2 = np.moveaxis(b, -1, 0)

    avg_lp = (L1 + L2) * 0.5
    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    avg_c = (c1 + c2) * 0.5
    g = 0.5 * (1.0 - np.sqrt((avg_c ** 7) / ((avg_c ** 7) + (25.0 ** 7) + 1e-12)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.sqrt(a1p * a1p + b1 * b1)
    c2p = np.sqrt(a2p * a2p + b2 * b2)
    avg_cp = (c1p + c2p) * 0.5

    h1p = (np.degrees(np.arctan2(b1, a1p)) + 360.0) % 360.0
    h2p = (np.degrees(np.arctan2(b2, a2p)) + 360.0) % 360.0

    delta_lp = L2 - L1
    delta_cp = c2p - c1p

    dh = h2p - h1p
    dh = np.where(c1p * c2p == 0, 0.0, dh)
    dh = np.where(dh > 180.0, dh - 360.0, dh)
    dh = np.where(dh < -180.0, dh + 360.0, dh)
    delta_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) * 0.5)

    avg_hp = np.where(
        c1p * c2p == 0,
        h1p + h2p,
        np.where(
            np.abs(h1p - h2p) > 180.0,
            np.where(h1p + h2p < 360.0, (h1p + h2p + 360.0) * 0.5, (h1p + h2p - 360.0) * 0.5),
            (h1p + h2p) * 0.5,
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(avg_hp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * avg_hp))
        + 0.32 * np.cos(np.radians(3.0 * avg_hp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * avg_hp - 63.0))
    )
    delta_theta = 30.0 * np.exp(-(((avg_hp - 275.0) / 25.0) ** 2))
    rc = 2.0 * np.sqrt((avg_cp ** 7) / ((avg_cp ** 7) + (25.0 ** 7) + 1e-12))
    sl = 1.0 + (0.015 * ((avg_lp - 50.0) ** 2)) / np.sqrt(20.0 + ((avg_lp - 50.0) ** 2))
    sc = 1.0 + 0.045 * avg_cp
    sh = 1.0 + 0.015 * avg_cp * t
    rt = -np.sin(np.radians(2.0 * delta_theta)) * rc

    return np.sqrt(
        (delta_lp / sl) ** 2
        + (delta_cp / sc) ** 2
        + (delta_hp / sh) ** 2
        + rt * (delta_cp / sc) * (delta_hp / sh)
    )


def _relabel_sequential(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    unique = [int(v) for v in np.unique(labels)]
    mapping = {value: index for index, value in enumerate(unique)}
    if unique == list(range(len(unique))):
        return labels.astype(np.uint8), mapping
    out = np.empty_like(labels, dtype=np.uint8)
    for value, index in mapping.items():
        out[labels == value] = np.uint8(index)
    return out, mapping


def _merge_perceptual_colors(labels: np.ndarray, lab: np.ndarray, centers_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge colors that humans would perceive as effectively the same.

    This keeps the rest of the pipeline untouched and only changes the color
    detection result. It intentionally merges anti-aliased edge colors and very
    small, barely distinguishable palette entries back into their nearest main
    color. High-contrast small details (eyes, beard lines, nostrils, etc.) are
    preserved because their perceptual distance remains large.
    """
    merged_labels = labels.astype(np.int32, copy=True)
    lab_flat = lab.reshape(-1, 3).astype(np.float32)
    total = int(merged_labels.size)

    for _ in range(6):
        unique = [int(v) for v in np.unique(merged_labels)]
        counts = {idx: int(np.count_nonzero(merged_labels == idx)) for idx in unique}
        means = {}
        for idx in unique:
            mask = merged_labels.ravel() == idx
            if np.any(mask):
                means[idx] = lab_flat[mask].mean(axis=0)
            else:
                means[idx] = np.asarray(centers_lab[idx], dtype=np.float32)

        merge_from = None
        merge_to = None
        best_score = None

        # 1) Merge truly indistinguishable colors first.
        for pos, left in enumerate(unique):
            for right in unique[pos + 1:]:
                delta_e = float(_lab_distance_ciede2000(means[left], means[right]))
                if delta_e > 9.0:
                    continue
                combined = counts[left] + counts[right]
                score = (delta_e, combined)
                if best_score is None or score < best_score:
                    if counts[left] <= counts[right]:
                        merge_from, merge_to = left, right
                    else:
                        merge_from, merge_to = right, left
                    best_score = score

        # 2) If no direct merge exists, absorb very tiny, low-visibility colors.
        if merge_from is None:
            tiny_limit = max(24, int(total * 0.0035))
            very_tiny_limit = max(16, int(total * 0.0018))
            for left in unique:
                area = counts[left]
                if area > tiny_limit:
                    continue
                nearest = None
                nearest_delta = None
                for right in unique:
                    if right == left:
                        continue
                    delta_e = float(_lab_distance_ciede2000(means[left], means[right]))
                    if nearest_delta is None or delta_e < nearest_delta:
                        nearest = right
                        nearest_delta = delta_e
                if nearest is None:
                    continue
                limit = 11.5 if area <= very_tiny_limit else 8.5
                if nearest_delta is not None and nearest_delta <= limit:
                    score = (nearest_delta, area)
                    if best_score is None or score < best_score:
                        merge_from, merge_to = left, nearest
                        best_score = score

        if merge_from is None:
            break

        merged_labels[merged_labels == merge_from] = merge_to

    merged_labels, mapping = _relabel_sequential(merged_labels.astype(np.uint8))
    unique = [int(v) for v in np.unique(merged_labels)]
    final_centers = np.zeros((len(unique), 3), np.float32)
    for idx in unique:
        mask = merged_labels.ravel() == idx
        final_centers[idx] = lab_flat[mask].mean(axis=0)
    return merged_labels.astype(np.uint8), final_centers


def _edge_absorb_labels(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Absorb tiny connected components when their color is visually close.

    This only touches anti-aliased fringe noise; strong-contrast small details
    are left alone because the perceptual distance test fails.
    """
    out = labels.astype(np.uint8, copy=True)
    h, w = out.shape
    palette_lab = cv2.cvtColor(np.asarray(palette, dtype=np.uint8)[None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    base_limit = max(8, int(h * w * 0.00010))
    kernel = np.ones((3, 3), np.uint8)
    for color_index in [int(v) for v in np.unique(out)]:
        mask = (out == color_index).astype(np.uint8)
        count, component_map, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        for component_index in range(1, count):
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if area > base_limit:
                continue
            component = component_map == component_index
            ring = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool) & ~component
            neighbours = out[ring]
            neighbours = neighbours[neighbours != color_index]
            if neighbours.size == 0:
                continue
            values, counts = np.unique(neighbours, return_counts=True)
            ranked = sorted(zip(values.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True)
            target = None
            best_de = None
            for value, _count in ranked[:4]:
                delta_e = float(_lab_distance_ciede2000(palette_lab[color_index], palette_lab[int(value)]))
                if best_de is None or delta_e < best_de:
                    best_de = delta_e
                    target = int(value)
            if target is not None and best_de is not None and best_de <= 9.5:
                out[component] = np.uint8(target)
    return out


def _absorb_microscopic_components(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Reassign individual sub-visible fragments without deleting geometry.

    A tiny coherent feature may be meaningful, so this pass only targets
    genuinely microscopic components: very low pixel area and very short span.
    """
    out = labels.astype(np.uint8, copy=True)
    h, w = out.shape
    total_pixels = int(h * w)
    palette = np.asarray(palette, dtype=np.uint8)
    palette_lab = cv2.cvtColor(palette[None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)

    micro_area = max(8, int(round(total_pixels * 0.000012)))
    small_area = max(16, int(round(total_pixels * 0.000024)))
    micro_span = max(5, int(round(math.sqrt(total_pixels) * 0.005)))

    for _pass in range(2):
        changed = False
        snapshot = out.copy()
        for color_index in [int(v) for v in np.unique(snapshot)]:
            mask = (snapshot == color_index).astype(np.uint8)
            count, component_map, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
            for component_index in range(1, count):
                x, y, bw, bh, area = [int(v) for v in stats[component_index]]
                microscopic = area <= micro_area
                tiny_compact = area <= small_area and max(bw, bh) <= micro_span
                if not (microscopic or tiny_compact):
                    continue

                component = component_map == component_index
                ring = cv2.dilate(component.astype(np.uint8), kernel, iterations=2).astype(bool) & ~component
                neighbours = snapshot[ring]
                neighbours = neighbours[neighbours != color_index]
                if neighbours.size == 0:
                    continue

                values, contacts = np.unique(neighbours, return_counts=True)
                target = None
                best_score = None
                for value, contact in zip(values.tolist(), contacts.tolist()):
                    value = int(value)
                    delta_e = float(_lab_distance_ciede2000(palette_lab[color_index], palette_lab[value]))
                    score = (-int(contact), delta_e)
                    if best_score is None or score < best_score:
                        best_score = score
                        target = value
                if target is not None:
                    out[component] = np.uint8(target)
                    changed = True
        if not changed:
            break
    return out


def _absorb_invisible_palette_clusters(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Remove palette entries made only of invisible scattered specks.

    Pixels are never deleted. Every rejected speck is reassigned to the most
    appropriate neighbouring major paper color. A compact high-contrast detail
    can stay even when small; a color made from many 1-10 px fragments cannot
    become its own paper layer.
    """
    out = labels.astype(np.uint8, copy=True)
    h, w = out.shape
    total_pixels = int(h * w)
    palette = np.asarray(palette, dtype=np.uint8)
    palette_lab = cv2.cvtColor(palette[None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)

    # Scale thresholds with image size. At the standard A4 working size these
    # are roughly 250 total pixels and 45 pixels for one coherent component.
    min_total_area = max(70, int(round(total_pixels * 0.00025)))
    min_coherent_area = max(18, int(round(total_pixels * 0.000045)))
    very_tiny_area = max(24, int(round(total_pixels * 0.00008)))

    for _pass in range(4):
        unique = [int(v) for v in np.unique(out)]
        if len(unique) <= 2:
            break

        candidates: list[int] = []
        component_data: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
        for color_index in unique:
            mask = (out == color_index).astype(np.uint8)
            count, component_map, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
            areas = stats[1:, cv2.CC_STAT_AREA].astype(int) if count > 1 else np.asarray([], dtype=int)
            total_area = int(areas.sum())
            largest_area = int(areas.max()) if areas.size else 0
            component_count = int(areas.size)
            component_data[color_index] = (count, component_map, stats)

            # Keep a rare color when it forms one clearly visible coherent
            # feature. Reject it when its total area is tiny, or when almost all
            # of it is fragmented into microscopic disconnected dots.
            fragmented = (
                total_area < min_total_area * 2
                and component_count >= 4
                and largest_area < min_coherent_area
            )
            microscopic = total_area <= very_tiny_area
            invisible_cluster = total_area < min_total_area and largest_area < min_coherent_area
            if microscopic or fragmented or invisible_cluster:
                candidates.append(color_index)

        if not candidates:
            break

        changed = False
        candidate_set = set(candidates)
        # Major colors are preferred as destinations. If all remaining colors
        # are candidates, keep the largest one and absorb the rest into it.
        major_colors = [idx for idx in unique if idx not in candidate_set]
        if not major_colors:
            counts = {idx: int(np.count_nonzero(out == idx)) for idx in unique}
            keep = max(unique, key=lambda idx: counts[idx])
            major_colors = [keep]
            candidates = [idx for idx in candidates if idx != keep]

        for color_index in candidates:
            count, component_map, stats = component_data[color_index]
            for component_index in range(1, count):
                component = component_map == component_index
                area = int(stats[component_index, cv2.CC_STAT_AREA])
                if area <= 0:
                    continue

                # Use a two-pixel ring so anti-alias fragments can find the
                # actual shape color even when surrounded by another speck.
                ring = cv2.dilate(component.astype(np.uint8), kernel, iterations=2).astype(bool) & ~component
                neighbours = out[ring]
                neighbours = neighbours[(neighbours != color_index) & np.isin(neighbours, major_colors)]

                target = None
                if neighbours.size:
                    values, contacts = np.unique(neighbours, return_counts=True)
                    best_score = None
                    for value, contact in zip(values.tolist(), contacts.tolist()):
                        value = int(value)
                        delta_e = float(_lab_distance_ciede2000(palette_lab[color_index], palette_lab[value]))
                        # Neighbour contact dominates; Delta E resolves ties.
                        score = (-int(contact), delta_e)
                        if best_score is None or score < best_score:
                            best_score = score
                            target = value
                else:
                    # Borderless isolated speck: use the closest major color.
                    best_delta = None
                    for value in major_colors:
                        delta_e = float(_lab_distance_ciede2000(palette_lab[color_index], palette_lab[value]))
                        if best_delta is None or delta_e < best_delta:
                            best_delta = delta_e
                            target = int(value)

                if target is not None:
                    out[component] = np.uint8(target)
                    changed = True

        if not changed:
            break

    return out


def _kmeans_quantize(rgb: np.ndarray, color_count: int, seed: int = 41) -> tuple[np.ndarray, np.ndarray]:
    color_count = max(2, min(12, int(color_count)))
    # Edge-preserving denoise removes gradient noise but keeps facial lines.
    filtered = cv2.bilateralFilter(rgb, d=7, sigmaColor=18, sigmaSpace=6)
    lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    total = len(pixels)
    rng = np.random.default_rng(seed)
    sample_count = min(120_000, total)
    if sample_count < total:
        sample = pixels[rng.choice(total, sample_count, replace=False)]
    else:
        sample = pixels
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.25)
    _compactness, _sample_labels, centers = cv2.kmeans(
        sample,
        color_count,
        None,
        criteria,
        7,
        cv2.KMEANS_PP_CENTERS,
    )
    # Assign in chunks to avoid a large W*H*K temporary allocation on A3.
    labels_flat = np.empty(total, np.uint8)
    chunk = 160_000
    for start in range(0, total, chunk):
        values = pixels[start : start + chunk]
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels_flat[start : start + len(values)] = distances.argmin(axis=1).astype(np.uint8)
    labels = labels_flat.reshape(lab.shape[:2])

    # v1.2: merge colors using perceptual Delta E rather than treating every
    # technical k-means center as a human-visible paper color.
    labels, centers_lab = _merge_perceptual_colors(labels, lab, centers)
    centers_u8 = np.uint8(np.clip(centers_lab, 0, 255))[None, :, :]
    palette = cv2.cvtColor(centers_u8, cv2.COLOR_LAB2RGB)[0]
    labels = _edge_absorb_labels(labels, palette.astype(np.uint8))
    labels = _absorb_invisible_palette_clusters(labels, palette.astype(np.uint8))
    labels, mapping = _relabel_sequential(labels)
    palette = palette[[old for old, _new in sorted(mapping.items(), key=lambda item: item[1])]].astype(np.uint8)
    return labels.astype(np.uint8), palette.astype(np.uint8)


def _clean_labels(labels: np.ndarray) -> np.ndarray:
    out = cv2.medianBlur(labels.astype(np.uint8), 3)
    h, w = out.shape
    minimum = max(2, int(round(h * w * 0.000003)))
    kernel = np.ones((3, 3), np.uint8)
    # Only remove true specks. Meaningful disconnected details remain intact.
    for _pass in range(2):
        for color_index in np.unique(out):
            mask = (out == color_index).astype(np.uint8)
            count, component_map, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
            for component_index in range(1, count):
                area = int(stats[component_index, cv2.CC_STAT_AREA])
                if area >= minimum:
                    continue
                component = component_map == component_index
                ring = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool) & ~component
                neighbours = out[ring]
                neighbours = neighbours[neighbours != color_index]
                if neighbours.size:
                    values, counts = np.unique(neighbours, return_counts=True)
                    out[component] = values[int(np.argmax(counts))]
    return out


def _border_background_label(labels: np.ndarray) -> int:
    h, w = labels.shape
    band = max(1, int(round(min(h, w) * 0.012)))
    border = np.concatenate(
        [
            labels[:band, :].ravel(),
            labels[-band:, :].ravel(),
            labels[:, :band].ravel(),
            labels[:, -band:].ravel(),
        ]
    )
    values, counts = np.unique(border, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _place_on_page(
    fitted_rgb: np.ndarray,
    fitted_labels: np.ndarray,
    fit_rect: tuple[int, int, int, int],
    page_shape: tuple[int, int],
    background_label: int,
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    page_h, page_w = page_shape
    x, y, fit_w, fit_h = fit_rect
    page_labels = np.full((page_h, page_w), int(background_label), dtype=np.uint8)
    page_rgb = np.empty((page_h, page_w, 3), dtype=np.uint8)
    page_rgb[:] = palette[int(background_label)]
    page_labels[y : y + fit_h, x : x + fit_w] = fitted_labels
    page_rgb[y : y + fit_h, x : x + fit_w] = palette[fitted_labels]
    return page_labels, page_rgb


def _components_and_adjacency(labels: np.ndarray) -> tuple[np.ndarray, list[dict], dict[int, dict[int, int]], int]:
    h, w = labels.shape
    component_map = np.full((h, w), -1, np.int32)
    components: list[dict] = []
    next_id = 0
    for color_index in [int(v) for v in np.unique(labels)]:
        mask = (labels == color_index).astype(np.uint8)
        count, local, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for local_id in range(1, count):
            area = int(stats[local_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            component_map[local == local_id] = next_id
            x, y, bw, bh, _ = [int(v) for v in stats[local_id]]
            components.append(
                {
                    "id": next_id,
                    "color_index": color_index,
                    "area": area,
                    "bbox": [x, y, bw, bh],
                    "centroid": [float(centroids[local_id, 0]), float(centroids[local_id, 1])],
                    "depth": 0,
                }
            )
            next_id += 1

    adjacency: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for first, second in ((component_map[:, :-1], component_map[:, 1:]), (component_map[:-1, :], component_map[1:, :])):
        changed = (first != second) & (first >= 0) & (second >= 0)
        if not np.any(changed):
            continue
        pairs = np.stack([first[changed], second[changed]], axis=1)
        pairs.sort(axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (left, right), contact in zip(unique_pairs, counts):
            left_i, right_i = int(left), int(right)
            adjacency[left_i][right_i] += int(contact)
            adjacency[right_i][left_i] += int(contact)

    # The real background is the component occupying the largest amount of the page border.
    border_ids = np.concatenate(
        [component_map[0, :], component_map[-1, :], component_map[:, 0], component_map[:, -1]]
    )
    border_ids = border_ids[border_ids >= 0]
    values, counts = np.unique(border_ids, return_counts=True)
    root_id = int(values[int(np.argmax(counts))]) if values.size else 0

    depths = {root_id: 0}
    queue = deque([root_id])
    while queue:
        current = queue.popleft()
        for neighbour in adjacency.get(current, {}):
            if neighbour not in depths:
                depths[neighbour] = depths[current] + 1
                queue.append(neighbour)
    # The map is fully colored, so unreachable components are unusual. Keep them
    # behind the closest broad layer rather than deleting them.
    for component in components:
        component["depth"] = int(depths.get(component["id"], 1))
    return component_map, components, adjacency, root_id


def _layer_mask(component_map: np.ndarray, component_ids: list[int]) -> np.ndarray:
    if not component_ids:
        return np.zeros(component_map.shape, np.uint8)
    return np.isin(component_map, np.asarray(component_ids, dtype=np.int32)).astype(np.uint8)


def build_sheet_masks(layers: list[dict], component_map: np.ndarray) -> list[np.ndarray]:
    layer_masks = [_layer_mask(component_map, list(layer.get("component_ids", []))) for layer in layers]
    deeper_union = np.zeros(component_map.shape, np.uint8)
    sheets: list[np.ndarray] = [np.zeros_like(deeper_union) for _ in layers]
    for index in range(len(layers) - 1, -1, -1):
        sheets[index] = (deeper_union == 0).astype(np.uint8)
        deeper_union = np.maximum(deeper_union, layer_masks[index])
    return sheets


def floating_islands(sheet_mask: np.ndarray) -> list[dict]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(sheet_mask.astype(np.uint8), 8)
    h, w = sheet_mask.shape
    result = []
    for index in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[index]]
        touches = x <= 0 or y <= 0 or x + bw >= w or y + bh >= h
        if not touches:
            result.append({"area": area, "bbox": [x, y, bw, bh]})
    return result


def validate_layers(layers: list[dict], component_map: np.ndarray) -> dict:
    sheets = build_sheet_masks(layers, component_map)
    per_layer = [floating_islands(sheet) for sheet in sheets]
    return {
        "safe": not any(per_layer),
        "floating_island_count": sum(len(items) for items in per_layer),
        "per_layer": per_layer,
    }


def _initial_layers(components: list[dict], component_map: np.ndarray, palette: np.ndarray) -> list[dict]:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    area_by_group: dict[tuple[int, int], int] = defaultdict(int)
    for component in components:
        key = (int(component["color_index"]), int(component["depth"]))
        grouped[key].append(int(component["id"]))
        area_by_group[key] += int(component["area"])
    keys = sorted(grouped, key=lambda key: (key[1], -area_by_group[key], key[0]))
    layers = []
    for serial, key in enumerate(keys, start=1):
        color_index, depth = key
        layers.append(
            {
                "id": f"layer_{serial:03d}",
                "color_index": int(color_index),
                "depth_hint": int(depth),
                "component_ids": list(grouped[key]),
                "area": int(area_by_group[key]),
            }
        )
    return layers


def _merge_layers(layers: list[dict], first: int, second: int, target: str, component_map: np.ndarray) -> list[dict]:
    if first > second:
        first, second = second, first
    target_index = second if target == "deeper" else first
    merged_ids = list(layers[first]["component_ids"]) + list(layers[second]["component_ids"])
    output = []
    for index, layer in enumerate(layers):
        if index in (first, second):
            if index == target_index:
                merged = deepcopy(layer)
                merged["component_ids"] = merged_ids
                merged["area"] = int(_layer_mask(component_map, merged_ids).sum())
                merged["depth_hint"] = (
                    max(int(layers[first].get("depth_hint", 0)), int(layers[second].get("depth_hint", 0)))
                    if target == "deeper"
                    else min(int(layers[first].get("depth_hint", 0)), int(layers[second].get("depth_hint", 0)))
                )
                output.append(merged)
            continue
        output.append(deepcopy(layer))
    return output


def compact_layers(layers: list[dict], component_map: np.ndarray) -> list[dict]:
    """Merge repeated-color sheets only when physical safety remains exact.

    This is the key rule the earlier experiments violated: repeated colors are
    allowed.  The optimizer removes a repeat only when doing so creates no
    detached paper island anywhere in the stack.
    """
    current = deepcopy(layers)
    while True:
        best = None
        for first in range(len(current)):
            for second in range(first + 1, len(current)):
                if int(current[first]["color_index"]) != int(current[second]["color_index"]):
                    continue
                for target in ("deeper", "shallower"):
                    candidate = _merge_layers(current, first, second, target, component_map)
                    validation = validate_layers(candidate, component_map)
                    if not validation["safe"]:
                        continue
                    combined_area = int(current[first].get("area", 0)) + int(current[second].get("area", 0))
                    # Prefer absorbing tiny repeat groups; deeper is safer for reveal details.
                    score = (combined_area, 0 if target == "deeper" else 1, second - first)
                    if best is None or score < best[0]:
                        best = (score, candidate)
        if best is None:
            break
        current = best[1]
    return current


def _rename_layers(layers: list[dict], palette: np.ndarray) -> None:
    repetitions: dict[int, int] = defaultdict(int)
    totals: dict[int, int] = defaultdict(int)
    for layer in layers:
        totals[int(layer["color_index"])] += 1
    for index, layer in enumerate(layers):
        color_index = int(layer["color_index"])
        repetitions[color_index] += 1
        name = color_name(palette[color_index])
        repeat = f" {repetitions[color_index]}" if totals[color_index] > 1 else ""
        if index == 0:
            role = "Front Background"
        elif index == len(layers) - 1:
            role = "Solid Backing"
        elif int(layer.get("depth_hint", 0)) >= 2:
            role = "Reveal Details"
        else:
            role = "Middle Color"
        layer["name"] = f"Layer {index + 1:02d} — {name}{repeat} — {role}"
        layer["is_backing"] = index == len(layers) - 1
        layer["rgb"] = [int(v) for v in palette[color_index]]
        layer["id"] = f"layer_{index + 1:03d}"


def render_composite(layers: list[dict], sheets: list[np.ndarray], palette: np.ndarray) -> np.ndarray:
    if not layers:
        raise ValueError("No layers are available.")
    h, w = sheets[0].shape
    image = np.zeros((h, w, 3), np.uint8)
    covered = np.zeros((h, w), bool)
    for layer, sheet in zip(layers, sheets):
        visible = sheet.astype(bool) & ~covered
        image[visible] = palette[int(layer["color_index"])]
        covered |= visible
    return image


def analyze_image(
    source_path: str | Path,
    page_width_mm: float,
    page_height_mm: float,
    color_count: int = 6,
    margin_mm: float = 8.0,
    max_dimension: int = 1200,
    progress: Progress = None,
) -> dict:
    _progress(progress, "Loading image", 1, 9)
    source_rgb = _load_rgb(source_path)
    fitted_rgb, fit_rect, px_per_mm = _resize_for_page(
        source_rgb, page_width_mm, page_height_mm, margin_mm, max_dimension
    )
    page_w = max(1, int(round(page_width_mm * px_per_mm)))
    page_h = max(1, int(round(page_height_mm * px_per_mm)))

    _progress(progress, "Detecting real paper colors", 2, 9)
    labels, palette = _kmeans_quantize(fitted_rgb, color_count)
    _progress(progress, "Preserving fine color details", 3, 9)
    labels = _clean_labels(labels)
    # Final human-visibility pass after median cleanup. This is intentionally
    # still part of color detection; all later layer/cut/export code is unchanged.
    for _visibility_pass in range(3):
        before = labels.copy()
        labels = _absorb_microscopic_components(labels, palette)
        labels = _absorb_invisible_palette_clusters(labels, palette)
        if np.array_equal(labels, before):
            break
    labels, mapping = _relabel_sequential(labels)
    palette = palette[[old for old, _new in sorted(mapping.items(), key=lambda item: item[1])]].astype(np.uint8)
    background_label = _border_background_label(labels)
    page_labels, quantized_rgb = _place_on_page(
        fitted_rgb,
        labels,
        fit_rect,
        (page_h, page_w),
        background_label,
        palette,
    )

    _progress(progress, "Separating disconnected vector components", 4, 9)
    component_map, components, adjacency, root_id = _components_and_adjacency(page_labels)
    _progress(progress, "Creating safe repeated-color depth layers", 5, 9)
    layers = _initial_layers(components, component_map, palette)
    initial_validation = validate_layers(layers, component_map)
    if not initial_validation["safe"]:
        # Shortest-path depth normally starts safe. Keep the detailed stack if a
        # rare shape still has islands; never delete the involved components.
        pass
    _progress(progress, "Compacting only physically safe repeats", 6, 9)
    layers = compact_layers(layers, component_map)
    _rename_layers(layers, palette)

    _progress(progress, "Building cumulative cut sheets", 7, 9)
    sheets = build_sheet_masks(layers, component_map)
    validation = validate_layers(layers, component_map)
    composite = render_composite(layers, sheets, palette)

    _progress(progress, "Preparing previews", 8, 9)
    project = {
        "format_version": 1,
        "source_path": str(Path(source_path).resolve()),
        "page_width_mm": float(page_width_mm),
        "page_height_mm": float(page_height_mm),
        "margin_mm": float(margin_mm),
        "color_count": int(color_count),
        "working_width": int(page_w),
        "working_height": int(page_h),
        "px_per_mm": float(px_per_mm),
        "fit_rect": [int(v) for v in fit_rect],
        "palette": palette.astype(np.uint8),
        "labels": page_labels.astype(np.uint8),
        "component_map": component_map.astype(np.int32),
        "components": components,
        "adjacency": {int(k): {int(n): int(v) for n, v in values.items()} for k, values in adjacency.items()},
        "root_component_id": int(root_id),
        "layers": layers,
        "sheets": sheets,
        "composite": composite,
        "validation": validation,
        "quantized_rgb": quantized_rgb,
    }
    _progress(progress, "Finished", 9, 9)
    return project


def rebuild_project(project: dict) -> dict:
    result = deepcopy(project)
    palette = np.asarray(result["palette"], dtype=np.uint8)
    component_map = np.asarray(result["component_map"], dtype=np.int32)
    layers = deepcopy(result["layers"])
    # Remove empty non-backing layers quietly. Components are never deleted.
    layers = [layer for layer in layers if layer.get("component_ids")]
    _rename_layers(layers, palette)
    sheets = build_sheet_masks(layers, component_map)
    result["layers"] = layers
    result["sheets"] = sheets
    result["composite"] = render_composite(layers, sheets, palette)
    result["validation"] = validate_layers(layers, component_map)
    return result


def move_component(project: dict, component_id: int, target_layer_index: int) -> dict:
    result = deepcopy(project)
    component_id = int(component_id)
    target_layer_index = int(target_layer_index)
    if not (0 <= target_layer_index < len(result["layers"])):
        raise IndexError(target_layer_index)
    component = next((item for item in result["components"] if int(item["id"]) == component_id), None)
    if component is None:
        raise KeyError(component_id)
    target = result["layers"][target_layer_index]
    if int(target["color_index"]) != int(component["color_index"]):
        # A paper sheet has one color. Create a same-color sheet at the requested depth.
        target = {
            "id": "new_layer",
            "color_index": int(component["color_index"]),
            "depth_hint": int(target.get("depth_hint", target_layer_index)),
            "component_ids": [],
            "area": 0,
        }
        result["layers"].insert(target_layer_index, target)
    for layer in result["layers"]:
        layer["component_ids"] = [int(v) for v in layer.get("component_ids", []) if int(v) != component_id]
    target["component_ids"].append(component_id)
    return rebuild_project(result)


def reorder_layer(project: dict, source_index: int, target_index: int) -> dict:
    result = deepcopy(project)
    layers = result["layers"]
    source_index, target_index = int(source_index), int(target_index)
    if not (0 <= source_index < len(layers) and 0 <= target_index < len(layers)):
        raise IndexError((source_index, target_index))
    layer = layers.pop(source_index)
    layers.insert(target_index, layer)
    return rebuild_project(result)


def add_same_color_layer(project: dict, source_layer_index: int, insert_index: int | None = None) -> dict:
    result = deepcopy(project)
    source_layer_index = int(source_layer_index)
    if not (0 <= source_layer_index < len(result["layers"])):
        raise IndexError(source_layer_index)
    source = result["layers"][source_layer_index]
    new_layer = {
        "id": "new_layer",
        "color_index": int(source["color_index"]),
        "depth_hint": int(source.get("depth_hint", source_layer_index)) + 1,
        "component_ids": [],
        "area": 0,
    }
    position = source_layer_index + 1 if insert_index is None else int(insert_index)
    result["layers"].insert(max(0, min(len(result["layers"]), position)), new_layer)
    _rename_layers(result["layers"], np.asarray(result["palette"], dtype=np.uint8))
    return result


def component_layer_index(project: dict, component_id: int) -> int:
    """Return the physical layer that owns a component."""
    component_id = int(component_id)
    for index, layer in enumerate(project.get("layers", [])):
        if component_id in {int(value) for value in layer.get("component_ids", [])}:
            return int(index)
    return -1


def component_at_point(project: dict, x: int, y: int) -> int:
    """Return the component id at an image-space point, or -1 outside the page."""
    component_map = np.asarray(project["component_map"], dtype=np.int32)
    x, y = int(x), int(y)
    if not (0 <= y < component_map.shape[0] and 0 <= x < component_map.shape[1]):
        return -1
    return int(component_map[y, x])


def component_preview(project: dict, component_id: int) -> np.ndarray:
    """Dim the full design and highlight one component in clear red."""
    component_id = int(component_id)
    component_map = np.asarray(project["component_map"], dtype=np.int32)
    component = next(item for item in project["components"] if int(item["id"]) == component_id)
    base = np.asarray(project["composite"], dtype=np.uint8).copy()
    mask = component_map == component_id
    # Keep enough context to recognize the portrait while making the selected
    # component unmistakable. This is display-only and never alters export data.
    result = np.clip(base.astype(np.float32) * 0.30 + 22.0, 0, 255).astype(np.uint8)
    result[mask] = np.array([255, 0, 0], dtype=np.uint8)
    outline = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool) & ~mask
    result[outline] = np.array([255, 70, 70], dtype=np.uint8)
    x, y, width, height = [int(v) for v in component.get("bbox", [0, 0, 0, 0])]
    if width > 0 and height > 0:
        thickness = max(1, int(round(min(result.shape[:2]) / 350)))
        cv2.rectangle(result, (x, y), (max(x, x + width - 1), max(y, y + height - 1)), (255, 0, 0), thickness)
    return result


def component_focus_preview(project: dict, component_id: int, padding_ratio: float = 0.55) -> np.ndarray:
    """Create a zoomed context view for tiny selected components."""
    component_id = int(component_id)
    component = next(item for item in project["components"] if int(item["id"]) == component_id)
    full = component_preview(project, component_id)
    x, y, width, height = [int(v) for v in component.get("bbox", [0, 0, 1, 1])]
    h, w = full.shape[:2]
    pad = max(24, int(round(max(width, height) * float(padding_ratio))))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + width + pad), min(h, y + height + pad)
    if x1 <= x0 or y1 <= y0:
        return full
    return full[y0:y1, x0:x1].copy()
