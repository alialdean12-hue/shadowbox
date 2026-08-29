from __future__ import annotations

from copy import deepcopy
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QProcess, QProcessEnvironment, QStandardPaths, Qt, QTimer, QSize, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressDialog,
    QPushButton, QSpinBox, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from core.engine import (
    add_same_color_layer, component_at_point, component_focus_preview,
    component_layer_index, component_preview, move_component, rebuild_project,
    reorder_layer,
)
from core.exporter import load_project, save_project

APP_NAME = "Color Shadow Box Studio"
APP_VERSION = "1.6.2 Rebuilt — A4 SVG Page + Ultra DPI PNG"
ROOT_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
PAGE_PRESETS = {
    "A4 Portrait — 210 × 297 mm": (210.0, 297.0),
    "A4 Landscape — 297 × 210 mm": (297.0, 210.0),
    "A3 Portrait — 297 × 420 mm": (297.0, 420.0),
    "A3 Landscape — 420 × 297 mm": (420.0, 297.0),
}


def numpy_pixmap(image: np.ndarray) -> QPixmap:
    data = np.ascontiguousarray(image.astype(np.uint8))
    if data.ndim == 2:
        qimage = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_Grayscale8)
    elif data.shape[2] == 4:
        qimage = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGBA8888)
    else:
        qimage = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


def color_icon(rgb) -> QIcon:
    pixmap = QPixmap(30, 18)
    pixmap.fill(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return QIcon(pixmap)


def checker_sheet(mask: np.ndarray, rgb) -> np.ndarray:
    h, w = mask.shape
    tile = max(8, int(round(min(h, w) / 38)))
    yy, xx = np.indices((h, w))
    checker = ((xx // tile + yy // tile) % 2).astype(np.uint8)
    background = np.where(checker[..., None] == 0, 226, 194).astype(np.uint8)
    background = np.repeat(background, 3, axis=2) if background.shape[2] == 1 else background
    result = background.copy()
    result[mask.astype(bool)] = np.asarray(rgb, dtype=np.uint8)
    return result


def selected_layer_image(project: dict, component_id: int) -> np.ndarray:
    """Show the selected component in red inside its real physical layer."""
    component_id = int(component_id)
    layer_index = component_layer_index(project, component_id)
    if layer_index < 0:
        return np.asarray(project["composite"], dtype=np.uint8)
    layer = project["layers"][layer_index]
    palette = np.asarray(project["palette"], dtype=np.uint8)
    sheet = np.asarray(project["sheets"][layer_index], dtype=np.uint8)
    image = checker_sheet(sheet, palette[int(layer["color_index"])])
    component_map = np.asarray(project["component_map"], dtype=np.int32)
    mask = component_map == component_id
    image[mask] = np.array([255, 0, 0], dtype=np.uint8)
    outline = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool) & ~mask
    image[outline] = np.array([255, 70, 70], dtype=np.uint8)
    return image


class ClickableImageLabel(QLabel):
    """A preview label that reports clicks in original image coordinates."""
    imageClicked = Signal(int, int)

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._source_pixmap = None
        self._image_width = 0
        self._image_height = 0
        self.setAlignment(Qt.AlignCenter)

    def set_image(self, image: np.ndarray):
        array = np.asarray(image, dtype=np.uint8)
        self._image_height, self._image_width = array.shape[:2]
        self._source_pixmap = numpy_pixmap(array)
        self._update_scaled_pixmap()

    def clear_image(self, message=""):
        self._source_pixmap = None
        self._image_width = self._image_height = 0
        self.clear()
        if message:
            self.setText(message)

    def _update_scaled_pixmap(self):
        if self._source_pixmap is not None and not self._source_pixmap.isNull():
            self.setPixmap(self._source_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def mousePressEvent(self, event):
        if self._source_pixmap is None or self._image_width <= 0 or self._image_height <= 0:
            return super().mousePressEvent(event)
        scale = min(self.width() / self._image_width, self.height() / self._image_height)
        drawn_w = self._image_width * scale
        drawn_h = self._image_height * scale
        left = (self.width() - drawn_w) * 0.5
        top = (self.height() - drawn_h) * 0.5
        point = event.position()
        if not (left <= point.x() < left + drawn_w and top <= point.y() < top + drawn_h):
            return
        x = int((point.x() - left) / scale)
        y = int((point.y() - top) / scale)
        self.imageClicked.emit(max(0, min(self._image_width - 1, x)), max(0, min(self._image_height - 1, y)))


class WorkerJob:
    def __init__(self, parent, worker: Path, payload: dict, title: str, message: str):
        self.parent = parent
        self.worker = Path(worker)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="colorbox_"))
        self.input_path = self.temp_dir / "input.pkl"
        self.output_path = self.temp_dir / "output.pkl"
        self.log_path = self.temp_dir / "worker.log"
        self.progress_path = self.temp_dir / "progress.jsonl"
        self.progress_offset = 0
        self.frozen_mode = bool(getattr(sys, "frozen", False))
        with self.input_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=5)
        self.process = QProcess(parent)
        self.progress = QProgressDialog(message, "Cancel", 0, 100, parent)
        self.progress.setWindowTitle(title)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)
        self.progress.canceled.connect(self.cancel)
        self.buffer = ""
        self.output = []
        self.success_callback = None
        self.error_callback = None
        self.last_message = message
        self.idle_seconds = 0
        self.timer = QTimer(parent)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.finished)

    def start(self, success, error):
        self.success_callback = success
        self.error_callback = error
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", str(ROOT_DIR))
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        self.progress.show()
        self.timer.start()
        if self.frozen_mode:
            mode = "--analyze-worker" if self.worker.stem == "analyze_worker" else "--export-worker"
            self.process.start(
                sys.executable,
                [mode, str(self.input_path), str(self.output_path), str(self.progress_path)],
            )
        else:
            self.process.start(sys.executable, [str(self.worker), str(self.input_path), str(self.output_path)])

    def tick(self):
        if self.frozen_mode:
            self.read_progress_file()
        self.idle_seconds += 1
        if self.idle_seconds >= 15 and self.process.state() != QProcess.ProcessState.NotRunning:
            self.progress.setLabelText(
                f"{self.last_message}\nStill working safely in a separate process. "
                f"No new stage for {self.idle_seconds} seconds."
            )

    def handle_progress_payload(self, payload: dict):
        if payload.get("type") == "progress":
            current = int(payload.get("current", 0))
            total = max(1, int(payload.get("total", 1)))
            self.last_message = str(payload.get("message", "Working…"))
            self.idle_seconds = 0
            self.progress.setLabelText(self.last_message)
            self.progress.setValue(min(99, max(0, int(current * 100 / total))))
        elif payload.get("type") == "error":
            self.output.append(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_progress_file(self):
        if not self.progress_path.exists():
            return
        try:
            with self.progress_path.open("r", encoding="utf-8") as handle:
                handle.seek(self.progress_offset)
                data = handle.read()
                self.progress_offset = handle.tell()
        except Exception:
            return
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            self.output.append(line + "\n")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.handle_progress_payload(payload)

    def cancel(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(2500)

    def read_stderr(self):
        text = bytes(self.process.readAllStandardError()).decode("utf-8", "replace")
        if text:
            self.output.append(text)

    def read_stdout(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self.output.append(line + "\n")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.handle_progress_payload(payload)

    def finished(self, exit_code, exit_status):
        self.timer.stop()
        if self.frozen_mode:
            self.read_progress_file()
        self.read_stdout(); self.read_stderr()
        self.progress.close()
        try:
            self.log_path.write_text("".join(self.output), encoding="utf-8")
        except Exception:
            pass
        if exit_status == QProcess.ExitStatus.CrashExit or exit_code != 0 or not self.output_path.exists():
            message = "The isolated worker stopped before finishing."
            for line in reversed(self.output):
                try:
                    item = json.loads(line)
                    if item.get("type") == "error":
                        message = str(item.get("message") or message)
                        break
                except Exception:
                    pass
            if self.error_callback:
                self.error_callback(message, self.log_path)
            return
        try:
            with self.output_path.open("rb") as handle:
                result = pickle.load(handle)
        except Exception as exc:
            if self.error_callback:
                self.error_callback(f"Could not read the result: {exc}", self.log_path)
            return
        if self.success_callback:
            self.success_callback(result)


class ComponentsDialog(QDialog):
    def __init__(self, project: dict, selected_layer_index: int, selected_component_id: int | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Direct Component Control")
        self.resize(1240, 760)
        self.project = deepcopy(project)
        self.layer_index = max(0, min(len(self.project["layers"]) - 1, int(selected_layer_index)))
        self.selected_component_id = int(selected_component_id) if selected_component_id is not None else None
        self.history = [deepcopy(self.project)]
        self.history_index = 0
        self._refreshing = False

        root = QHBoxLayout(self)
        previews = QVBoxLayout()
        top_label = QLabel("Full design — click any colored part to select it")
        top_label.setObjectName("previewTitle")
        previews.addWidget(top_label)
        self.full_preview = ClickableImageLabel("Select a component")
        self.full_preview.setMinimumSize(650, 390)
        self.full_preview.setStyleSheet("background:#101823;border:1px solid #33465f;")
        self.full_preview.imageClicked.connect(self.preview_clicked)
        previews.addWidget(self.full_preview, 3)

        bottom = QHBoxLayout()
        layer_col = QVBoxLayout()
        layer_col.addWidget(QLabel("Selected part in its physical layer"))
        self.layer_preview = ClickableImageLabel("Layer preview")
        self.layer_preview.setMinimumSize(310, 250)
        self.layer_preview.setStyleSheet("background:#101823;border:1px solid #33465f;")
        self.layer_preview.imageClicked.connect(self.preview_clicked)
        layer_col.addWidget(self.layer_preview, 1)
        bottom.addLayout(layer_col, 1)

        focus_col = QVBoxLayout()
        focus_col.addWidget(QLabel("Zoomed selected part"))
        self.focus_preview = ClickableImageLabel("Part zoom")
        self.focus_preview.setMinimumSize(310, 250)
        self.focus_preview.setStyleSheet("background:#101823;border:1px solid #33465f;")
        focus_col.addWidget(self.focus_preview, 1)
        bottom.addLayout(focus_col, 1)
        previews.addLayout(bottom, 2)
        root.addLayout(previews, 3)

        side = QVBoxLayout()
        info = QLabel(
            "Click a part in the design or choose it from the list. The selected part is shown in red in the full design and in its real paper layer. Moving a part never changes its shape, color, or position."
        )
        info.setWordWrap(True)
        side.addWidget(info)
        side.addWidget(QLabel("Current layer"))
        self.layer_combo = QComboBox()
        self.layer_combo.currentIndexChanged.connect(self.layer_changed)
        side.addWidget(self.layer_combo)
        side.addWidget(QLabel("Parts in this layer"))
        self.components = QListWidget()
        self.components.currentItemChanged.connect(self.component_changed)
        side.addWidget(self.components, 1)
        self.selection_info = QLabel("No part selected")
        self.selection_info.setWordWrap(True)
        self.selection_info.setObjectName("status")
        side.addWidget(self.selection_info)
        side.addWidget(QLabel("Move selected part to this depth"))
        target_row = QHBoxLayout()
        self.target_combo = QComboBox()
        target_row.addWidget(self.target_combo, 1)
        move = QPushButton("Move Selected Part")
        move.clicked.connect(self.move_selected)
        target_row.addWidget(move)
        side.addLayout(target_row)
        undo_row = QHBoxLayout()
        undo = QPushButton("Undo")
        redo = QPushButton("Redo")
        undo.clicked.connect(self.undo)
        redo.clicked.connect(self.redo)
        undo_row.addWidget(undo); undo_row.addWidget(redo)
        side.addLayout(undo_row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        side.addWidget(buttons)
        root.addLayout(side, 2)
        self.refresh(self.selected_component_id)

    def refresh(self, preferred_component_id: int | None = None):
        self._refreshing = True
        current_layer = self.layer_index
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear(); self.target_combo.clear()
        for layer in self.project["layers"]:
            self.layer_combo.addItem(layer["name"])
            self.target_combo.addItem(layer["name"])
        self.layer_index = max(0, min(len(self.project["layers"]) - 1, current_layer))
        self.layer_combo.setCurrentIndex(self.layer_index)
        self.target_combo.setCurrentIndex(self.layer_index)
        self.layer_combo.blockSignals(False)
        self.components.blockSignals(True)
        self.components.clear()
        ids = set(int(v) for v in self.project["layers"][self.layer_index].get("component_ids", []))
        selected_row = -1
        for component in self.project["components"]:
            component_id = int(component["id"])
            if component_id not in ids:
                continue
            x, y, width, height = [int(v) for v in component.get("bbox", [0, 0, 0, 0])]
            item = QListWidgetItem(
                f"Part {component_id + 1:03d} — {int(component['area'])} px — {width}×{height}"
            )
            item.setData(Qt.UserRole, component_id)
            self.components.addItem(item)
            if preferred_component_id is not None and component_id == int(preferred_component_id):
                selected_row = self.components.count() - 1
        self.components.blockSignals(False)
        self._refreshing = False
        if self.components.count():
            self.components.setCurrentRow(selected_row if selected_row >= 0 else 0)
        else:
            self.selected_component_id = None
            self.full_preview.set_image(np.asarray(self.project["composite"], dtype=np.uint8))
            self.layer_preview.clear_image("This layer has no components")
            self.focus_preview.clear_image("No selected part")
            self.selection_info.setText("This layer has no parts.")

    def layer_changed(self, index):
        if self._refreshing:
            return
        if index >= 0:
            self.layer_index = index
            self.selected_component_id = None
            self.refresh()

    def preview_clicked(self, x: int, y: int):
        component_id = component_at_point(self.project, x, y)
        if component_id < 0:
            return
        layer_index = component_layer_index(self.project, component_id)
        if layer_index < 0:
            return
        self.layer_index = layer_index
        self.selected_component_id = component_id
        self.refresh(component_id)

    def component_changed(self, current, _previous):
        if current is None or self._refreshing:
            return
        component_id = int(current.data(Qt.UserRole))
        self.selected_component_id = component_id
        self.update_component_previews(component_id)

    def update_component_previews(self, component_id: int):
        component_id = int(component_id)
        layer_index = component_layer_index(self.project, component_id)
        if layer_index >= 0 and layer_index != self.layer_index:
            self.layer_index = layer_index
        component = next(item for item in self.project["components"] if int(item["id"]) == component_id)
        self.full_preview.set_image(component_preview(self.project, component_id))
        self.layer_preview.set_image(selected_layer_image(self.project, component_id))
        self.focus_preview.set_image(component_focus_preview(self.project, component_id))
        layer_name = self.project["layers"][layer_index]["name"] if layer_index >= 0 else "Unknown layer"
        self.selection_info.setText(
            f"Selected: Part {component_id + 1:03d}\nLayer: {layer_name}\nArea: {int(component['area'])} px"
        )
        if layer_index >= 0:
            self.target_combo.setCurrentIndex(layer_index)

    def commit(self, project, selected_component_id: int | None = None):
        self.project = project
        self.history = self.history[: self.history_index + 1]
        self.history.append(deepcopy(project))
        self.history_index += 1
        if selected_component_id is not None:
            self.selected_component_id = int(selected_component_id)
            self.layer_index = max(0, component_layer_index(project, self.selected_component_id))
        self.refresh(self.selected_component_id)

    def move_selected(self):
        item = self.components.currentItem()
        if item is None:
            return
        component_id = int(item.data(Qt.UserRole))
        before = deepcopy(self.project)
        try:
            changed = move_component(self.project, component_id, self.target_combo.currentIndex())
        except Exception as exc:
            QMessageBox.warning(self, "Move Component", str(exc)); return
        if not changed["validation"]["safe"]:
            QMessageBox.warning(
                self, "Unsafe move",
                "That move would create detached paper material. The part was returned to its previous layer."
            )
            self.project = before
            return
        self.layer_index = max(0, component_layer_index(changed, component_id))
        self.commit(changed, component_id)

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.project = deepcopy(self.history[self.history_index])
            if self.selected_component_id is not None:
                self.layer_index = max(0, component_layer_index(self.project, self.selected_component_id))
            self.refresh(self.selected_component_id)

    def redo(self):
        if self.history_index + 1 < len(self.history):
            self.history_index += 1
            self.project = deepcopy(self.history[self.history_index])
            if self.selected_component_id is not None:
                self.layer_index = max(0, component_layer_index(self.project, self.selected_component_id))
            self.refresh(self.selected_component_id)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — {APP_VERSION}")
        self.resize(1180, 760)
        self.setMinimumSize(900, 620)
        self.project = None
        self.source_path = ""
        self.project_path = ""
        self.active_job = None
        self.selected_component_id = None
        self.edit_history = []
        self.edit_history_index = -1
        self._syncing_selection = False
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        self.app_data = Path(app_data or (Path.home() / ".color_shadow_box_rebuilt")) / "rebuilt_v1_6_2"
        self.app_data.mkdir(parents=True, exist_ok=True)
        self.autosave_path = self.app_data / "autosave.colorbox"
        self.session_lock = self.app_data / "session.lock"
        previous_unclean = self.session_lock.exists()
        self.session_lock.write_text("running", encoding="utf-8")
        self.build_ui()
        self.apply_style()
        if previous_unclean and self.autosave_path.exists():
            answer = QMessageBox.question(
                self, "Restore Autosave",
                "The previous session did not close normally. Restore the last autosaved project?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.load_project_file(self.autosave_path, silent=True)

    def build_ui(self):
        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)
        self.setup_page = QWidget(); self.result_page = QWidget()
        self.pages.addWidget(self.setup_page); self.pages.addWidget(self.result_page)

        setup = QVBoxLayout(self.setup_page)
        setup.setContentsMargins(34, 28, 34, 28); setup.setSpacing(18)
        title = QLabel("Color Shadow Box Studio")
        title.setObjectName("title")
        setup.addWidget(title)
        subtitle = QLabel(
            "Rebuilt around one rule: no colored component is deleted. Repeated paper colors are allowed whenever they are needed to keep fine details physically cuttable."
        )
        subtitle.setObjectName("subtitle"); subtitle.setWordWrap(True)
        setup.addWidget(subtitle)
        card = QFrame(); card.setObjectName("card")
        form = QFormLayout(card); form.setContentsMargins(26, 24, 26, 24); form.setSpacing(14)
        self.project_name = QLineEdit("My_Color_Shadow_Box")
        form.addRow("Project name", self.project_name)
        source_row = QHBoxLayout()
        self.source_edit = QLineEdit(); self.source_edit.setReadOnly(True)
        browse = QPushButton("Choose Image"); browse.clicked.connect(self.choose_image)
        source_row.addWidget(self.source_edit, 1); source_row.addWidget(browse)
        form.addRow("Source image", source_row)
        self.page_preset = QComboBox(); self.page_preset.addItems(PAGE_PRESETS.keys())
        form.addRow("Paper size", self.page_preset)
        self.color_count = QSpinBox(); self.color_count.setRange(2, 10); self.color_count.setValue(6)
        self.color_count.setToolTip("Choose the intended number of solid paper colors. Six is a good starting point for portraits.")
        form.addRow("Solid colors", self.color_count)
        self.margin = QDoubleSpinBox(); self.margin.setRange(0.0, 25.0); self.margin.setDecimals(1); self.margin.setValue(8.0); self.margin.setSuffix(" mm")
        form.addRow("Page margin", self.margin)
        setup.addWidget(card)
        row = QHBoxLayout()
        open_project = QPushButton("Open Project"); open_project.clicked.connect(self.open_project)
        analyze = QPushButton("Analyze and Build Safe Layers"); analyze.setObjectName("primary"); analyze.clicked.connect(self.analyze)
        row.addWidget(open_project); row.addStretch(1); row.addWidget(analyze)
        setup.addLayout(row); setup.addStretch(1)

        result = QVBoxLayout(self.result_page)
        result.setContentsMargins(18, 16, 18, 16); result.setSpacing(12)
        top = QHBoxLayout()
        back = QPushButton("New / Settings"); back.clicked.connect(lambda: self.pages.setCurrentWidget(self.setup_page))
        self.status = QLabel("No project")
        self.status.setObjectName("status")
        save = QPushButton("Save Project"); save.clicked.connect(self.save_project_as)
        export = QPushButton("Export SVG / PNG / PDF"); export.setObjectName("primary"); export.clicked.connect(self.export)
        top.addWidget(back); top.addWidget(self.status, 1); top.addWidget(save); top.addWidget(export)
        result.addLayout(top)
        splitter = QSplitter(Qt.Horizontal)
        preview_card = QFrame(); preview_card.setObjectName("card")
        pv = QVBoxLayout(preview_card)
        preview_buttons = QHBoxLayout()
        composite_button = QPushButton("Final Composite"); composite_button.clicked.connect(self.show_composite)
        sheet_button = QPushButton("Selected Cut Sheet"); sheet_button.clicked.connect(self.show_selected_sheet)
        clear_part = QPushButton("Clear Part Selection"); clear_part.clicked.connect(self.clear_component_selection)
        preview_buttons.addWidget(composite_button); preview_buttons.addWidget(sheet_button); preview_buttons.addWidget(clear_part); preview_buttons.addStretch(1)
        pv.addLayout(preview_buttons)
        preview_split = QSplitter(Qt.Horizontal)
        full_panel = QWidget(); full_layout = QVBoxLayout(full_panel); full_layout.setContentsMargins(0, 0, 0, 0)
        self.full_preview_title = QLabel("Final design — click any part")
        full_layout.addWidget(self.full_preview_title)
        self.preview = ClickableImageLabel("Analyze an image")
        self.preview.setMinimumSize(300, 540)
        self.preview.setStyleSheet("background:#0b121d;border:1px solid #30445e;")
        self.preview.imageClicked.connect(self.preview_component_clicked)
        full_layout.addWidget(self.preview, 1)
        preview_split.addWidget(full_panel)
        layer_panel = QWidget(); layer_layout = QVBoxLayout(layer_panel); layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_preview_title = QLabel("Selected layer — selected part appears in red")
        layer_layout.addWidget(self.layer_preview_title)
        self.layer_preview = ClickableImageLabel("Select a layer or part")
        self.layer_preview.setMinimumSize(300, 540)
        self.layer_preview.setStyleSheet("background:#0b121d;border:1px solid #30445e;")
        self.layer_preview.imageClicked.connect(self.preview_component_clicked)
        layer_layout.addWidget(self.layer_preview, 1)
        preview_split.addWidget(layer_panel)
        preview_split.setStretchFactor(0, 1); preview_split.setStretchFactor(1, 1)
        pv.addWidget(preview_split, 1)
        splitter.addWidget(preview_card)
        side_card = QFrame(); side_card.setObjectName("card")
        side = QVBoxLayout(side_card)
        side.addWidget(QLabel("Physical layers — front to back"))
        self.layer_list = QListWidget(); self.layer_list.currentRowChanged.connect(self.layer_selection_changed)
        side.addWidget(self.layer_list, 2)
        move_row = QHBoxLayout()
        up = QPushButton("Move Layer Up"); down = QPushButton("Move Layer Down")
        up.clicked.connect(lambda: self.move_layer(-1)); down.clicked.connect(lambda: self.move_layer(1))
        move_row.addWidget(up); move_row.addWidget(down)
        side.addLayout(move_row)
        side.addWidget(QLabel("Parts in selected layer — click a part to highlight it"))
        self.component_list = QListWidget(); self.component_list.currentItemChanged.connect(self.main_component_changed)
        side.addWidget(self.component_list, 2)
        self.component_info = QLabel("No part selected")
        self.component_info.setWordWrap(True); self.component_info.setObjectName("status")
        side.addWidget(self.component_info)
        self.target_layer_combo = QComboBox()
        side.addWidget(self.target_layer_combo)
        part_row = QHBoxLayout()
        move_part = QPushButton("Move Selected Part"); move_part.clicked.connect(self.move_selected_component)
        undo_part = QPushButton("Undo"); undo_part.clicked.connect(self.undo_component_edit)
        redo_part = QPushButton("Redo"); redo_part.clicked.connect(self.redo_component_edit)
        part_row.addWidget(move_part); part_row.addWidget(undo_part); part_row.addWidget(redo_part)
        side.addLayout(part_row)
        advanced = QPushButton("Open Large Component Editor"); advanced.clicked.connect(self.open_components)
        side.addWidget(advanced)
        note = QLabel(
            "Direct selection is display-only until you move a part. Moving a part keeps its exact shape, color, and position; only its physical depth changes."
        )
        note.setWordWrap(True); note.setObjectName("subtitle")
        side.addWidget(note)
        splitter.addWidget(side_card)
        splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 2)
        result.addWidget(splitter, 1)

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#0c1626;color:#eef5ff;font-size:13px;}
            QLabel#title{font-size:28px;font-weight:700;color:#55c7ff;}
            QLabel#subtitle{color:#b8c8dc;}
            QLabel#status{padding:8px 12px;background:#13243a;border-radius:7px;}
            QFrame#card{background:#101d30;border:1px solid #263d59;border-radius:10px;}
            QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QListWidget{background:#0d192a;border:1px solid #35506e;border-radius:6px;padding:8px;color:#f3f7ff;}
            QPushButton{background:#17304d;border:1px solid #3b6289;border-radius:7px;padding:9px 14px;color:white;}
            QPushButton:hover{background:#214568;}
            QPushButton#primary{background:#1683bf;border-color:#55c7ff;font-weight:700;}
            QListWidget::item{padding:9px;border-bottom:1px solid #20334a;}
            QListWidget::item:selected{background:#1f527b;}
        """)

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose image",
            "",
            "Supported Images (*.png *.jpg *.jpeg *.jfif *.webp *.bmp *.dib *.tif *.tiff *.gif *.ppm *.pgm *.pbm *.pnm *.hdr *.exr *.avif *.heic *.heif);;All Files (*.*)",
        )
        if path:
            self.source_path = path; self.source_edit.setText(path)
            if self.project_name.text().strip() in ("", "My_Color_Shadow_Box"):
                self.project_name.setText(Path(path).stem)

    def analyze(self):
        if not self.source_path or not Path(self.source_path).is_file():
            QMessageBox.warning(self, "Choose Image", "Choose a source image first."); return
        page_width, page_height = PAGE_PRESETS[self.page_preset.currentText()]
        payload = {
            "source_path": self.source_path,
            "page_width_mm": page_width,
            "page_height_mm": page_height,
            "color_count": self.color_count.value(),
            "margin_mm": self.margin.value(),
            "max_dimension": 1200,
        }
        self.active_job = WorkerJob(self, ROOT_DIR / "workers" / "analyze_worker.py", payload, "Analyze Color Shadow Box", "Starting analysis…")
        self.active_job.start(self.analysis_finished, self.job_failed)

    def analysis_finished(self, project):
        self.project = project
        self.selected_component_id = None
        self.reset_edit_history()
        self.refresh_project()
        self.pages.setCurrentWidget(self.result_page)
        self.autosave()

    def job_failed(self, message, log_path):
        destination = self.app_data / "last_worker.log"
        try: shutil.copy2(log_path, destination)
        except Exception: pass
        QMessageBox.critical(self, "Operation Failed", f"{message}\n\nDiagnostic log: {destination}")

    def reset_edit_history(self):
        if self.project is None:
            self.edit_history = []
            self.edit_history_index = -1
            return
        self.edit_history = [deepcopy(self.project)]
        self.edit_history_index = 0

    def commit_project_edit(self, project: dict, selected_component_id: int | None = None):
        self.project = project
        self.edit_history = self.edit_history[: self.edit_history_index + 1]
        self.edit_history.append(deepcopy(project))
        self.edit_history_index += 1
        self.selected_component_id = int(selected_component_id) if selected_component_id is not None else None
        self.refresh_project()
        self.autosave()

    def refresh_project(self):
        if self.project is None:
            return
        previous_layer = self.layer_list.currentRow() if hasattr(self, "layer_list") else 0
        selected_layer = component_layer_index(self.project, self.selected_component_id) if self.selected_component_id is not None else -1
        desired_layer = selected_layer if selected_layer >= 0 else max(0, previous_layer)
        self._syncing_selection = True
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        self.target_layer_combo.clear()
        palette = np.asarray(self.project["palette"], dtype=np.uint8)
        for layer in self.project["layers"]:
            rgb = palette[int(layer["color_index"])]
            item = QListWidgetItem(color_icon(rgb), layer["name"])
            self.layer_list.addItem(item)
            self.target_layer_combo.addItem(layer["name"])
        if self.layer_list.count():
            desired_layer = max(0, min(self.layer_list.count() - 1, desired_layer))
            self.layer_list.setCurrentRow(desired_layer)
            self.target_layer_combo.setCurrentIndex(desired_layer)
        self.layer_list.blockSignals(False)
        self.populate_component_list(self.selected_component_id)
        self._syncing_selection = False
        validation = self.project.get("validation", {})
        if validation.get("safe", False):
            self.status.setText(f"{len(self.project['layers'])} layers • physically safe • 0 floating paper islands")
            self.status.setStyleSheet("color:#b8ffd2;")
        else:
            self.status.setText(f"Needs review • {validation.get('floating_island_count', 0)} floating paper islands")
            self.status.setStyleSheet("color:#ffd28a;")
        self.render_previews()

    def populate_component_list(self, preferred_component_id: int | None = None):
        if self.project is None:
            return
        layer_index = self.layer_list.currentRow()
        self.component_list.blockSignals(True)
        self.component_list.clear()
        selected_row = -1
        if 0 <= layer_index < len(self.project["layers"]):
            ids = {int(v) for v in self.project["layers"][layer_index].get("component_ids", [])}
            for component in self.project["components"]:
                component_id = int(component["id"])
                if component_id not in ids:
                    continue
                _x, _y, width, height = [int(v) for v in component.get("bbox", [0, 0, 0, 0])]
                item = QListWidgetItem(
                    f"Part {component_id + 1:03d} — {int(component['area'])} px — {width}×{height}"
                )
                item.setData(Qt.UserRole, component_id)
                self.component_list.addItem(item)
                if preferred_component_id is not None and component_id == int(preferred_component_id):
                    selected_row = self.component_list.count() - 1
        self.component_list.blockSignals(False)
        if selected_row >= 0:
            self.component_list.setCurrentRow(selected_row)
        else:
            self.component_list.setCurrentRow(-1)

    def set_preview(self, label: ClickableImageLabel, image):
        label.set_image(np.asarray(image, dtype=np.uint8))

    def render_previews(self):
        if self.project is None:
            return
        if self.selected_component_id is not None and component_layer_index(self.project, self.selected_component_id) >= 0:
            component_id = int(self.selected_component_id)
            layer_index = component_layer_index(self.project, component_id)
            component = next(item for item in self.project["components"] if int(item["id"]) == component_id)
            self.set_preview(self.preview, component_preview(self.project, component_id))
            self.set_preview(self.layer_preview, selected_layer_image(self.project, component_id))
            self.full_preview_title.setText(f"Selected Part {component_id + 1:03d} — red in the full design")
            self.layer_preview_title.setText(f"{self.project['layers'][layer_index]['name']} — selected part in red")
            self.component_info.setText(
                f"Selected Part {component_id + 1:03d}\n"
                f"Layer: {self.project['layers'][layer_index]['name']}\n"
                f"Area: {int(component['area'])} px"
            )
            self.target_layer_combo.setCurrentIndex(layer_index)
            return
        self.selected_component_id = None
        self.set_preview(self.preview, self.project["composite"])
        self.full_preview_title.setText("Final design — click any colored part")
        layer_index = self.layer_list.currentRow()
        if 0 <= layer_index < len(self.project["layers"]):
            layer = self.project["layers"][layer_index]
            palette = np.asarray(self.project["palette"], dtype=np.uint8)
            image = checker_sheet(np.asarray(self.project["sheets"][layer_index], dtype=np.uint8), palette[int(layer["color_index"])])
            self.set_preview(self.layer_preview, image)
            self.layer_preview_title.setText(layer["name"])
        else:
            self.layer_preview.clear_image("Select a layer")
        self.component_info.setText("No part selected. Click a part in either preview or choose it from the list.")

    def show_composite(self):
        if self.project is None:
            return
        self.clear_component_selection()

    def show_selected_sheet(self, *_args):
        if self.project is None:
            return
        self.render_previews()

    def clear_component_selection(self):
        self.selected_component_id = None
        self._syncing_selection = True
        self.component_list.setCurrentRow(-1)
        self._syncing_selection = False
        self.render_previews()

    def layer_selection_changed(self, index):
        if self.project is None or self._syncing_selection:
            return
        if not (0 <= index < len(self.project["layers"])):
            return
        self.selected_component_id = None
        self.target_layer_combo.setCurrentIndex(index)
        self.populate_component_list()
        self.render_previews()

    def main_component_changed(self, current, _previous):
        if current is None or self._syncing_selection:
            return
        self.select_component(int(current.data(Qt.UserRole)))

    def preview_component_clicked(self, x: int, y: int):
        if self.project is None:
            return
        component_id = component_at_point(self.project, x, y)
        if component_id >= 0:
            self.select_component(component_id)

    def select_component(self, component_id: int):
        if self.project is None:
            return
        component_id = int(component_id)
        layer_index = component_layer_index(self.project, component_id)
        if layer_index < 0:
            return
        self.selected_component_id = component_id
        self._syncing_selection = True
        self.layer_list.setCurrentRow(layer_index)
        self.target_layer_combo.setCurrentIndex(layer_index)
        self.populate_component_list(component_id)
        for row in range(self.component_list.count()):
            item = self.component_list.item(row)
            if int(item.data(Qt.UserRole)) == component_id:
                self.component_list.setCurrentRow(row)
                break
        self._syncing_selection = False
        self.render_previews()

    def move_layer(self, direction):
        if self.project is None:
            return
        source = self.layer_list.currentRow(); target = source + int(direction)
        if not (0 <= source < len(self.project["layers"]) and 0 <= target < len(self.project["layers"])):
            return
        before = deepcopy(self.project)
        changed = reorder_layer(self.project, source, target)
        if not changed["validation"]["safe"]:
            QMessageBox.warning(self, "Unsafe Order", "That order would create detached paper material, so it was not applied.")
            self.project = before
            return
        selected_id = self.selected_component_id
        self.commit_project_edit(changed, selected_id)
        if selected_id is None:
            self.layer_list.setCurrentRow(target)

    def move_selected_component(self):
        if self.project is None or self.selected_component_id is None:
            QMessageBox.information(self, "Move Part", "Select a part first.")
            return
        component_id = int(self.selected_component_id)
        target_index = self.target_layer_combo.currentIndex()
        before = deepcopy(self.project)
        try:
            changed = move_component(self.project, component_id, target_index)
        except Exception as exc:
            QMessageBox.warning(self, "Move Part", str(exc))
            return
        if not changed["validation"]["safe"]:
            self.project = before
            QMessageBox.warning(
                self, "Unsafe Move",
                "That move would create detached paper material. The part remains in its original layer."
            )
            return
        self.commit_project_edit(changed, component_id)

    def undo_component_edit(self):
        if self.edit_history_index > 0:
            self.edit_history_index -= 1
            self.project = deepcopy(self.edit_history[self.edit_history_index])
            if self.selected_component_id is not None and component_layer_index(self.project, self.selected_component_id) < 0:
                self.selected_component_id = None
            self.refresh_project(); self.autosave()

    def redo_component_edit(self):
        if self.edit_history_index + 1 < len(self.edit_history):
            self.edit_history_index += 1
            self.project = deepcopy(self.edit_history[self.edit_history_index])
            if self.selected_component_id is not None and component_layer_index(self.project, self.selected_component_id) < 0:
                self.selected_component_id = None
            self.refresh_project(); self.autosave()

    def open_components(self):
        if self.project is None:
            return
        dialog = ComponentsDialog(
            self.project,
            max(0, self.layer_list.currentRow()),
            self.selected_component_id,
            self,
        )
        if dialog.exec() == QDialog.Accepted:
            self.commit_project_edit(dialog.project, dialog.selected_component_id)

    def autosave(self):
        if self.project is None: return
        try:
            save_project(self.autosave_path, self.project, self.project_name.text(), self.source_path)
        except Exception:
            pass

    def save_project_as(self):
        if self.project is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", self.project_name.text() + ".colorbox", "Color Shadow Box Project (*.colorbox)")
        if path:
            if not path.lower().endswith(".colorbox"): path += ".colorbox"
            try:
                save_project(path, self.project, self.project_name.text(), self.source_path)
                self.project_path = path
                QMessageBox.information(self, "Saved", "Project saved successfully.")
            except Exception as exc:
                QMessageBox.critical(self, "Save Project", str(exc))

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Color Shadow Box Project (*.colorbox)")
        if path: self.load_project_file(path)

    def load_project_file(self, path, silent=False):
        try:
            project, name = load_project(path)
            self.project = project; self.project_name.setText(name); self.project_path = str(path)
            self.source_path = str(project.get("source_path", "")); self.source_edit.setText(self.source_path)
            self.selected_component_id = None
            self.reset_edit_history()
            self.refresh_project(); self.pages.setCurrentWidget(self.result_page)
        except Exception as exc:
            if not silent: QMessageBox.critical(self, "Open Project", str(exc))

    def export(self):
        if self.project is None: return
        if not self.project.get("validation", {}).get("safe", False):
            QMessageBox.warning(self, "Export", "Resolve the physical safety issue before export."); return
        destination = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not destination: return
        payload = {
            "project": self.project,
            "destination": destination,
            "project_name": self.project_name.text(),
            "source_path": self.source_path,
        }
        self.active_job = WorkerJob(self, ROOT_DIR / "workers" / "export_worker.py", payload, "Export Color Shadow Box", "Preparing export…")
        self.active_job.start(self.export_finished, self.job_failed)

    def export_finished(self, result):
        QMessageBox.information(self, "Export Complete", f"Files were created in:\n{result['export_root']}")

    def closeEvent(self, event):
        self.autosave()
        try:
            self.session_lock.unlink(missing_ok=True)
        except Exception:
            pass
        super().closeEvent(event)


def run():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow(); window.show()
    sys.exit(app.exec())
