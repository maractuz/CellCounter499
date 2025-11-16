import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from PyQt6 import QtCore


def _resource_path(relative: str) -> str:
    """
    Resolve a resource path in both dev and PyInstaller exe modes.

    - In dev: this file lives in the repo root, ai_model/ is a sibling folder.
    - In exe: PyInstaller extracts datas into sys._MEIPASS, preserving the
      relative paths from the .spec (e.g. 'ai_model/...').
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        # cell_detector_wrapper.py is in the repo root next to ai_model/
        base = Path(__file__).resolve().parent

    return str(base / relative)


class CellDetectorWrapper(QtCore.QObject):

    detection_started = QtCore.pyqtSignal()
    detection_complete = QtCore.pyqtSignal(list)
    detection_error = QtCore.pyqtSignal(str)
    detection_log = QtCore.pyqtSignal(str)

    def __init__(self, inference_script_path: str = None, model_path: str = None):
        super().__init__()

    # Print diagnostic info
        print(f"[DEBUG] sys.frozen: {getattr(sys, 'frozen', False)}")
        print(f"[DEBUG] hasattr _MEIPASS: {hasattr(sys, '_MEIPASS')}")

        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            print(f"[DEBUG] _MEIPASS: {sys._MEIPASS}")
            print(f"[DEBUG] Contents of _MEIPASS:")
            import os
            for root, dirs, files in os.walk(sys._MEIPASS):
                level = root.replace(sys._MEIPASS, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files[:20]:  # Limit output
                    print(f'{subindent}{file}')
        else:
            print(f"[DEBUG] __file__: {__file__}")
            print(f"[DEBUG] __file__ parent: {Path(__file__).resolve().parent}")

        # Default relative paths (inside bundle / repo)
        default_script = "ai_model/infer_single_overlay_improved.py"
        default_model = "ai_model/cell_classifier_best.pth"

        # Resolve to absolute paths using _resource_path
        if inference_script_path is None:
            self.inference_script_path = _resource_path(default_script)
        elif os.path.isabs(inference_script_path):
            self.inference_script_path = inference_script_path
        else:
            self.inference_script_path = _resource_path(inference_script_path)

        if model_path is None:
            self.model_path = _resource_path(default_model)
        elif os.path.isabs(model_path):
            self.model_path = model_path
        else:
            self.model_path = _resource_path(model_path)

        print(f"[DEBUG] Final script path: {self.inference_script_path}")
        print(f"[DEBUG] Final model path: {self.model_path}")
        print(f"[DEBUG] Script exists: {os.path.exists(self.inference_script_path)}")
        print(f"[DEBUG] Model exists: {os.path.exists(self.model_path)}")

    def detect_cells(self, image_path: str) -> List[Tuple[int, int]]:
        try:
            self.detection_started.emit()

            if not os.path.exists(self.inference_script_path):
                raise FileNotFoundError(
                    f"Inference script not found: {self.inference_script_path}"
                )

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model weights not found: {self.model_path}"
                )

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            temp_dir = tempfile.mkdtemp(prefix="cell_detection_")

            # CRITICAL FIX: Use the correct Python interpreter
            if getattr(sys, 'frozen', False):
                # Running as PyInstaller bundle - extract and use bundled Python
                # PyInstaller doesn't bundle a Python interpreter we can call directly
                # So we need to run the inference script as a module using the same interpreter

                # Option 1: Use python from PATH (if available)
                python_exe = "python"  # Try system Python first

                # Check if python is available
                try:
                    result = subprocess.run([python_exe, "--version"],
                                          capture_output=True,
                                          timeout=5)
                    if result.returncode != 0:
                        raise FileNotFoundError("Python not found in PATH")
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    error_msg = (
                        "Cannot find Python interpreter.\n\n"
                        "When running from the packaged executable, you need Python installed "
                        "on your system and available in your PATH.\n\n"
                        "Please install Python 3.8+ and make sure it's added to your system PATH."
                    )
                    raise RuntimeError(error_msg)
            else:
                # Running in development - use the same interpreter
                python_exe = sys.executable

            cmd = [
                python_exe,
                self.inference_script_path,
                "--image", image_path,
                "--weights", self.model_path,
                "--out_dir", temp_dir,
                "--device", "cpu",
                "--color_order", "rgb",
                "--window", "200",
                "--use_edge_gate",
                "--keep_border",
                "--border_px", "2",
                "--green_k", "0.24",
                "--tR", "0.55",
                "--tG", "0.55",
                "--tB", "0.42",
                "--delta_rg", "0.16",
                "--rg_ratio", "0.82",
                "--min_cc_area_gate", "8",
                "--max_cc_area_gate", "240",
                "--candidate_union_radius", "5",
                "--nms_radius", "20",
                "--post_tight_pair_floor", "18",
                "--fp_gate_proximity",
                "--fp_rededge",
                "--snap",
                "--snap_search_r", "10",
                "--snap_jump_limit", "12",
                "--snap_s_lo", "150",
                "--snap_v_lo", "170",
                "--h_lo", "24",
                "--h_hi", "34",
                "--s_lo", "210",
                "--v_lo", "210",
                "--min_area", "8",
                "--max_area", "240",
                "--threshold", "0.92",
            ]

            self.detection_log.emit(
                f"Running inference script:\n  {self.inference_script_path}\n"
                f"Using weights:\n  {self.model_path}\n"
                f"Python: {python_exe}\n"
                f"Command: {' '.join(cmd[:3])}..."
            )

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    self.detection_log.emit(output.strip())
                    QtCore.QCoreApplication.processEvents()

            stderr = process.stderr.read()
            returncode = process.poll()

            if returncode != 0:
                error_msg = f"Inference script failed with code {returncode}\n"
                if stderr:
                    error_msg += f"Error: {stderr}"
                raise RuntimeError(error_msg)

            image_stem = Path(image_path).stem
            pnt_file = os.path.join(temp_dir, f"{image_stem}.pnt")

            if not os.path.exists(pnt_file):
                # Log what files were created
                files_created = os.listdir(temp_dir) if os.path.exists(temp_dir) else []
                raise FileNotFoundError(
                    f"Expected output file not found: {pnt_file}\n"
                    f"Files in temp dir: {files_created}"
                )

            coordinates = self.load_pnt_file(pnt_file)

            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            self.detection_complete.emit(coordinates)
            return coordinates

        except subprocess.TimeoutExpired:
            error_msg = (
                "Detection timed out (>10 minutes). "
                "Try a smaller image or adjust parameters."
            )
            self.detection_error.emit(error_msg)
            return []
        except Exception as e:
            error_msg = f"Error during detection: {str(e)}"
            self.detection_error.emit(error_msg)
            return []

    def load_pnt_file(self, pnt_file: str) -> List[Tuple[int, int]]:
        with open(pnt_file, "r") as f:
            data = json.load(f)

        coordinates: List[Tuple[int, int]] = []

        if "points" in data:
            for _, classes in data["points"].items():
                for _, points in classes.items():
                    for point in points:
                        x = int(point["x"])
                        y = int(point["y"])
                        coordinates.append((x, y))
        return coordinates


class DetectionWorker(QtCore.QRunnable):

    def __init__(self, detector: CellDetectorWrapper, image_path: str):
        super().__init__()
        self.detector = detector
        self.image_path = image_path

    def run(self):
        self.detector.detect_cells(self.image_path)
