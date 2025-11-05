import os
import json
import subprocess
import tempfile
from pathlib import Path
from PyQt6 import QtCore
from typing import List, Tuple


class CellDetectorWrapper(QtCore.QObject):
    
    detection_started = QtCore.pyqtSignal()
    detection_complete = QtCore.pyqtSignal(list)
    detection_error = QtCore.pyqtSignal(str)
    detection_log = QtCore.pyqtSignal(str)
    
    def __init__(self, inference_script_path: str = None, model_path: str = None):
        super().__init__()
        self.inference_script_path = inference_script_path or "ai_model/infer_single_overlay_improved.py"
        self.model_path = model_path or "ai_model/cell_classifier_best.pth"
        
    def detect_cells(self, image_path: str) -> List[Tuple[int, int]]:
        try:
            self.detection_started.emit()
            
            if not os.path.exists(self.inference_script_path):
                raise FileNotFoundError(f"Inference script not found: {self.inference_script_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model weights not found: {self.model_path}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            temp_dir = tempfile.mkdtemp(prefix="cell_detection_")
            
            cmd = [
                "python",
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
                "--threshold", "0.92"
            ]
            
            self.detection_log.emit("Running inference script...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.detection_log.emit(line)
            
            if result.returncode != 0:
                error_msg = f"Inference script failed with code {result.returncode}\n"
                if result.stderr:
                    error_msg += f"Error: {result.stderr}"
                raise RuntimeError(error_msg)
            
            image_stem = Path(image_path).stem
            pnt_file = os.path.join(temp_dir, f"{image_stem}.pnt")
            
            if not os.path.exists(pnt_file):
                raise FileNotFoundError(f"Expected output file not found: {pnt_file}")
            
            coordinates = self.load_pnt_file(pnt_file)
            
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            self.detection_complete.emit(coordinates)
            return coordinates
            
        except subprocess.TimeoutExpired:
            error_msg = "Detection timed out (>10 minutes). Try a smaller image or adjust parameters."
            self.detection_error.emit(error_msg)
            return []
        except Exception as e:
            error_msg = f"Error during detection: {str(e)}"
            self.detection_error.emit(error_msg)
            return []
    
    def load_pnt_file(self, pnt_file: str) -> List[Tuple[int, int]]:
        with open(pnt_file, 'r') as f:
            data = json.load(f)
        
        coordinates = []
        
        if 'points' in data:
            for image_name, classes in data['points'].items():
                for class_name, points in classes.items():
                    for point in points:
                        x = int(point['x'])
                        y = int(point['y'])
                        coordinates.append((x, y))
        return coordinates


class DetectionWorker(QtCore.QRunnable):
    
    def __init__(self, detector: CellDetectorWrapper, image_path: str):
        super().__init__()
        self.detector = detector
        self.image_path = image_path
        
    def run(self):
        self.detector.detect_cells(self.image_path)