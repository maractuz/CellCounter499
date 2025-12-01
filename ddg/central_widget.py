# -*- coding: utf-8 -*-
#
# DotDotGoose
# Author: Peter Ersts (ersts@amnh.org)
#
# --------------------------------------------------------------------------
#
# This file is part of the DotDotGoose application.
# DotDotGoose was forked from the Neural Network Image Classifier (Nenetic).
#
# DotDotGoose is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DotDotGoose is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with with this software.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------
import os
import sys
from PyQt6 import QtCore, QtWidgets, QtGui, uic

from ddg import Canvas
from ddg import PointWidget
from ddg.fields import BoxText, LineText

# from .ui_central_widget import Ui_central as CLASS_DIALOG
if getattr(sys, 'frozen', False):
    bundle_dir = os.path.join(sys._MEIPASS, 'ddg')
else:
    bundle_dir = os.path.dirname(__file__)
CLASS_DIALOG, _ = uic.loadUiType(os.path.join(bundle_dir, 'central_widget.ui'))


class CentralWidget(QtWidgets.QDialog, CLASS_DIALOG):

    load_custom_data = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        self.setupUi(self)
        self.canvas = Canvas(self)

        self.point_widget = PointWidget(self.canvas, self)
        self.findChild(QtWidgets.QFrame, 'framePointWidget').layout().addWidget(self.point_widget)
        self.point_widget.hide_custom_fields.connect(self.hide_custom_fields)
        self.canvas.saving.connect(self.display_quick_save)

        # Keyboard shortcuts
        # Quick save using Ctrl+S
        self.save_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.Key.Key_S), self)
        self.save_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.save_shortcut.activated.connect(self.canvas.quick_save)

        # Undo Redo shortcuts
        self.save_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.Key.Key_Z), self)
        self.save_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.save_shortcut.activated.connect(self.canvas.undo)

        self.save_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.Key.Key_Y), self)
        self.save_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.save_shortcut.activated.connect(self.canvas.redo)

        # Arrow short cuts to move among images
        self.up_arrow = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Up), self)
        self.up_arrow.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.up_arrow.activated.connect(self.point_widget.previous)

        self.down_arrow = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Down), self)
        self.down_arrow.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.down_arrow.activated.connect(self.point_widget.next)

        # Same as arrow keys but conventient for right handed people
        self.up_arrow = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_W), self)
        self.up_arrow.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.up_arrow.activated.connect(self.point_widget.previous)

        self.down_arrow = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_S), self)
        self.down_arrow.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.down_arrow.activated.connect(self.point_widget.next)

        # Make signal slot connections
        self.graphicsView.setScene(self.canvas)
        self.graphicsView.drop_complete.connect(self.canvas.load)
        self.graphicsView.region_selected.connect(self.canvas.select_points)
        self.graphicsView.delete_selection.connect(self.canvas.delete_selected_points)
        self.graphicsView.relabel_selection.connect(self.canvas.relabel_selected_points)
        self.graphicsView.toggle_points.connect(self.point_widget.checkBoxDisplayPoints.toggle)
        self.graphicsView.toggle_grid.connect(self.point_widget.checkBoxDisplayGrid.toggle)
        self.graphicsView.switch_class.connect(self.point_widget.set_active_class)
        self.graphicsView.add_point.connect(self.canvas.add_point)
        self.canvas.image_loaded.connect(self.graphicsView.image_loaded)
        self.canvas.directory_set.connect(self.display_working_directory)

        # Image data fields
        self.canvas.image_loaded.connect(self.display_coordinates)
        self.canvas.image_loaded.connect(self.get_custom_field_data)
        self.canvas.fields_updated.connect(self.display_custom_fields)
        self.lineEditX.textEdited.connect(self.update_coordinates)
        self.lineEditY.textEdited.connect(self.update_coordinates)

        # Buttons
        self.pushButtonAddField.clicked.connect(self.add_field_dialog)
        self.pushButtonDeleteField.clicked.connect(self.delete_field_dialog)
        self.pushButtonFolder.clicked.connect(self.select_folder)
        self.pushButtonZoomOut.clicked.connect(self.graphicsView.zoom_out)
        self.pushButtonZoomIn.clicked.connect(self.graphicsView.zoom_in)
        self.automaticallyDetectButton.clicked.connect(self.detect_cells)

        # Fix icons since no QRC file integration
        self.pushButtonFolder.setIcon(QtGui.QIcon('icons:folder.svg'))
        self.pushButtonZoomIn.setIcon(QtGui.QIcon('icons:zoom_in.svg'))
        self.pushButtonZoomOut.setIcon(QtGui.QIcon('icons:zoom_out.svg'))
        self.pushButtonDeleteField.setIcon(QtGui.QIcon('icons:delete.svg'))
        self.pushButtonAddField.setIcon(QtGui.QIcon('icons:add.svg'))

        self.quick_save_frame = QtWidgets.QFrame(self.graphicsView)
        self.quick_save_frame.setStyleSheet("QFrame { background: #4caf50;color: #FFF;font-weight: bold}")
        self.quick_save_frame.setLayout(QtWidgets.QHBoxLayout())
        self.quick_save_frame.layout().addWidget(QtWidgets.QLabel(self.tr('Saving...')))
        self.quick_save_frame.setGeometry(3, 3, 100, 35)
        self.quick_save_frame.hide()

        self.lineEditSurveyId.textChanged.connect(self.canvas.update_survey_id)
        self.canvas.points_loaded.connect(self.lineEditSurveyId.setText)

    def resizeEvent(self, theEvent):
        self.graphicsView.resize_image()

    # Image data field functions
    def add_field(self):
        field_def = (self.field_name.text(), self.field_type.currentText())
        field_names = [x[0] for x in self.canvas.custom_fields['fields']]
        if field_def[0] in field_names:
            QtWidgets.QMessageBox.warning(self, self.tr('Warning'), self.tr('Field name already exists'))
        else:
            self.canvas.add_custom_field(field_def)
            self.add_dialog.close()

    def add_field_dialog(self):
        self.field_name = QtWidgets.QLineEdit()
        self.field_type = QtWidgets.QComboBox()
        self.field_type.addItems(['line', 'box'])
        self.add_button = QtWidgets.QPushButton(self.tr('Save'))
        self.add_button.clicked.connect(self.add_field)
        self.add_dialog = QtWidgets.QDialog(self)
        self.add_dialog.setWindowTitle(self.tr('Add Custom Field'))
        self.add_dialog.setLayout(QtWidgets.QVBoxLayout())
        self.add_dialog.layout().addWidget(self.field_name)
        self.add_dialog.layout().addWidget(self.field_type)
        self.add_dialog.layout().addWidget(self.add_button)
        self.add_dialog.resize(250, self.add_dialog.height())
        self.add_dialog.show()

    def delete_field(self):
        self.canvas.delete_custom_field(self.field_list.currentText())
        self.delete_dialog.close()

    def delete_field_dialog(self):
        self.field_list = QtWidgets.QComboBox()
        self.field_list.addItems([x[0] for x in self.canvas.custom_fields['fields']])
        self.delete_button = QtWidgets.QPushButton(self.tr('Delete'))
        self.delete_button.clicked.connect(self.delete_field)
        self.delete_dialog = QtWidgets.QDialog(self)
        self.delete_dialog.setWindowTitle(self.tr('Delete Custom Field'))
        self.delete_dialog.setLayout(QtWidgets.QVBoxLayout())
        self.delete_dialog.layout().addWidget(self.field_list)
        self.delete_dialog.layout().addWidget(self.delete_button)
        self.delete_dialog.resize(250, self.delete_dialog.height())
        self.delete_dialog.show()

    def display_coordinates(self, directory, image):
        if image in self.canvas.coordinates:
            self.lineEditX.setText(self.canvas.coordinates[image]['x'])
            self.lineEditY.setText(self.canvas.coordinates[image]['y'])
        else:
            self.lineEditX.setText('')
            self.lineEditY.setText('')

    def display_custom_fields(self, fields):

        def build(item):
            container = QtWidgets.QGroupBox(item[0], self)
            container.setObjectName(item[0])
            container.setLayout(QtWidgets.QVBoxLayout())
            if item[1].lower() == 'line':
                edit = LineText(container)
            else:
                edit = BoxText(container)
            edit.update.connect(self.canvas.save_custom_field_data)
            self.load_custom_data.connect(edit.load_data)
            container.layout().addWidget(edit)
            return container

        custom_fields = self.findChild(QtWidgets.QFrame, 'frameCustomFields')
        if custom_fields.layout() is None:
            custom_fields.setLayout(QtWidgets.QVBoxLayout())
        else:
            layout = custom_fields.layout()
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

        for item in fields:
            widget = build(item)
            custom_fields.layout().addWidget(widget)
        v = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        custom_fields.layout().addItem(v)
        self.get_custom_field_data()

    def display_working_directory(self, directory):
        self.labelWorkingDirectory.setText(directory)

    def display_quick_save(self):
        self.quick_save_frame.show()
        QtCore.QTimer.singleShot(500, self.quick_save_frame.hide)

    def get_custom_field_data(self):
        self.load_custom_data.emit(self.canvas.get_custom_field_data())

    def hide_custom_fields(self, hide):
        if hide is True:
            self.frameCustomField.hide()
        else:
            self.frameCustomField.show()

    def select_folder(self):
        name = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr('Select image folder'), self.canvas.directory)
        if name != '':
            self.canvas.load([QtCore.QUrl('file:{}'.format(name))])

    def update_coordinates(self, text):
        x = self.lineEditX.text()
        y = self.lineEditY.text()
        self.canvas.save_coordinates(x, y)

    #Skeleton Code for Detect Cells written by ChatGPT and Architecture suggested by ChatGPT
    def detect_cells(self):
        """Run cell detection on current or all images."""
        if not self._validate_detection_prerequisites():
            return
    
        process_all_images = self._show_detection_confirmation_dialog()
        if process_all_images is None:
            return
    
        images_to_process = self._get_images_to_process(process_all_images)
        if not images_to_process:
            return
    
        detector = self._initialize_detector()
        if detector is None:
            return
    
        self._run_detection_batch(detector, images_to_process)

    def _validate_detection_prerequisites(self):
        """Check if image and class are selected."""
        if self.canvas.current_image_name is None:
            QtWidgets.QMessageBox.warning(
                self, 
                self.tr('No Image'), 
                self.tr('Load an image first.'),
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            return False
    
        if self.canvas.current_class_name is None:
            QtWidgets.QMessageBox.warning(
                self, 
                self.tr('No Class Selected'), 
                self.tr('Select a class for the detected points.'),
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            return False
    
        return True

    def _show_detection_confirmation_dialog(self):
        """Show confirmation with option to process all images.
    
        Returns None if cancelled, True for batch, False for single image.
        """
        msgBox = QtWidgets.QMessageBox()
        msgBox.setWindowTitle(self.tr('Detect Cells'))
        msgBox.setText(self.tr('Run detection on this image?'))
        msgBox.setInformativeText(
        self.tr('Points will be added to the current class. '
                'This might take a few minutes.')
        )
    
        checkbox = QtWidgets.QCheckBox(self.tr('Process entire directory'))
        msgBox.setCheckBox(checkbox)
    
        msgBox.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel | 
        QtWidgets.QMessageBox.StandardButton.Ok
        )
        msgBox.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)
    
        if msgBox.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
            return None
    
        return checkbox.isChecked()

    def _get_images_to_process(self, process_all):
        """Get list of images to run detection on."""
        if process_all:
            images = list(self.canvas.points.keys())
            if not images:
                QtWidgets.QMessageBox.warning(
                self,
                self.tr('No Images'),
                self.tr('No images in current directory.'),
                QtWidgets.QMessageBox.StandardButton.Ok
                )
                return None
            return images
        else:
            return [self.canvas.current_image_name]

    def _initialize_detector(self):
        """Set up detector with model and inference script."""
        try:
            from cell_detector_wrapper import CellDetectorWrapper
        except ImportError:
            QtWidgets.QMessageBox.critical(
            self,
            self.tr('Import Error'),
            self.tr('Cannot find cell_detector_wrapper.py'),
            QtWidgets.QMessageBox.StandardButton.Ok
            )
            return None
    
        inference_script = os.path.join('ai_model', 'infer_single_overlay_improved.py')
        model_path = os.path.join('ai_model', 'cell_classifier_best.pth')
    
        if not os.path.exists(inference_script):
            QtWidgets.QMessageBox.critical(
                self,
                self.tr('Missing File'),
                self.tr('Cannot find ai_model/infer_single_overlay_improved.py\n\n'
                    'Put these in an ai_model folder:\n'
                    '- infer_single_overlay_improved.py\n'
                    '- cell_classifier_best.pth'),
            QtWidgets.QMessageBox.StandardButton.Ok
            )
            return None
    
        if not os.path.exists(model_path):
            QtWidgets.QMessageBox.critical(
            self,
            self.tr('Missing Model'),
            self.tr('Cannot find ai_model/cell_classifier_best.pth'),
            QtWidgets.QMessageBox.StandardButton.Ok
            )
            return None
    
        return CellDetectorWrapper(
            inference_script_path=inference_script,
            model_path=model_path
        )

    def _run_detection_batch(self, detector, images_to_process):
        """Process images one at a time."""
        progress = self._create_progress_dialog(len(images_to_process))
        log_window = self._create_log_window()
    
        state = {
        'current_index': 0,
        'total_detections': 0,
        'saved_image': self.canvas.current_image_name
        }
    
        def on_log(message):
            log_window.append(message)
            QtCore.QCoreApplication.processEvents()
    
        def process_next_image():
            if state['current_index'] >= len(images_to_process):
                self._finish_detection(progress, log_window, state, images_to_process)
                return
        
            image_name = images_to_process[state['current_index']]
            self._update_progress(progress, log_window, state['current_index'], len(images_to_process), image_name)
        
            def on_complete(coordinates):
                self._handle_image_complete(
                coordinates, image_name, state, log_window, 
                detector, process_next_image
                )
        
            def on_error(error_msg):
                self._handle_image_error(
                error_msg, image_name, state, log_window, 
                detector, process_next_image
                )
        
            detector.detection_complete.disconnect()
            detector.detection_error.disconnect()
            detector.detection_complete.connect(on_complete)
            detector.detection_error.connect(on_error)
        
            image_path = os.path.join(self.canvas.directory, image_name)
            detector.detect_cells(image_path)
    
        detector.detection_log.connect(on_log)
        detector.detection_complete.connect(lambda coords: None)
        detector.detection_error.connect(lambda msg: self._handle_fatal_error(msg, progress, log_window))
    
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        process_next_image()

    def _create_progress_dialog(self, total_images):
        """Make progress bar."""
        progress = QtWidgets.QProgressDialog(
        self.tr('Running detection...'), 
        None,
        0, total_images,
        self
        )
        progress.setWindowTitle(self.tr('Cell Detection'))
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()
        return progress

    def _create_log_window(self):
        """Make log window."""
        log_window = QtWidgets.QTextEdit()
        log_window.setWindowTitle(self.tr('Detection Log'))
        log_window.setReadOnly(True)
        log_window.resize(800, 400)
        log_window.show()
        return log_window

    def _update_progress(self, progress, log_window, current_index, total_images, image_name):
        """Update progress bar and log."""
        log_window.append(f"\n{'='*60}")
        log_window.append(f"Processing {current_index + 1}/{total_images}: {image_name}")
        log_window.append(f"{'='*60}")
    
        progress.setValue(current_index)
        progress.setLabelText(f"Processing {image_name}...\n({current_index + 1}/{total_images})")

    def _handle_image_complete(self, coordinates, image_name, state, log_window, detector, callback):
        """Add detected points to canvas."""
        if len(coordinates) > 0:
            if image_name not in self.canvas.points:
                self.canvas.points[image_name] = {}
            if self.canvas.current_class_name not in self.canvas.points[image_name]:
                self.canvas.points[image_name][self.canvas.current_class_name] = []
        
            for x, y in coordinates:
                point_data = QtCore.QPointF(x, y)
                self.canvas.points[image_name][self.canvas.current_class_name].append(point_data)
        
            state['total_detections'] += len(coordinates)
            log_window.append(f"Found {len(coordinates)} cells in {image_name}")
        
            self.canvas.update_point_count.emit(
                image_name, 
                self.canvas.current_class_name, 
                len(self.canvas.points[image_name][self.canvas.current_class_name])
            )
        else:
            log_window.append(f"No cells found in {image_name}")
    
        self.canvas.dirty = True
        state['current_index'] += 1
        QtCore.QTimer.singleShot(100, callback)
        
    def _handle_image_error(self, error_msg, image_name, state, log_window, detector, callback):
        """Log error and continue to next image."""
        log_window.append(f"ERROR processing {image_name}: {error_msg}")
        self.canvas.current_image_name = state['saved_image']
        state['current_index'] += 1
        QtCore.QTimer.singleShot(100, callback)

    def _finish_detection(self, progress, log_window, state, images_to_process):
        """Show summary when done."""
        QtWidgets.QApplication.restoreOverrideCursor()
        progress.close()
    
        log_window.append("\n" + "="*60)
        log_window.append("DONE")
        log_window.append(f"Processed {len(images_to_process)} images")
        log_window.append(f"Found {state['total_detections']} cells total")
        log_window.append("="*60)
    
        QtWidgets.QMessageBox.information(
        self,
        self.tr('Done'),
        self.tr(f'Processed {len(images_to_process)} images\n'
                f'Found {state["total_detections"]} cells\n\n'
                f'Close the log when ready.'),
        QtWidgets.QMessageBox.StandardButton.Ok
        )

    def _handle_fatal_error(self, error_msg, progress, log_window):
        """Stop everything if something breaks."""
        QtWidgets.QApplication.restoreOverrideCursor()
        progress.close()
    
        log_window.append("\nERROR")
        log_window.append(error_msg)
    
        QtWidgets.QMessageBox.critical(
        self,
        self.tr('Error'),
        self.tr(f'Detection failed:\n\n{error_msg}\n\nCheck the log for details.'),
        QtWidgets.QMessageBox.StandardButton.Ok
        )
