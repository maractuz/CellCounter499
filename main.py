#!/usr/bin/python3
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
# along with this software.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------

import os
import sys
from pathlib import Path
from PyQt6 import QtWidgets, QtCore
from ddg import ExceptionHandler, MainWindow, DarkModePalette


# --------------------------------------------------------------------------
# 🧠 PATCH: Ensure child processes use the SAME Python interpreter (your venv)
# --------------------------------------------------------------------------
def _force_children_to_use_current_python():
    """
    Ensures any child call to 'python' (via subprocess or QProcess)
    resolves to this exact Python interpreter — typically the one inside
    your ddg-env virtual environment.
    """
    exe = Path(sys.executable)
    scripts_dir = exe.parent  # e.g. C:\Users\Avi\IdeaProjects\CellCounter499\ddg-env\Scripts

    # Prepend the Scripts directory to PATH so 'python' resolves correctly
    os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")

    # Expose a variable that other scripts (like DDG detection handlers) can read
    os.environ["DDG_PYTHON"] = str(exe)

    # Remove PYTHONHOME (can break subprocess isolation)
    os.environ.pop("PYTHONHOME", None)

    # Debug output (helps verify it’s using ddg-env)
    print(f"[DDG] MAIN PYTHON: {exe}")
    print(f"[DDG] Updated PATH to prioritize: {scripts_dir}")


_force_children_to_use_current_python()


# --------------------------------------------------------------------------
# 🚀 Launch DotDotGoose GUI
# --------------------------------------------------------------------------
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    if getattr(sys, 'frozen', False):
        QtCore.QDir.addSearchPath('icons', os.path.join(sys._MEIPASS, 'icons'))
        QtCore.QDir.addSearchPath('i18n', os.path.join(sys._MEIPASS, 'i18n'))
    else:
        QtCore.QDir.addSearchPath('icons', './icons/')
        QtCore.QDir.addSearchPath('i18n', './i18n/')

    app.setStyle('fusion')

    # Enable dark mode palette if supported
    if app.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark:
        app.setPalette(DarkModePalette())
        # Palette colors are not honored by Qt6.5.3
        app.setStyleSheet("QToolTip { color: #ffffff; background-color: #000000; border: 0px; padding: 2px}")

    # Localization support
    settings = QtCore.QSettings("AMNH", "DotDotGoose")
    translator = QtCore.QTranslator()
    if settings.value('locale'):
        if translator.load(QtCore.QLocale(settings.value('locale')), "ddg", "_", "i18n:/"):
            QtCore.QCoreApplication.installTranslator(translator)
    else:
        if translator.load(QtCore.QLocale(), "ddg", "_", "i18n:/"):
            QtCore.QCoreApplication.installTranslator(translator)

    # Main window setup
    main = MainWindow()
    handler = ExceptionHandler()
    handler.exception.connect(main.display_exception)
    main.show()

    # Screen configuration
    screen = app.primaryScreen()
    for s in app.screens():
        if screen.geometry().width() < s.geometry().width():
            screen = s
    main.windowHandle().setScreen(screen)
    main.resize(int(screen.geometry().width()), int(screen.geometry().height() * 0.85))

    sys.exit(app.exec())
