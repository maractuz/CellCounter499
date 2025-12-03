# ddg.spec

from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE

# Only PyQt6 needs to be force-collected
hiddenimports = collect_submodules("PyQt6")

a = Analysis(
    ['../main.py'],
    pathex=['C:/Users/Avi/IdeaProjects/CellCounter499'],
    binaries=[],
    datas=[
        # UI files
        ("../ddg/about_dialog.ui", "ddg/"),
        ("../ddg/chip_dialog.ui", "ddg/"),
        ("../ddg/point_widget.ui", "ddg/"),
        ("../ddg/central_widget.ui", "ddg/"),
        ("../ddg/central_graphics_view.py", "ddg/"),

        # Icons
        ("../icons/add.svg", "icons/"),
        ("../icons/cancel.svg", "icons/"),
        ("../icons/ddg.png", "icons/"),
        ("../icons/delete.svg", "icons/"),
        ("../icons/export.svg", "icons/"),
        ("../icons/folder.svg", "icons/"),
        ("../icons/import.svg", "icons/"),
        ("../icons/load.svg", "icons/"),
        ("../icons/reset.svg", "icons/"),
        ("../icons/save.svg", "icons/"),
        ("../icons/zoom_in.svg", "icons/"),
        ("../icons/zoom_out.svg", "icons/"),

        # Translations
        ("../i18n/ddg_es.qm", "i18n/"),
        ("../i18n/ddg_fr.qm", "i18n/"),
        ("../i18n/ddg_vi.qm", "i18n/"),
        ("../i18n/ddg_zh_hans_cn.qm", "i18n/"),

        # Doc
        ("../doc/CellCounter.pdf", "doc/"),

        # AI model files (bundled as plain data, not analyzed)
        ("../ai_model/infer_single_overlay_improved.py", "ai_model/"),
        ("../ai_model/cell_classifier_best.pth", "ai_model/"),
    ],
    hiddenimports=hiddenimports,
    hooksconfig={},
    runtime_hooks=[],
    # we do NOT want PyInstaller to try to bundle these heavy libs, since
    # inference runs in an external Python:
    excludes=['torch', 'torchvision', 'cv2'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ddg',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # turn off UPX to avoid extra work
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
