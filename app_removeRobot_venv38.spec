# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# Thêm đường dẫn đến virtual environment .venv38
venv_path = os.path.join(os.getcwd(), '.venv38')
if os.path.exists(venv_path):
    venv_site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
    if os.path.exists(venv_site_packages):
        sys.path.insert(0, venv_site_packages)

block_cipher = None

a = Analysis(
    ['app_removeRobot.py'],
    pathex=[
        os.getcwd(),
        venv_site_packages if 'venv_site_packages' in locals() else ''
    ],
    binaries=[],
    datas=[
        ('best.pt', '.'),
        ('roi_config.json', '.'),
        ('target_point.py', '.'),
        ('calibration', 'calibration'),
        ('calibration5', 'calibration5'),
        ('charuco_camera_calibration.yaml', '.'),
    ],
    hiddenimports=[
        'ultralytics',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.detect',
        'ultralytics.models.yolo.segment',
        'ultralytics.utils',
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'pyrealsense2',
        'open3d',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'target_point',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'onnx',
        'onnxruntime',
        'tensorflow',
        'sklearn',
        'matplotlib',
        'pandas',
        'scipy',
        'sympy',
        'networkx',
        'seaborn',
        'flask',
        'werkzeug',
        'click',
        'blinker',
        'asgiref',
        'itsdangerous',
        'thop',
        'onnxslim',
        'shapely',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='YOLO_PCA_Pose_Tracking_venv38',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
