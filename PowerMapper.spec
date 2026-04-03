# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Power Mapper LLM Worker menu bar app.

Build with:
    source venv/bin/activate
    pyinstaller PowerMapper.spec
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata

VENV = Path("venv/lib/python3.11/site-packages")
LLAMA_LIB = VENV / "llama_cpp" / "lib"

SHARED_BINARIES = [
    # llama_cpp ships its Metal dylibs in llama_cpp/lib/ — PyInstaller
    # won't find these automatically since they're loaded via ctypes at
    # runtime, not via a Python import.
    (str(LLAMA_LIB / "*.dylib"), "llama_cpp/lib"),
]

SHARED_DATAS = [
    # TLS certificates.
    ("server_certificate.pem", "."),   # RabbitMQ client auth
    ("tls/ca.crt",             "tls"), # Redis CA
    ("tls/client.crt",         "tls"), # Redis client cert
    ("tls/client.key",         "tls"), # Redis client key
    # transformers needs its configuration files (tokenizer configs etc.)
    (str(VENV / "transformers"), "transformers"),
    # Packages that call importlib.metadata at import time need their dist-info.
    *copy_metadata("aio-pika"),
    *copy_metadata("aiormq"),
]

a = Analysis(
    ["menubar_app.py"],
    pathex=["."],
    binaries=SHARED_BINARIES,
    datas=SHARED_DATAS,
    hiddenimports=[
        # keyring backends — include the macOS one explicitly
        "keyring.backends.macOS",
        # rumps internals
        "rumps",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pyannote",
        "whisper",
        "pydub",
        "torchaudio",
        "sentence_transformers",
        "scipy",
        "sklearn",
        "matplotlib",
        "pytest",
    ],
    noarchive=False,
)

# llm.py gets its own executable so the menubar app can launch it directly
# without needing a Python interpreter.
a_worker = Analysis(
    ["llm.py"],
    pathex=["."],
    binaries=SHARED_BINARIES,
    datas=SHARED_DATAS + [
        ("utils.py",           "."),
        ("wire_formats.py",    "."),
        ("serialization.py",   "."),
        ("cached_iterator.py", "."),
        ("logger.py",          "."),
    ],
    hiddenimports=[
        # llama_cpp loads these dynamically via ctypes
        "llama_cpp",
        "llama_cpp.llama",
        "llama_cpp.llama_cpp",
        "llama_cpp._ctypes_extensions",
        # aio_pika pulls in these at runtime
        "aio_pika",
        "aiormq",
        # redis asyncio client
        "redis",
        "redis.asyncio",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pyannote",
        "whisper",
        "pydub",
        "torchaudio",
        "sentence_transformers",
        "scipy",
        "sklearn",
        "matplotlib",
        "pytest",
        "rumps",
        "keyring",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)
pyz_worker = PYZ(a_worker.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PowerMapper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,        # UPX breaks code-signing; leave off
    console=False,    # no terminal window
    argv_emulation=True,   # required for macOS menu bar apps
)

exe_worker = EXE(
    pyz_worker,
    a_worker.scripts,
    [],
    exclude_binaries=True,
    name="llm_worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,     # worker is headless; stdout/stderr go to the log file
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    exe_worker,
    a_worker.binaries,
    a_worker.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PowerMapper",
)

app = BUNDLE(
    coll,
    name="PowerMapper.app",
    # icon="icon.icns",   # uncomment once you have an icon
    bundle_identifier="org.doodledome.powermapper",
    version="0.1.0",
    info_plist={
        "LSUIElement": True,           # hides the Dock icon; menu-bar-only app
        "NSHighResolutionCapable": True,
        "CFBundleShortVersionString": "0.1.0",
        "NSAppleEventsUsageDescription": "Power Mapper uses AppleScript for notifications.",
        "NSPrincipalClass": "NSApplication",
    },
)
