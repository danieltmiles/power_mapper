# dmgbuild settings for PowerMapper
# https://dmgbuild.readthedocs.io/en/latest/settings.html

import os

# Volume name shown when the DMG is mounted
volume_name = "PowerMapper"

# Output format
format = "UDZO"          # compressed
compression_level = 9

# Window appearance
window_rect = ((200, 120), (540, 380))
icon_size = 128
background = "builtin-arrow"   # simple drag-to-install arrow background

files = [
    os.path.join(os.path.dirname(__file__), "dist", "PowerMapper.app"),
]

symlinks = {
    "Applications": "/Applications",
}

icon_locations = {
    "PowerMapper.app": (160, 185),
    "Applications":    (380, 185),
}
