#!/bin/bash
# Build, sign, notarize, and staple PowerMapper.
#
# Fill in the variables below before running.
# Usage: ./build_app.sh

set -euo pipefail

# ---------------------------------------------------------------
# Configuration — fill these in
# ---------------------------------------------------------------

# From: security find-identity -v -p codesigning
# e.g. "Developer ID Application: Jane Smith (ABCDE12345)"
SIGNING_IDENTITY=""

# Your Apple ID (email)
APPLE_ID=""

# 10-character Team ID shown in parentheses in the signing identity
TEAM_ID=""

# App-specific password from appleid.apple.com
# (generate one under Sign-In and Security → App-Specific Passwords)
APP_SPECIFIC_PASSWORD=""

# ---------------------------------------------------------------
# Derived / fixed values
# ---------------------------------------------------------------

VERSION="0.1.0"
APP_NAME="PowerMapper"
DMG="${APP_NAME}-${VERSION}.dmg"
APP_BUNDLE="dist/${APP_NAME}.app"
ENTITLEMENTS="entitlements.plist"
KEYCHAIN_PROFILE="${APP_NAME}-notary"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------
# Sanity-check that the required variables are set
# ---------------------------------------------------------------

for var in SIGNING_IDENTITY APPLE_ID TEAM_ID APP_SPECIFIC_PASSWORD; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: $var is not set. Edit the variables at the top of $0."
        exit 1
    fi
done

# ---------------------------------------------------------------
# Store notarization credentials in the Keychain (idempotent)
# ---------------------------------------------------------------

echo "==> Storing notarization credentials in Keychain..."
xcrun notarytool store-credentials "$KEYCHAIN_PROFILE" \
    --apple-id "$APPLE_ID" \
    --team-id "$TEAM_ID" \
    --password "$APP_SPECIFIC_PASSWORD"

# ---------------------------------------------------------------
# Clean
# ---------------------------------------------------------------

echo "==> Cleaning previous build..."
rm -rf build/ dist/ "$DMG"

# ---------------------------------------------------------------
# Build
# ---------------------------------------------------------------

echo "==> Activating virtual environment..."
source venv/bin/activate

echo "==> Running PyInstaller..."
pyinstaller PowerMapper.spec

# ---------------------------------------------------------------
# Sign
# ---------------------------------------------------------------

echo "==> Signing app bundle..."
codesign --deep --force --verify --verbose \
    --options runtime \
    --entitlements "$ENTITLEMENTS" \
    --sign "$SIGNING_IDENTITY" \
    "$APP_BUNDLE"

echo "==> Verifying signature..."
codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"

# ---------------------------------------------------------------
# DMG
# ---------------------------------------------------------------

echo "==> Creating DMG..."
dmgbuild \
    --settings dmgbuild_settings.py \
    --defines "signing_identity=$SIGNING_IDENTITY" \
    "$APP_NAME" "$DMG"

echo "==> Signing DMG..."
codesign --sign "$SIGNING_IDENTITY" "$DMG"

# ---------------------------------------------------------------
# Notarize
# ---------------------------------------------------------------

echo "==> Submitting for notarization (this takes several minutes)..."
xcrun notarytool submit "$DMG" \
    --keychain-profile "$KEYCHAIN_PROFILE" \
    --wait

# ---------------------------------------------------------------
# Staple
# ---------------------------------------------------------------

echo "==> Stapling notarization ticket..."
xcrun stapler staple "$DMG"

# ---------------------------------------------------------------
# Verify
# ---------------------------------------------------------------

echo "==> Verifying final DMG..."
spctl --assess --verbose --type install "$DMG"

echo ""
echo "==> Done: $DMG"
