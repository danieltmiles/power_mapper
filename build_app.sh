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
SIGNING_IDENTITY="Daniel Miles (CJ76Q945DP)"

# Your Apple ID (email)
APPLE_ID="daniel.t.miles@gmail.com"

# 10-character Team ID shown in parentheses in the signing identity
TEAM_ID="CJ76Q945DP"

# App-specific password from appleid.apple.com
# (generate one under Sign-In and Security → App-Specific Passwords)
APP_SPECIFIC_PASSWORD="goxm-ypab-cyee-iyms"

# ---------------------------------------------------------------
# Derived / fixed values
# ---------------------------------------------------------------

VERSION="0.1.1"
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
dmgbuild -s dmgbuild_settings.py "$APP_NAME" "$DMG"

echo "==> Signing DMG..."
codesign --sign "$SIGNING_IDENTITY" "$DMG"

# ---------------------------------------------------------------
# Notarize + Staple
# Uncomment once Apple's notarization service is reliable again.
# Until then, recipients can right-click → Open to bypass Gatekeeper.
# ---------------------------------------------------------------

echo "==> Submitting for notarization..."
SUBMISSION_ID=$(xcrun notarytool submit "$DMG" \
    --keychain-profile "$KEYCHAIN_PROFILE" \
    --output-format json | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "    Submission ID: $SUBMISSION_ID"
xcrun notarytool wait "$SUBMISSION_ID" \
    --keychain-profile "$KEYCHAIN_PROFILE" \
    --timeout 7200
echo "==> Stapling notarization ticket..."
xcrun stapler staple "$DMG"
echo "==> Verifying final DMG..."
spctl --assess --verbose --type install "$DMG"

echo ""
echo "==> Done: $DMG"
