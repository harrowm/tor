#!/usr/bin/env bash
# setup_magnet_handler.sh
#
# Registers this bittorrent client as the default handler for magnet: links on macOS.
#
# Requirements:
#   - macOS (tested on 13+)
#   - Xcode Command Line Tools (for swiftc):  xcode-select --install
#   - uv must be installed and on PATH
#
# Usage:
#   bash setup_magnet_handler.sh
#   bash setup_magnet_handler.sh --uninstall

set -euo pipefail

APP_NAME="TorrentClient"
APP_PATH="/Applications/$APP_NAME.app"
BUNDLE_ID="com.$(whoami).torrentclient"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

if [[ "${1:-}" == "--uninstall" ]]; then
    echo "Unregistering $APP_PATH ..."
    "$LSREGISTER" -u "$APP_PATH" 2>/dev/null || true
    rm -rf "$APP_PATH"
    echo "Done. You may need to log out and back in for the change to take effect."
    exit 0
fi

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: this script is macOS-only." >&2
    exit 1
fi

if ! command -v swiftc &>/dev/null; then
    echo "Error: swiftc not found. Install Xcode Command Line Tools:" >&2
    echo "  xcode-select --install" >&2
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "Error: uv not found. Install it from https://github.com/astral-sh/uv" >&2
    exit 1
fi

UV_PATH="$(command -v uv)"

# ---------------------------------------------------------------------------
# Build app bundle
# ---------------------------------------------------------------------------

echo "Building $APP_PATH ..."

# Clean any prior install
rm -rf "$APP_PATH"
mkdir -p "$APP_PATH/Contents/MacOS"
mkdir -p "$APP_PATH/Contents/Resources"

# --- Info.plist ---
cat > "$APP_PATH/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>CFBundleURLTypes</key>
    <array>
        <dict>
            <key>CFBundleURLName</key>
            <string>Magnet Link</string>
            <key>CFBundleURLSchemes</key>
            <array>
                <string>magnet</string>
            </array>
        </dict>
    </array>
</dict>
</plist>
PLIST

# --- launch.sh (embedded in Resources) ---
# This is what actually runs the bittorrent client.
cat > "$APP_PATH/Contents/Resources/launch.sh" <<LAUNCH
#!/bin/bash
# Called by the app bundle with the magnet URL as \$1.
MAGNET="\$1"
LOG="\$HOME/Library/Logs/TorrentClient.log"
mkdir -p "\$(dirname "\$LOG")"
echo "\$(date): opening \$MAGNET" >> "\$LOG"
cd "$REPO_DIR"

# Open a Terminal window that runs the client with its Rich progress UI.
osascript <<APPLESCRIPT
tell application "Terminal"
    activate
    do script "cd '$REPO_DIR' && '$UV_PATH' run bittorrent '\$MAGNET' --output-dir \"\$HOME/Downloads\"; echo; echo 'Download complete — press any key to close'; read -n1"
end tell
APPLESCRIPT
LAUNCH
chmod +x "$APP_PATH/Contents/Resources/launch.sh"

# --- Swift source ---
# Calls LSSetDefaultHandlerForURLScheme on first launch to claim the magnet: scheme,
# even if another app currently owns it.
SWIFT_SRC="$(mktemp /tmp/torrentclient_XXXXXX.swift)"
cat > "$SWIFT_SRC" <<SWIFT
import Cocoa

class AppDelegate: NSObject, NSApplicationDelegate {
    func application(_ application: NSApplication, open urls: [URL]) {
        for url in urls {
            guard let launchScript = Bundle.main.path(forResource: "launch", ofType: "sh") else {
                NSLog("TorrentClient: launch.sh not found in bundle")
                continue
            }
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/bin/bash")
            task.arguments = [launchScript, url.absoluteString]
            do {
                try task.run()
                task.waitUntilExit()
            } catch {
                NSLog("TorrentClient: failed to launch: \\(error)")
            }
        }
        NSApp.terminate(nil)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Claim the magnet: scheme as our default — overrides any existing handler.
        let bundleID = Bundle.main.bundleIdentifier! as CFString
        LSSetDefaultHandlerForURLScheme("magnet" as CFString, bundleID)

        // Quit after a short delay if no URL was delivered (e.g. accidental double-click).
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            NSApp.terminate(nil)
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
SWIFT

# Compile
echo "Compiling Swift launcher ..."
swiftc "$SWIFT_SRC" -o "$APP_PATH/Contents/MacOS/$APP_NAME"
rm -f "$SWIFT_SRC"

# ---------------------------------------------------------------------------
# Register with Launch Services and claim the scheme
# ---------------------------------------------------------------------------

echo "Registering with Launch Services ..."
"$LSREGISTER" -f "$APP_PATH"

# Launch the app once so it can call LSSetDefaultHandlerForURLScheme on itself.
# It will quit automatically after ~1 second.
echo "Claiming magnet: scheme ..."
open -a "$APP_PATH"
sleep 2

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "Done! $APP_PATH installed and set as the default magnet: handler."
echo ""
echo "Test it:"
echo "  open 'magnet:?xt=urn:btih:0000000000000000000000000000000000000000'"
echo ""
echo "Logs are written to: ~/Library/Logs/TorrentClient.log"
echo ""
echo "To uninstall:"
echo "  bash setup_magnet_handler.sh --uninstall"
