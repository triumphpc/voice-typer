#!/usr/bin/env bash
#
# Собирает VoiceTyper.app и ставит его в /Applications.
#
# Бандл ссылается на этот каталог проекта (скрипт и venv), а не копирует его:
# правки в voice_typer.py подхватываются следующим запуском, без пересборки.
# Если каталог переехал - пересобери.
#
set -euo pipefail
cd "$(dirname "$0")"

PROJECT="$(pwd)"
APP_NAME="VoiceTyper"
BUNDLE_ID="local.voicetyper"
PYVER="3.13"
PYFRAMEWORK="/Library/Frameworks/Python.framework/Versions/$PYVER"
SITE="$PROJECT/.venv/lib/python$PYVER/site-packages"
DEST="${1:-/Applications}"
APP="$DEST/$APP_NAME.app"

if [ ! -d "$PYFRAMEWORK" ]; then
  echo "Нужен python.org framework build $PYVER ($PYFRAMEWORK не найден)" >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "==> создаю venv"
  "$PYFRAMEWORK/bin/python$PYVER" -m venv .venv
fi
echo "==> зависимости"
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt

echo "==> собираю бандл: $APP"
# Приложение может быть запущено - иначе замена бинарника ничего не изменит.
pkill -f "$APP/Contents/MacOS/$APP_NAME" 2>/dev/null || true

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

clang -O2 -Wall -Wextra -arch arm64 \
  -I"$PYFRAMEWORK/include/python$PYVER" \
  -F/Library/Frameworks -framework Python \
  -DVT_PROGRAM_NAME="\"$PYFRAMEWORK/bin/python$PYVER\"" \
  -DVT_SCRIPT="\"$PROJECT/voice_typer.py\"" \
  -DVT_SITE_PACKAGES="\"$SITE\"" \
  -o "$APP/Contents/MacOS/$APP_NAME" \
  launcher.c

echo "==> иконка"
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
./.venv/bin/python make_icon.py "$ICONSET/icon_512x512@2x.png"
for spec in "16 16x16" "32 16x16@2x" "32 32x32" "64 32x32@2x" \
            "128 128x128" "256 128x128@2x" "256 256x256" "512 256x256@2x" "512 512x512"; do
  set -- $spec
  sips -z "$1" "$1" "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_$2.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
rm -rf "$(dirname "$ICONSET")"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>                  <string>$APP_NAME</string>
  <key>CFBundleDisplayName</key>           <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>            <string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key>            <string>$APP_NAME</string>
  <key>CFBundleIconFile</key>              <string>AppIcon</string>
  <key>CFBundlePackageType</key>           <string>APPL</string>
  <key>CFBundleInfoDictionaryVersion</key> <string>6.0</string>
  <key>CFBundleShortVersionString</key>    <string>1.0</string>
  <key>CFBundleVersion</key>               <string>1</string>
  <key>LSMinimumSystemVersion</key>        <string>13.0</string>
  <key>NSHighResolutionCapable</key>       <true/>
  <!-- Живём в статус-баре: без иконки в Dock и без переключения приложений. -->
  <key>LSUIElement</key>                   <true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Для распознавания речи. Аудио обрабатывается локально и никуда не отправляется.</string>
  <key>NSAppleEventsUsageDescription</key>
  <string>Для системных уведомлений о статусе распознавания.</string>
</dict>
</plist>
PLIST

# Подпись ad-hoc: без неё у бандла нет стабильной идентичности и macOS будет
# сбрасывать выданные права. Идентификатор фиксируем явно - к нему привязан TCC.
codesign --force --sign - --identifier "$BUNDLE_ID" "$APP"

# Чтобы Spotlight/Launchpad увидели приложение сразу, а не через несколько минут.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
  -f "$APP" 2>/dev/null || true

echo
echo "Готово: $APP"
echo "Запуск: open -a $APP_NAME   (или двойным кликом в Finder / из Launchpad)"
