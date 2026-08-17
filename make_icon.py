#!/usr/bin/env python3
"""Рисует иконку приложения (микрофон на градиентном скруглённом квадрате).

Запускается из build_app.sh, результат - AppIcon.icns.
"""

import sys

import AppKit
from Foundation import NSAttributedString, NSMakePoint, NSMakeRect

SIZE = 1024
GLYPH = "🎙"

PNG_TYPE = getattr(AppKit, "NSBitmapImageFileTypePNG", None)
if PNG_TYPE is None:  # старые pyobjc
    PNG_TYPE = AppKit.NSPNGFileType


def render(out_path: str) -> None:
    rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, SIZE, SIZE, 8, 4, True, False, AppKit.NSDeviceRGBColorSpace, 0, 0
    )
    ctx = AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)

    AppKit.NSGraphicsContext.saveGraphicsState()
    AppKit.NSGraphicsContext.setCurrentContext_(ctx)

    inset = SIZE * 0.055
    rect = NSMakeRect(inset, inset, SIZE - 2 * inset, SIZE - 2 * inset)
    shape = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        rect, SIZE * 0.2, SIZE * 0.2
    )
    gradient = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.42, 0.36, 0.85, 1.0),
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.11, 0.12, 0.21, 1.0),
    )
    gradient.drawInBezierPath_angle_(shape, -90)

    font = AppKit.NSFont.fontWithName_size_("Apple Color Emoji", SIZE * 0.44)
    glyph = NSAttributedString.alloc().initWithString_attributes_(
        GLYPH, {AppKit.NSFontAttributeName: font}
    )
    size = glyph.size()
    glyph.drawAtPoint_(
        NSMakePoint((SIZE - size.width) / 2, (SIZE - size.height) / 2 - SIZE * 0.02)
    )

    AppKit.NSGraphicsContext.restoreGraphicsState()

    data = rep.representationUsingType_properties_(PNG_TYPE, {})
    if not data.writeToFile_atomically_(out_path, True):
        raise SystemExit(f"cannot write {out_path}")


if __name__ == "__main__":
    render(sys.argv[1])
