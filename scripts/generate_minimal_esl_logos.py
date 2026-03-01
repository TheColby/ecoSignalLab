from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path("/Users/cleider/dev/ecoSignalLab/assets/logos/esl/minimal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 1024
BLACK = (0, 0, 0, 255)
TRANSPARENT = (0, 0, 0, 0)


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (SIZE, SIZE), TRANSPARENT)
    return img, ImageDraw.Draw(img)


def load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), px)
            except Exception:
                continue
    return ImageFont.load_default()


def centered_text(draw: ImageDraw.ImageDraw, text: str, y: float, size: int) -> None:
    font = load_font(size)
    box = draw.textbbox((0, 0), text, font=font)
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = (SIZE - w) // 2
    yy = int(y - h / 2)
    draw.text((x, yy), text, fill=BLACK, font=font)


def caption_text(draw: ImageDraw.ImageDraw, text: str = "ecoSignalLab", y: float = 956, size: int = 52) -> None:
    centered_text(draw, text, y=y, size=size)


def save(img: Image.Image, name: str) -> None:
    out = OUT_DIR / name
    img.save(out, "PNG")


def polyline(draw: ImageDraw.ImageDraw, points: Iterable[tuple[float, float]], width: int) -> None:
    pts = [(int(x), int(y)) for x, y in points]
    draw.line(pts, fill=BLACK, width=width, joint="curve")


def logo_01_ring_monogram() -> Image.Image:
    img, d = new_canvas()
    d.ellipse((130, 130, 894, 894), outline=BLACK, width=26)
    centered_text(d, "esl", y=512, size=260)
    caption_text(d)
    return img


def logo_02_square_monogram() -> Image.Image:
    img, d = new_canvas()
    d.rounded_rectangle((170, 170, 854, 854), radius=96, outline=BLACK, width=24)
    centered_text(d, "esl", y=512, size=240)
    caption_text(d)
    return img


def logo_03_wave_wordmark() -> Image.Image:
    img, d = new_canvas()
    pts: list[tuple[float, float]] = []
    for i in range(420):
        x = 130 + i
        phase = (i / 420.0) * 8.0 * 3.14159265
        y = 512 + 70 * __import__("math").sin(phase)
        pts.append((x, y))
    polyline(d, pts, width=24)
    font = load_font(210)
    d.text((620, 410), "esl", fill=BLACK, font=font)
    caption_text(d)
    return img


def logo_04_circle_wave() -> Image.Image:
    img, d = new_canvas()
    d.ellipse((160, 160, 864, 864), outline=BLACK, width=20)
    pts: list[tuple[float, float]] = []
    for i in range(640):
        x = 190 + i
        phase = (i / 640.0) * 10.0 * 3.14159265
        y = 512 + 46 * __import__("math").sin(phase)
        pts.append((x, y))
    polyline(d, pts, width=20)
    centered_text(d, "esl", y=700, size=120)
    caption_text(d)
    return img


def logo_05_equalizer() -> Image.Image:
    img, d = new_canvas()
    xs = [270, 360, 450, 540, 630, 720]
    tops = [470, 360, 290, 390, 500, 430]
    for x, top in zip(xs, tops):
        d.line((x, 730, x, top), fill=BLACK, width=38)
    centered_text(d, "esl", y=826, size=150)
    caption_text(d)
    return img


def logo_06_leaf_wave() -> Image.Image:
    img, d = new_canvas()
    d.ellipse((220, 220, 560, 760), outline=BLACK, width=20)
    d.line((300, 680, 470, 310), fill=BLACK, width=16)
    pts: list[tuple[float, float]] = []
    for i in range(300):
        x = 520 + i
        phase = (i / 300.0) * 6.0 * 3.14159265
        y = 560 + 42 * __import__("math").sin(phase)
        pts.append((x, y))
    polyline(d, pts, width=18)
    centered_text(d, "esl", y=360, size=110)
    caption_text(d)
    return img


def logo_07_orbit() -> Image.Image:
    img, d = new_canvas()
    d.ellipse((190, 190, 834, 834), outline=BLACK, width=14)
    for cx, cy in [(512, 190), (834, 512), (512, 834), (190, 512)]:
        d.ellipse((cx - 34, cy - 34, cx + 34, cy + 34), fill=BLACK)
    centered_text(d, "esl", y=512, size=220)
    caption_text(d)
    return img


def logo_08_badge() -> Image.Image:
    img, d = new_canvas()
    d.rounded_rectangle((130, 250, 894, 774), radius=120, outline=BLACK, width=24)
    d.line((250, 450, 774, 450), fill=BLACK, width=18)
    centered_text(d, "esl", y=568, size=190)
    caption_text(d)
    return img


def main() -> None:
    logos = [
        ("esl_logo_04_circle_wave.png", logo_04_circle_wave),
    ]
    for filename, fn in logos:
        save(fn(), filename)
    print(f"Wrote logos to: {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("esl_logo_*.png")):
        print(p)


if __name__ == "__main__":
    main()
