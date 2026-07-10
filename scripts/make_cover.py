"""Generate the lablab.ai cover image (1920x1080) for ZeroToken Router."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
img = Image.new("RGB", (W, H), "#0e1116")
d = ImageDraw.Draw(img)

# subtle diagonal accent bands
for i, (x0, c) in enumerate([(1300, "#16324a"), (1420, "#1d4260"), (1540, "#265379")]):
    d.polygon([(x0, H), (x0 + 360, H), (x0 + 860, 0), (x0 + 500, 0)], fill=c)

def font(size, bold=True):
    names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()

d.text((120, 180), "ZeroToken", font=font(150), fill="#f2f5f7")
d.text((120, 340), "Router", font=font(150), fill="#57a8e0")
d.text((120, 560), "Hybrid Token-Efficient Routing Agent", font=font(56, False), fill="#c9d4dc")
d.text((120, 650), "8 task categories  ·  local-first inference  ·  0 Fireworks tokens",
       font=font(42, False), fill="#8fa3b0")

d.rounded_rectangle([120, 800, 700, 890], radius=16, fill="#173042")
d.text((150, 820), "AMD Developer Hackathon: ACT II", font=font(38, False), fill="#57a8e0")
d.text((120, 950), "Track 1  ·  Qwen3.5-2B on llama.cpp  ·  Docker", font=font(36, False), fill="#5f7482")

img.save("cover.png")
print("saved cover.png", img.size)
