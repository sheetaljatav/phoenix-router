"""Generate the lablab.ai cover image (1920x1080) for Phoenix Router."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
img = Image.new("RGB", (W, H), "#120d0a")
d = ImageDraw.Draw(img)

# diagonal ember bands
for x0, c in [(1300, "#3a1c10"), (1420, "#57250f"), (1540, "#7a3410")]:
    d.polygon([(x0, H), (x0 + 360, H), (x0 + 860, 0), (x0 + 500, 0)], fill=c)

def font(size, bold=True):
    names = ["arialbd.ttf" if bold else "arial.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()

d.text((120, 180), "Phoenix", font=font(150), fill="#f5efe9")
d.text((120, 340), "Router", font=font(150), fill="#f2803c")
d.text((120, 560), "Hybrid Token-Efficient Routing Agent", font=font(56, False), fill="#d8c9bd")
d.text((120, 650), "8 task categories  ·  local-first inference  ·  0 Fireworks tokens",
       font=font(42, False), fill="#a68f7e")

d.rounded_rectangle([120, 800, 700, 890], radius=16, fill="#33200f")
d.text((150, 820), "AMD Developer Hackathon: ACT II", font=font(38, False), fill="#f2803c")
d.text((120, 950), "Track 1  ·  Qwen3.5-2B on llama.cpp  ·  Docker", font=font(36, False), fill="#7d6a5c")

img.save("cover.png")
print("saved cover.png", img.size)
