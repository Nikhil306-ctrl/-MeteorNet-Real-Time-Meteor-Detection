
from pathlib import Path
from PIL import Image, ImageDraw
import random

root = Path("D_2 DATASET (METEORS)/sum_images")

random.seed(42)

for class_name in ["meteor", "non_meteor"]:

    folder = root / class_name

    images = [
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    selected = random.sample(images, 30)

    sheet = Image.new("RGB", (750, 600), "white")
    draw = ImageDraw.Draw(sheet)

    for i, path in enumerate(selected):

        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((140, 105))

        x = (i % 5) * 150
        y = (i // 5) * 100

        sheet.paste(image, (x, y))

        draw.text(
            (x + 3, y + 78),
            f"Image {i + 1}",
            fill="red"
        )

    output = Path("results") / f"{class_name}_samples.jpg"
    output.parent.mkdir(exist_ok=True)

    sheet.save(output)
    

    print("Saved:", output)
