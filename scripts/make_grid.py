from PIL import Image
import argparse
import math
import os

def main():
    parser = argparse.ArgumentParser(description="Make NxN image grid from numbered PNGs")
    parser.add_argument("--indir", type=str, default="cifar10", help="Input directory containing 000000.png ...")
    parser.add_argument("--n", type=int, default=64, help="Number of images (must be a perfect square)")
    parser.add_argument("--out", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    N = args.n
    grid = int(math.sqrt(N))
    if grid * grid != N:
        raise ValueError(f"--n must be a perfect square, got {N}")

    out_path = args.out or f"out_grid_{grid}x{grid}.png"

    paths = [os.path.join(args.indir, f"{i:06d}.png") for i in range(N)]
    imgs = [Image.open(p) for p in paths]

    w, h = imgs[0].size
    canvas = Image.new("RGB", (grid * w, grid * h))

    for idx, img in enumerate(imgs):
        row = idx // grid
        col = idx % grid
        canvas.paste(img, (col * w, row * h))

    canvas.save(out_path)
    print(f"saved {out_path}")

if __name__ == "__main__":
    main()