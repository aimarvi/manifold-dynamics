from __future__ import annotations

import argparse
import io
import pickle
from pathlib import Path

import fsspec
import imageio.v2 as iio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GIF of time-time RDMs across increasing image-set size.")
    parser.add_argument("--targets", type=str, nargs="+", required=True,
        help="One or more ROI targets. Each target can be a 4-part UID or 3-part ROI key.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bin-size-ms", type=int, default=20)
    parser.add_argument("--tstart", type=int, default=100)
    parser.add_argument("--tend", type=int, default=350)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--start-k", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--loop", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--cmap", type=str, default="rocket")
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--cbar", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg)

    def resolve_roi_label(target: str) -> str:
        parts = target.split(".")
        if len(parts) == 4:
            return f"{int(parts[1]):02d}.{parts[2]}.{parts[3]}"
        if len(parts) == 3:
            return f"{int(parts[0]):02d}.{parts[1]}.{parts[2]}"
        raise ValueError(
            "Invalid target format. Use 4-part UID (SesIdx.RoiIndex.AREALABEL.Categoty) "
            "or 3-part ROI key (RoiIndex.AREALABEL.Categoty)."
        )

    def render_tile(R: np.ndarray, vmin: float, vmax: float, cmap: str, dpi: int) -> Image.Image:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=dpi)
        fig.patch.set_alpha(0)
        ax.set_facecolor((1, 1, 1, 0))
        sns.heatmap(R, ax=ax, cmap=sns.color_palette(cmap, as_cmap=True),
            vmin=vmin, vmax=vmax, square=True, cbar=False, xticklabels=False, yticklabels=False)

        # Uncomment to show a tile-level title.
        # ax.set_title("Top-k")

        # Uncomment to show axis labels on every tile.
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Time")

        # Uncomment to show ticks instead of blank axes.
        # ax.set_xticks(np.arange(0, R.shape[0] + 50, 50))
        # ax.set_yticks(np.arange(0, R.shape[0] + 50, 50))

        ax.set_xlabel("")
        ax.set_ylabel("")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        buffer.seek(0)
        return Image.open(buffer).convert("RGBA")

    def render_cbar(vmin: float, vmax: float, cmap: str, height: int, dpi: int) -> Image.Image:
        fig, ax = plt.subplots(figsize=(0.4, 3), dpi=dpi)
        fig.patch.set_alpha(0)
        ax.set_facecolor((1, 1, 1, 0))
        cb = mpl.colorbar.ColorbarBase(ax=ax, cmap=plt.get_cmap(cmap),
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), orientation="vertical")
        cb.set_ticks(np.linspace(vmin, vmax, 5))
        cb.set_ticklabels([f"{t:.2f}" for t in np.linspace(vmin, vmax, 5)])
        ax.tick_params(labelsize=6)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        buffer.seek(0)
        img = Image.open(buffer).convert("RGBA")
        return img if img.height == height else img.resize((img.width, height), Image.BILINEAR)

    def write_gif(filepath_out: str, frames: list[Image.Image], duration: float, loop: int) -> None:
        frames_p = []
        for frame in frames:
            alpha = frame.getchannel("A")
            pal = frame.convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=255)
            mask = Image.eval(alpha, lambda a: 255 if a == 0 else 0)
            pal.paste(255, mask)
            pal.info["transparency"] = 255
            frames_p.append(pal)

        if "://" in filepath_out:
            buffer = io.BytesIO()
            frames_p[0].save(buffer, format="GIF", save_all=True, append_images=frames_p[1:],
                duration=int(duration * 1000), loop=loop, transparency=255, disposal=2)
            buffer.seek(0)
            with fsspec.open(filepath_out, "wb") as f:
                f.write(buffer.getvalue())
            return

        Path(filepath_out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        frames_p[0].save(filepath_out, save_all=True, append_images=frames_p[1:],
            duration=int(duration * 1000), loop=loop, transparency=255, disposal=2)

    topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
    with open(topk_local, "rb") as f:
        topk_vals = pickle.load(f)

    roi_labels = [resolve_roi_label(target) for target in args.targets]
    title_font = ImageFont.load_default(size=32)
    body_font = ImageFont.load_default(size=28)
    seqs_top, seqs_random, sizes_ref = {}, {}, None

    for target, roi_label in zip(args.targets, roi_labels):
        raster_4d = nu.significant_trial_raster(roi_uid=target, alpha=args.alpha, bin_size_ms=args.bin_size_ms)
        X = np.nanmean(raster_4d, axis=3)  # (units, time, images)
        n_images = X.shape[2]
        start_k = args.start_k if args.start_k is not None else int(topk_vals[roi_label]["k"])
        if start_k < 2:
            raise ValueError(f"{target}: start_k must be >= 2, got {start_k}")
        sizes = [k for k in range(start_k, min(args.k_max, n_images) + 1, args.step)]
        if len(sizes) == 0:
            raise ValueError(f"{target}: empty size schedule. Check --start-k/--k-max.")
        if sizes_ref is None:
            sizes_ref = sizes
        elif sizes != sizes_ref:
            raise ValueError(
                f"{target}: size schedule does not match the first ROI. "
                "For multi-ROI GIFs, pass an explicit shared --start-k and --k-max."
            )

        order_top = np.asarray(tut.rank_images_by_response(X), dtype=int)
        order_random = np.random.default_rng(args.random_state).permutation(n_images)

        seqs_top[target] = []
        seqs_random[target] = []
        for k in tqdm(sizes, desc=f"{target} [top]"):
            R, _ = tut.tuning_rdm(X=X, indices=order_top[:k], tstart=args.tstart, tend=args.tend, metric="correlation")
            seqs_top[target].append(R)
        if args.include_random:
            for k in tqdm(sizes, desc=f"{target} [random]"):
                R, _ = tut.tuning_rdm(X=X, indices=order_random[:k], tstart=args.tstart, tend=args.tend, metric="correlation")
                seqs_random[target].append(R)

        vprint(f"{target}: computed {len(seqs_top[target])} top frames")

    if sizes_ref is None:
        raise ValueError("No targets were processed.")

    all_rdms = [R for seq in seqs_top.values() for R in seq]
    if args.include_random:
        all_rdms.extend([R for seq in seqs_random.values() for R in seq])
    vmin = 0.0
    vmax = 1.0
    # vmax = float(np.nanmax([np.nanmax(R) for R in all_rdms]))

    first_tile = render_tile(next(iter(seqs_top.values()))[0], vmin=vmin, vmax=vmax, cmap=args.cmap, dpi=args.dpi)
    tile_w, tile_h = first_tile.size
    cb_img = render_cbar(vmin=vmin, vmax=vmax, cmap=args.cmap,
        height=(2 if args.include_random else 1) * tile_h + 40, dpi=args.dpi)

    frames, title_h, footer_h, row_gap = [], 28, 28, 8
    for frame_idx in range(len(sizes_ref)):
        tiles_top = [render_tile(seqs_top[target][frame_idx], vmin=vmin, vmax=vmax, cmap=args.cmap, dpi=args.dpi) for target in args.targets]
        row_top = Image.new("RGBA", (tile_w * len(args.targets), tile_h), color=(255, 255, 255, 0))
        for j, img in enumerate(tiles_top):
            row_top.paste(img, (j * tile_w, 0), img)

        row_random = None
        grid_h = title_h + tile_h + footer_h
        if args.include_random:
            tiles_random = [render_tile(seqs_random[target][frame_idx], vmin=vmin, vmax=vmax, cmap=args.cmap, dpi=args.dpi) for target in args.targets]
            row_random = Image.new("RGBA", (tile_w * len(args.targets), tile_h), color=(255, 255, 255, 0))
            for j, img in enumerate(tiles_random):
                row_random.paste(img, (j * tile_w, 0), img)
            grid_h = title_h + (2 * tile_h) + row_gap + footer_h

        grid = Image.new("RGBA", (row_top.width, grid_h), color=(255, 255, 255, 0))
        grid.paste(row_top, (0, title_h), row_top)
        if row_random is not None:
            grid.paste(row_random, (0, title_h + tile_h + row_gap), row_random)

        draw = ImageDraw.Draw(grid)
        for j, roi_label in enumerate(roi_labels):
            xmid = j * tile_w + tile_w // 2

            # Uncomment to show ROI labels above each tile.
            # draw.text((xmid, title_h // 2), roi_label, fill=(0, 0, 0, 255), anchor="ma", font=title_font)

        # Uncomment to show row labels on the left margin.
        # draw.text((5, title_h + tile_h // 2), "top", fill=(0, 0, 0, 255), anchor="lm", font=body_font)
        # if row_random is not None:
        #     draw.text((5, title_h + tile_h + row_gap + tile_h // 2), "random", fill=(0, 0, 0, 255), anchor="lm", font=body_font)

        # Uncomment to show a footer with image counts.
        # used_k = int(sizes_ref[frame_idx])
        # footer = f"Images used: top {used_k}" if row_random is None else f"Images used: top {used_k} | random {used_k}"
        # draw.text((grid.width // 2, grid.height - footer_h // 2), footer, fill=(0, 0, 0, 255), anchor="ma", font=body_font)

        if args.cbar:
            canvas = Image.new("RGBA", (grid.width + cb_img.width + 20, grid.height), color=(255, 255, 255, 0))
            canvas.paste(grid, (0, 0), grid)
            canvas.paste(cb_img, (grid.width + 10, (canvas.height - cb_img.height) // 2), cb_img)
            frames.append(canvas)
        else:
            frames.append(grid)

    if args.reverse:
        frames = frames[::-1]

    output_name = "_".join(roi_labels)
    filepath_out = args.output or f"{pth.SAVEDIR}/timextime/make_gif/{output_name}.gif"
    if args.save:
        write_gif(filepath_out=filepath_out, frames=frames, duration=args.duration, loop=args.loop)
        vprint(f"Saved gif to {filepath_out}")

    filepath_local = Path.home() / "Downloads" / f"{output_name}.gif"
    write_gif(filepath_out=str(filepath_local), frames=frames, duration=args.duration, loop=args.loop)
    vprint(f"Saved local gif to {filepath_local}")


if __name__ == "__main__":
    main()
