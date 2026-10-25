"""
Flip between the panels of a figure (or between whole images) in one
window, to compare them by eye: differences stand out far better when one
image replaces another in place than when the two sit side by side.

Given one figure, the script finds its image panels and arranges them in
rows and columns as laid out; it flips across the panels of a row (say, one
crater processed several ways) and steps between rows. Each panel is shown
with its title and tick labels; panels are lined up on their image areas,
and a panel can be nudged by hand or matched to the previous one
automatically. Given several image files, it flips between the whole
images.

Keys (h shows them in the window):
  space or n / backspace or b   next / previous panel in the row; 1-9 jump
  ] or page down / [ or page up  next / previous row
  t        blink between the last two panels shown, every --interval s (t stops)
  arrows   nudge this panel one figure pixel (with shift: five)
  m        match this panel to the one shown before it (shift found by correlation)
  0        undo this panel's nudge;  p  print every nudge
  h        show or hide this list;  q  quit (prints the nudges)
Zoom and pan with the toolbar; the view stays put as you flip.

Usage (from the repo root):
    .conda/bin/python flip_compare.py results/EVE2026_CIRC/focus_check_stockert_wsrt_txmin100s_dual_crops.png
    .conda/bin/python flip_compare.py before.png after.png       # whole images
"""

import argparse

import numpy as np
import scipy.ndimage
import matplotlib

MIN_PANEL_FRACTION = 0.25   # panels smaller than this x the largest (colour bars) are ignored
MAX_MATCH_SHIFT = 15        # figure pixels searched by the m key
KEYS = __doc__[__doc__.index("Keys (h"):__doc__.index("\nUsage")].split("\n", 1)[1].rstrip()


def read_rgb(path):
    from matplotlib import image as mpimg
    img = mpimg.imread(path)
    if img.dtype == np.uint8:
        img = img / 255.0
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] == 4:                                  # onto white
        a = img[..., 3:]
        img = img[..., :3] * a + (1 - a)
    return img[..., :3].astype(np.float32)


def find_panels(img, grid=None):
    """Image areas of the figure's panels as (top, bottom, left, right)
    pixel boxes, grouped into rows (lists ordered left to right)."""
    h, w = img.shape[:2]
    if grid:
        nr, nc = grid
        return [[(r * h // nr, (r + 1) * h // nr, c * w // nc, (c + 1) * w // nc)
                 for c in range(nc)] for r in range(nr)]
    ink = img.min(axis=2) < 0.94
    solid = scipy.ndimage.binary_opening(scipy.ndimage.binary_closing(ink, np.ones((5, 5))),
                                         np.ones((21, 21)))
    lab, n = scipy.ndimage.label(solid)
    boxes = [(s[0].start, s[0].stop, s[1].start, s[1].stop) for s in scipy.ndimage.find_objects(lab)]
    # merge pieces of one panel (a white region can cut an image apart)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[1] + 10 and b[0] < a[1] + 10 and a[2] < b[3] + 10 and b[2] < a[3] + 10:
                    boxes[i] = (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    area = np.array([(b[1] - b[0]) * (b[3] - b[2]) for b in boxes])
    boxes = [b for b, a in zip(boxes, area) if a >= MIN_PANEL_FRACTION * area.max()]
    if not boxes:
        raise SystemExit("no panels found; give --grid ROWS COLS")
    rows = []
    for b in sorted(boxes):
        mid = (b[0] + b[1]) / 2
        for row in rows:
            if row[0][0] <= mid <= row[0][1]:
                row.append(b)
                break
        else:
            rows.append([b])
    return [sorted(row, key=lambda b: b[2]) for row in rows]


class Panel:
    def __init__(self, label, img, box, margins):
        """img: the whole figure; box: the panel's image area (top, bottom,
        left, right); margins: how far the crop reaches beyond it on each side."""
        t, b, l, r = box
        mt, mb, ml, mr = margins
        h, w = img.shape[:2]
        t0, b0, l0, r0 = max(0, t - mt), min(h, b + mb), max(0, l - ml), min(w, r + mr)
        self.label = label
        self.crop = img[t0:b0, l0:r0][::-1]                # bottom row first (origin lower)
        self.core = img[t:b, l:r].mean(axis=2)             # image area, grey, for matching
        # the image area's bottom-left corner at (0, 0), y up, in figure pixels
        self.extent = [l0 - l, r0 - l, b - b0, b - t0]
        self.offset = np.zeros(2)
        self.im = None


def build_panels(paths, grid, by_column):
    if len(paths) > 1:
        return [[Panel(p, im, (0, im.shape[0], 0, im.shape[1]), (0, 0, 0, 0))
                 for p, im in ((p, read_rgb(p)) for p in paths)]]
    img = read_rgb(paths[0])
    rows = find_panels(img, grid)
    # Crops reach into the gaps between panels for the panel's own labels:
    # the title just above (2/3 of the gap), x tick labels just below (1/3),
    # y tick labels to the left (all but 3 pixels of the gap); nothing right.
    gaps_v = [r2[0][0] - r1[0][1] for r1, r2 in zip(rows, rows[1:])]
    gaps_h = [b[2] - a[3] for row in rows for a, b in zip(row, row[1:])]
    gv = int(np.median(gaps_v)) if gaps_v else 45
    gh = int(np.median(gaps_h)) if gaps_h else 40
    panels = [[Panel(f"row {i + 1}, column {j + 1}", img, b, (gv * 2 // 3, gv // 3, gh - 3, 3))
               for j, b in enumerate(row)] for i, row in enumerate(rows)]
    if by_column:
        ncol = max(len(r) for r in panels)
        panels = [[r[j] for r in panels if j < len(r)] for j in range(ncol)]
    return panels


def match_shift(ref, cur):
    """(dx, dy) in display units (x right, y up) that moves cur's content
    onto ref's."""
    h, w = min(ref.shape[0], cur.shape[0]), min(ref.shape[1], cur.shape[1])
    a, b = ref[:h, :w] - ref[:h, :w].mean(), cur[:h, :w] - cur[:h, :w].mean()
    win = np.outer(np.hanning(h), np.hanning(w))
    xc = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(a * win) * np.conj(np.fft.fft2(b * win))).real)
    cy, cx, s = h // 2, w // 2, MAX_MATCH_SHIFT
    iy, ix = np.unravel_index(np.argmax(xc[cy - s:cy + s + 1, cx - s:cx + s + 1]), (2 * s + 1, 2 * s + 1))
    # ref(y, x) ~ cur(y - dy_rows, x - dx): move cur right by dx, down by dy_rows
    return float(ix - s), float(-(iy - s))


class Flipper:
    def __init__(self, fig, ax, rows, args):
        self.fig, self.ax, self.rows, self.args = fig, ax, rows, args
        self.r, self.c = 0, 0
        self.prev = (0, min(1, len(rows[0]) - 1))
        for row in rows:
            for p in row:
                p.im = ax.imshow(p.crop, extent=p.extent, origin="lower", visible=False,
                                 interpolation="nearest")
        ext = np.array([p.extent for row in rows for p in row])
        ax.set_xlim(ext[:, 0].min(), ext[:, 1].max())
        ax.set_ylim(ext[:, 2].min(), ext[:, 3].max())
        ax.set_axis_off()
        self.timer = fig.canvas.new_timer(interval=int(1000 * args.interval))
        self.timer.add_callback(self.blink)
        self.blinking = False
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        fig.canvas.mpl_connect("close_event", lambda ev: self.print_nudges())
        fig.text(0.01, 0.01, "h: keys", fontsize=9, color="0.3")
        self.help = ax.text(0.5, 0.5, KEYS, transform=ax.transAxes, ha="center", va="center", multialignment="left",
                            family="monospace", fontsize=9, visible=False, zorder=10,
                            bbox=dict(facecolor="white", alpha=0.92, edgecolor="0.5"))
        print(KEYS)
        self.show(0, 0)

    @property
    def panel(self):
        return self.rows[self.r][self.c]

    def show(self, r, c):
        c = min(c, len(self.rows[r]) - 1)
        if (r, c) != (self.r, self.c):
            self.prev = (self.r, self.c)
        self.panel.im.set_visible(False)
        self.r, self.c = r, c
        p = self.panel
        p.im.set_visible(True)
        title = f"{p.label}    ({c + 1} of {len(self.rows[r])} in this row; row {r + 1} of {len(self.rows)})"
        if p.offset.any():
            title += f"    nudged {p.offset[0]:+.0f}, {p.offset[1]:+.0f} px"
        self.ax.set_title(title, fontsize=9)
        self.fig.canvas.draw_idle()

    def blink(self):
        self.show(*self.prev)

    def set_offset(self, offset):
        p = self.panel
        p.offset = np.asarray(offset, dtype=float)
        x0, x1, y0, y1 = p.extent
        p.im.set_extent([x0 + p.offset[0], x1 + p.offset[0], y0 + p.offset[1], y1 + p.offset[1]])
        self.show(self.r, self.c)

    def print_nudges(self):
        for row in self.rows:
            for p in row:
                if p.offset.any():
                    print(f"  {p.label}: nudged {p.offset[0]:+.0f}, {p.offset[1]:+.0f} px (right, up)")

    def on_key(self, ev):
        key = ev.key or ""
        n = len(self.rows[self.r])
        if key in (" ", "n"):
            self.show(self.r, (self.c + 1) % n)
        elif key in ("backspace", "b"):
            self.show(self.r, (self.c - 1) % n)
        elif key.isdigit() and key != "0" and int(key) <= n:
            self.show(self.r, int(key) - 1)
        elif key in ("]", "pagedown", "[", "pageup"):
            step = 1 if key in ("]", "pagedown") else -1
            self.show((self.r + step) % len(self.rows), self.c)
        elif key == "h":
            self.help.set_visible(not self.help.get_visible())
            self.fig.canvas.draw_idle()
        elif key == "t":
            self.blinking = not self.blinking
            (self.timer.start if self.blinking else self.timer.stop)()
        elif key.split("+")[-1] in ("left", "right", "up", "down"):
            step = 5 if key.startswith("shift+") else 1
            dx, dy = {"left": (-step, 0), "right": (step, 0), "up": (0, step),
                      "down": (0, -step)}[key.split("+")[-1]]
            self.set_offset(self.panel.offset + (dx, dy))
        elif key == "m":
            ref = self.rows[self.prev[0]][self.prev[1]]
            if ref is not self.panel:
                self.set_offset(ref.offset + match_shift(ref.core, self.panel.core))
        elif key == "0":
            self.set_offset((0, 0))
        elif key == "p":
            self.print_nudges()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+",
                        help="one figure (its panels are flipped) or several images (flipped whole)")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("ROWS", "COLS"),
                        help="split the figure into an even grid instead of finding its panels")
    parser.add_argument("--by-column", action="store_true",
                        help="flip down the columns instead of across the rows")
    parser.add_argument("--interval", type=float, default=0.6, help="blink interval, s")
    parser.add_argument("--save", metavar="PREFIX",
                        help="write PREFIX_r<row>c<col>.png for each panel instead of opening a window")
    args = parser.parse_args()
    if args.save:
        matplotlib.use("Agg")
    for name, ours in (("keymap.back", ("left", "backspace")), ("keymap.forward", ("right",)),
                       ("keymap.pan", ("p",)), ("keymap.home", ("h",))):
        matplotlib.rcParams[name] = [k for k in matplotlib.rcParams[name] if k not in ours]
    from matplotlib import pyplot as pl

    rows = build_panels(args.images, args.grid, args.by_column)
    print(f"{sum(len(r) for r in rows)} panels in {len(rows)} rows: "
          + ", ".join(str(len(r)) for r in rows))
    fig, ax = pl.subplots(figsize=(8, 8))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.95)
    viewer = Flipper(fig, ax, rows, args)
    if args.save:
        for i, row in enumerate(rows):
            for j in range(len(row)):
                viewer.show(i, j)
                fig.savefig(f"{args.save}_r{i + 1}c{j + 1}.png", dpi=100)
        print(f"Saved {args.save}_r*c*.png")
        return
    pl.show()


if __name__ == "__main__":
    main()
