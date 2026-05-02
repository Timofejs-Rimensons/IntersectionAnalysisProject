import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def load_random_video_frame(video_path, seed=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to read any frame from {video_path}")
        return frame, 0

    rng = random.Random(seed)
    margin = max(1, int(total * 0.1))
    lo = margin
    hi = max(lo, total - margin - 1)
    target = rng.randint(lo, hi) if hi > lo else 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        target = 0

    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read frame {target} from {video_path}")
    return frame, target


def load_visual_frame(source_path, seed=None):
    source = Path(source_path)
    if source.is_dir():
        images = sorted(
            [p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        )
        if not images:
            raise FileNotFoundError(f"No image files in directory: {source}")
        chosen = random.Random(seed).choice(images)
        frame = cv2.imread(str(chosen))
        if frame is None:
            raise RuntimeError(f"Unable to read image: {chosen}")
        return frame, str(chosen)

    if source.is_file():
        suffix = source.suffix.lower()
        if suffix in IMAGE_EXTS:
            frame = cv2.imread(str(source))
            if frame is None:
                raise RuntimeError(f"Unable to read image: {source}")
            return frame, str(source)
        if suffix in VIDEO_EXTS:
            frame, frame_idx = load_random_video_frame(source, seed=seed)
            return frame, f"{source}#frame={frame_idx}"
        cap = cv2.VideoCapture(str(source))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to read frame from: {source}")
        return frame, str(source)

    raise FileNotFoundError(f"Source does not exist: {source}")


def is_video_file(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_image_sequence_dir(path):
    p = Path(path)
    if not p.is_dir():
        return False
    for f in p.iterdir():
        if f.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def collect_targets(input_path):
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        return [p]

    if p.is_dir():
        if is_image_sequence_dir(p) and not any(
            sub.is_dir() and is_image_sequence_dir(sub) for sub in p.iterdir()
        ) and not any(f.suffix.lower() in VIDEO_EXTS for f in p.iterdir()):
            return [p]

        targets = []
        for entry in sorted(p.iterdir()):
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                targets.append(entry)
            elif entry.is_dir() and is_image_sequence_dir(entry):
                targets.append(entry)

        if not targets:
            raise RuntimeError(
                f"No videos or image-sequence subdirectories found in {p}"
            )
        return targets

    raise FileNotFoundError(f"Unsupported path: {p}")


class LineSelector:
    def __init__(self, image, title="Select line points", max_lines=None):
        self.image = image.copy()
        self.current_canvas = image.copy()
        self.title = title
        self.points = []
        self.lines = []
        self.max_lines = max_lines
        self.done = False
        self.skip = False
        self.fig = None
        self.ax = None

    def _draw(self):
        self.current_canvas = self.image.copy()
        for line in self.lines:
            cv2.line(self.current_canvas, tuple(line[0]), tuple(line[1]), (0, 255, 0), 2)
            cv2.circle(self.current_canvas, tuple(line[0]), 4, (0, 255, 0), -1)
            cv2.circle(self.current_canvas, tuple(line[1]), 4, (0, 255, 0), -1)
        for idx, pt in enumerate(self.points):
            cv2.circle(self.current_canvas, tuple(pt), 4, (0, 0, 255), -1)
            cv2.putText(self.current_canvas, str(idx + 1), tuple(pt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        msg = f"Lines: {len(self.lines)} / {self.max_lines if self.max_lines else 'inf'}"
        cv2.putText(self.current_canvas, msg, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.current_canvas, "c/q=done  r=reset  s=skip  n=new frame  h=help",
                    (10, self.current_canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_matplotlib(self):
        if self.ax is None:
            return
        self.ax.clear()
        image = self.image
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.ax.imshow(image)
        for line in self.lines:
            xs = [line[0][0], line[1][0]]
            ys = [line[0][1], line[1][1]]
            self.ax.plot(xs, ys, color='lime', linewidth=2)
            self.ax.scatter(xs, ys, c='lime', s=30)
        for idx, pt in enumerate(self.points):
            self.ax.scatter(pt[0], pt[1], c='red', s=40)
            self.ax.text(pt[0], pt[1], str(idx + 1), color='red',
                         fontsize=12, weight='bold')
        msg = f"Lines: {len(self.lines)} / {self.max_lines if self.max_lines else 'inf'}"
        self.ax.set_title(msg + "  (c/q=done, r=reset, s=skip, h=help)")
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def update_image(self, image):
        self.image = image.copy()
        self.points = []
        self._draw()
        if self.fig is not None:
            self._draw_matplotlib()

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.max_lines and len(self.lines) >= self.max_lines:
                return
            self.points.append((x, y))
            if len(self.points) == 2:
                self.lines.append((self.points[0], self.points[1]))
                print(f"  Added line {len(self.lines)}: {self.lines[-1]}")
                self.points = []
            self._draw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                print("  Removed last point")
            elif self.lines:
                removed = self.lines.pop()
                print(f"  Removed last line: {removed}")
            self._draw()

    def _matplotlib_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.button == 1:
            if self.max_lines and len(self.lines) >= self.max_lines:
                return
            x, y = int(event.xdata), int(event.ydata)
            self.points.append((x, y))
            if len(self.points) == 2:
                self.lines.append((self.points[0], self.points[1]))
                print(f"  Added line {len(self.lines)}: {self.lines[-1]}")
                self.points = []
            self._draw_matplotlib()
        elif event.button == 3:
            if self.points:
                self.points.pop()
                print("  Removed last point")
            elif self.lines:
                removed = self.lines.pop()
                print(f"  Removed last line: {removed}")
            self._draw_matplotlib()

    def _matplotlib_key(self, event):
        if event.key in ['q', 'c']:
            self.done = True
            plt.close(self.fig)
        elif event.key == 's':
            self.skip = True
            self.done = True
            plt.close(self.fig)
        elif event.key == 'r':
            self.points = []
            self.lines = []
            self._draw_matplotlib()
            print("  Reset all lines")
        elif event.key == 'h':
            print("Controls: left=add point, right=undo, c/q=done, "
                  "r=reset, s=skip, n=new frame (opencv only)")

    def run(self, backend='auto', new_frame_callback=None):
        if backend == 'auto':
            try:
                return self.run(backend='opencv', new_frame_callback=new_frame_callback)
            except RuntimeError:
                if plt is not None:
                    print("OpenCV GUI unavailable, using matplotlib backend.")
                    return self.run(backend='matplotlib',
                                    new_frame_callback=new_frame_callback)
                raise

        if backend == 'opencv':
            try:
                cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            except cv2.error as exc:
                raise RuntimeError(
                    "OpenCV GUI unavailable. Use --backend matplotlib."
                ) from exc
            cv2.setMouseCallback(self.title, self._mouse_callback)
            self._draw()
            print("Controls: left-click=add point, right-click=undo, "
                  "c/q=done, r=reset, s=skip target, n=new frame, h=help")

            while not self.done:
                cv2.imshow(self.title, self.current_canvas)
                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    continue
                if key in (ord('q'), ord('c')):
                    self.done = True
                elif key == ord('s'):
                    self.skip = True
                    self.done = True
                elif key == ord('r'):
                    self.points = []
                    self.lines = []
                    self._draw()
                    print("  Reset all lines")
                elif key == ord('n') and new_frame_callback is not None:
                    new_img = new_frame_callback()
                    if new_img is not None:
                        self.update_image(new_img)
                        print("  Loaded new random frame")
                elif key == ord('h'):
                    print("Controls: left=add point, right=undo, "
                          "c/q=done, r=reset, s=skip target, n=new frame")

            cv2.destroyWindow(self.title)
            return self.lines, self.skip

        if backend == 'matplotlib':
            if plt is None:
                raise RuntimeError("matplotlib not installed.")
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self._draw_matplotlib()
            self.fig.canvas.mpl_connect('button_press_event', self._matplotlib_click)
            self.fig.canvas.mpl_connect('key_press_event', self._matplotlib_key)
            print("Controls: left=add point, right=undo, c/q=done, r=reset, s=skip")
            plt.show()
            return self.lines, self.skip

        raise ValueError(f"Unsupported backend: {backend}")


def load_existing_boundaries(output_path):
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
                if 'sequences' not in data:
                    data['sequences'] = {}
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {'sequences': {}}


def save_boundaries(output_path, boundaries):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(boundaries, f, indent=2)


def add_boundary_entry(boundaries, seq_name, lines, source_path, image_shape,
                       normalize_lines=True):
    if normalize_lines:
        height, width = image_shape[:2]
        stored_lines = [
            [[round(x1 / width, 6), round(y1 / height, 6)],
             [round(x2 / width, 6), round(y2 / height, 6)]]
            for (x1, y1), (x2, y2) in lines
        ]
    else:
        stored_lines = [
            [[int(x1), int(y1)], [int(x2), int(y2)]]
            for (x1, y1), (x2, y2) in lines
        ]

    boundaries.setdefault('sequences', {})[seq_name] = {
        'source': source_path,
        'normalized': normalize_lines,
        'lines': stored_lines,
    }
    return boundaries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual line selector for cross-section boundaries"
    )
    parser.add_argument('--input', type=str, default=None,
                        help='File OR directory. If a directory, every video '
                             'or image-sequence inside is processed in order.')
    parser.add_argument('--seq-dir', type=str,
                        default='datasets/VisDrone2019-MOT-train/sequences',
                        help='Default base folder if --input not given')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Single sequence name (resolved against --seq-dir)')
    parser.add_argument('--out', type=str,
                        default='datasets/crossection_boundaries.json',
                        help='Output JSON file (one file for all targets)')
    parser.add_argument('--max-lines', type=int, default=None)
    parser.add_argument('--normalize-lines', dest='normalize_lines',
                        action='store_true', default=True)
    parser.add_argument('--no-normalize-lines', dest='normalize_lines',
                        action='store_false')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'opencv', 'matplotlib'])
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible frame sampling')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-prompt for sequences that already exist in JSON')
    parser.add_argument('--list-sequences', action='store_true',
                        help='List subdirectories of --seq-dir and exit')
    return parser.parse_args()


def resolve_input_path(args):
    if args.input:
        return args.input
    if args.sequence:
        candidate = Path(args.sequence)
        if candidate.exists():
            return str(candidate)
        return os.path.join(args.seq_dir, args.sequence)
    return args.seq_dir


def process_target(target, boundaries, args, idx, total):
    target = Path(target)
    seq_name = target.stem if target.is_file() else target.name

    print(f"\n[{idx}/{total}] Target: {target}")
    print(f"        Saving as: '{seq_name}'")

    if not args.overwrite and seq_name in boundaries.get('sequences', {}):
        print(f"        Already in JSON, skipping (use --overwrite to redo)")
        return False

    seed = args.seed
    try:
        frame, resolved_path = load_visual_frame(str(target), seed=seed)
    except Exception as e:
        print(f"        Failed to load frame: {e}")
        return False

    if is_video_file(target):
        def new_frame_cb():
            try:
                f, _ = load_random_video_frame(target, seed=None)
                return f
            except Exception as exc:
                print(f"        Could not sample new frame: {exc}")
                return None
    elif target.is_dir():
        def new_frame_cb():
            try:
                f, _ = load_visual_frame(str(target), seed=None)
                return f
            except Exception as exc:
                print(f"        Could not sample new frame: {exc}")
                return None
    else:
        new_frame_cb = None

    selector = LineSelector(
        frame,
        title=f"[{idx}/{total}] {seq_name}",
        max_lines=args.max_lines,
    )
    lines, skipped = selector.run(
        backend=args.backend,
        new_frame_callback=new_frame_cb,
    )

    if skipped:
        print(f"        Skipped by user")
        return False
    if not lines:
        print(f"        No lines drawn, not saving")
        return False

    add_boundary_entry(
        boundaries=boundaries,
        seq_name=seq_name,
        lines=lines,
        source_path=resolved_path,
        image_shape=frame.shape,
        normalize_lines=args.normalize_lines,
    )
    save_boundaries(args.out, boundaries)
    print(f"        Saved {len(lines)} line(s) -> {args.out}")
    return True


def main():
    args = parse_args()

    if args.list_sequences:
        seq_dir = Path(args.seq_dir)
        if not seq_dir.exists():
            print(f"Sequence directory not found: {seq_dir}")
            sys.exit(1)
        items = sorted([p.name for p in seq_dir.iterdir()
                        if p.is_dir() or p.suffix.lower() in VIDEO_EXTS])
        print("Available targets:")
        for item in items:
            print(f"  - {item}")
        return

    input_path = resolve_input_path(args)
    if input_path is None:
        print("Provide --input, --sequence, or --seq-dir")
        sys.exit(1)

    targets = collect_targets(input_path)
    print(f"Found {len(targets)} target(s) to label")

    boundaries = load_existing_boundaries(args.out)
    print(f"Existing entries in JSON: {len(boundaries.get('sequences', {}))}")

    saved_count = 0
    for i, target in enumerate(targets, start=1):
        try:
            if process_target(target, boundaries, args, i, len(targets)):
                saved_count += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"        Error processing {target}: {e}")
            continue

    print(f"\nDone. {saved_count}/{len(targets)} target(s) labelled this session.")
    print(f"Total entries in {args.out}: {len(boundaries.get('sequences', {}))}")


if __name__ == '__main__':
    main()