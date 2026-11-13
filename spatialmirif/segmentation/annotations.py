"""General polygon/freehand/rectangle annotation with reusable geometry metadata.

Annotation masks are always stored separately from raw IF or automatic masks.
An optional interactive OpenCV editor runs locally (not in headless batch mode).
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from skimage.draw import polygon as raster_polygon
from ._stage import Stage, as_2d, grid_info, check_same_grid


def polygons_to_mask(shape_yx, polygons_xy):
    """Rasterize multiple polygons whose vertices use (x,y) image coordinates."""
    output = np.zeros(shape_yx, bool)
    for points in polygons_xy:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError('Each polygon must have at least three (x,y) vertices')
        rr, cc = raster_polygon(pts[:, 1], pts[:, 0], shape=output.shape)
        output[rr, cc] = True
    return output


def save_annotation(mask, reference, folder, *, sample_name='sample',
                    kind='exclusion', polygons_xy=None, shapes=None, source_sha256=None):
    """Save mask, image preview, full geometry, and optional editable vertices."""
    if kind not in ('addition', 'exclusion'):
        raise ValueError("kind must be 'addition' or 'exclusion'")
    s = Stage(folder, reference)
    ann = as_2d(mask).astype(bool)
    s.mask('annotation_mask', ann)
    s.preview('annotation_mask', ann)
    s.json('annotation', {'schema_version': 1, 'sample_name': sample_name,
                          'kind': kind, 'geometry': grid_info(reference),
                          'pixels': int(ann.sum()), 'polygons_xy': polygons_xy or [],
                          'shapes': shapes or [],
                          'source_sha256': source_sha256})
    return s.path


def load_annotation(folder, reference, *, sample_name=None, kind=None):
    """Refuse to reuse masks for another image grid/sample."""
    import SimpleITK as sitk
    folder = Path(folder)
    metadata_path = folder / 'annotation.json'
    path = folder / 'annotation_mask.nrrd'
    if not metadata_path.exists() or not path.exists():
        raise FileNotFoundError(f'No saved annotation pair in {folder}')
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    if meta.get('schema_version') != 1:
        raise ValueError('Unsupported annotation schema version')
    if sample_name is not None and meta.get('sample_name') != sample_name:
        raise ValueError('Annotation belongs to a different sample')
    if kind is not None and meta.get('kind') != kind:
        raise ValueError('Wrong annotation type')
    if meta.get('geometry') != grid_info(reference):
        raise ValueError('Saved annotation metadata does not match current image geometry')
    img = sitk.ReadImage(str(path))
    if not check_same_grid(img, reference):
        raise ValueError('Saved annotation NRRD and IF grid mismatch')
    return as_2d(img).astype(bool)


def apply_additions(automatic_mask, additions, *, reference=None, save_dir=None,
                    image=None, sample_name="sample"):
    auto, extra = as_2d(automatic_mask).astype(bool), as_2d(additions).astype(bool)
    if auto.shape != extra.shape: raise ValueError('Masks must have same shape')
    corrected = auto | extra
    stage = Stage(save_dir, reference)
    stage.mask('tissue_additions', extra)
    stage.mask('added_new_tissue_only', corrected & ~auto)
    stage.mask('corrected_tissue', corrected)
    stage.preview('corrected_tissue', corrected)
    stage.json('summary', {'pixels_added': int((corrected & ~auto).sum()),
                           'total_tissue_pixels': int(corrected.sum())})
    if image is not None and stage.path is not None:
        # Original S1_N3 three-panel QC figure, saved BY THIS FUNCTION.
        import matplotlib.pyplot as plt
        from .visualization import robust_display
        raw = as_2d(image)
        if raw.shape != auto.shape:
            raise ValueError('IF image and tissue mask have different shapes')
        new_tissue = corrected & ~auto
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=90)
        axes[0].imshow(auto, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Automatic tissue mask')
        axes[1].imshow(np.clip(robust_display(raw), 0, 1),
                       cmap='gray', vmin=0, vmax=1)
        rgba = np.zeros((*raw.shape, 4), dtype=np.uint8)
        rgba[new_tissue] = [255, 255, 0, 220]
        axes[1].imshow(rgba)
        axes[1].set_title(f'New tissue additions (yellow): {int(new_tissue.sum()):,} px')
        axes[2].imshow(corrected, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Corrected tissue mask')
        for ax in axes: ax.axis('off')
        fig.suptitle(f'{sample_name} — Polygon tissue correction', fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(stage.path / f'{sample_name}_polygon_correction_comparison.png',
                    dpi=140, bbox_inches='tight')
        plt.close(fig)
    return corrected


def apply_exclusions(candidate_mask, exclusion_mask, *, buffer_um=0.0,
                     spacing_xy_um=None, reference=None, save_dir=None):
    from scipy import ndimage as ndi
    candidate = as_2d(candidate_mask).astype(bool)
    exclusion = as_2d(exclusion_mask).astype(bool)
    if candidate.shape != exclusion.shape: raise ValueError('Mask grid mismatch')
    if buffer_um < 0: raise ValueError('buffer_um must be nonnegative')
    if buffer_um:
        if spacing_xy_um is None: raise ValueError('Spacing required for nonzero buffer')
        sx, sy = map(float, spacing_xy_um)
        if sx <= 0 or sy <= 0: raise ValueError('Invalid spacing')
        # Compute physical distance from any excluded pixel.
        exclusion = ndi.distance_transform_edt(~exclusion, sampling=(sy, sx)) <= buffer_um
    final = candidate & ~exclusion
    stage = Stage(save_dir, reference)
    stage.mask('effective_exclusions', exclusion)
    stage.mask('filtered_candidates', final)
    stage.preview('effective_exclusions', exclusion)
    stage.preview('filtered_candidates', final)
    stage.json('summary', {'buffer_um': buffer_um,
                           'excluded_area_pixels': int(exclusion.sum()),
                           'removed_candidate_pixels': int((candidate & exclusion).sum())})
    return {'mask': final, 'exclusion': exclusion}


def edit_annotation(image, *, starting_mask=None, starting_shapes=None, mode='polygon',
                    title='IF annotation', return_shapes=False):
    """Interactive local OpenCV editor. Returns a mask; does not write files.

    Keys: P polygon, R rectangle, F freehand, Z undo, C clear, Enter save,
    Esc/Q cancel. Left click/draw adds; right click closes polygon (or pans
    while held). Mouse wheel zooms around cursor. Returns bool mask.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError('Install optional opencv-python to draw annotations') from exc
    from .visualization import robust_display
    a = as_2d(image)
    gray = np.asarray(np.rint(robust_display(a) * 255), dtype=np.uint8)
    h, w = gray.shape
    strokes = []
    # Retain editable shape history when reopening a saved annotation.
    shapes = [] if starting_shapes is None else json.loads(json.dumps(starting_shapes))
    mask = np.zeros((h, w), dtype=np.uint8) if starting_mask is None else (as_2d(starting_mask) > 0).astype(np.uint8) * 255
    if mask.shape != gray.shape: raise ValueError('Starting annotation shape mismatch')
    initial = mask.copy()
    selected = {'mode': mode, 'pending': [], 'start': None, 'trail': [], 'pan': None}
    view = {'scale': min(1.0, 1200 / w, 850 / h), 'x': 0., 'y': 0.}
    win_w, win_h = max(1, int(round(w * view['scale']))), max(1, int(round(h * view['scale'])))

    def clamp():
        view['scale'] = float(np.clip(view['scale'], 0.05, 10.0))
        view['x'] = float(np.clip(view['x'], 0, max(0, w - win_w / view['scale'])))
        view['y'] = float(np.clip(view['y'], 0, max(0, h - win_h / view['scale'])))

    def to_original(x, y):
        return (int(np.clip(round(view['x'] + x / view['scale']), 0, w - 1)),
                int(np.clip(round(view['y'] + y / view['scale']), 0, h - 1)))

    def commit(new_mask, shape=None):
        strokes.append((mask.copy(), list(shapes)))
        mask[:] = np.maximum(mask, new_mask)
        if shape is not None:
            shapes.append(shape)

    def close_polygon():
        if len(selected['pending']) >= 3:
            overlay = np.zeros_like(mask)
            cv2.fillPoly(overlay, [np.asarray(selected['pending'], np.int32)], 255)
            commit(overlay, {'mode':'polygon','points_xy':[list(map(int, p)) for p in selected['pending']]})
        selected['pending'].clear()

    def handler(event, x, y, flags, userdata):
        pt = to_original(x, y)
        if event == cv2.EVENT_MOUSEWHEEL:
            old = view['scale']; view['scale'] *= 1.25 if flags > 0 else 1 / 1.25
            clamp()
            view['x'] += x / old - x / view['scale']
            view['y'] += y / old - y / view['scale']; clamp()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if selected['mode'] == 'polygon' and selected['pending']:
                close_polygon()
            else:
                selected['pan'] = (x, y, view['x'], view['y'])
        elif event == cv2.EVENT_MOUSEMOVE and selected['pan'] is not None and flags & cv2.EVENT_FLAG_RBUTTON:
            sx, sy, ox, oy = selected['pan']
            view['x'] = ox - (x - sx) / view['scale']; view['y'] = oy - (y - sy) / view['scale']; clamp()
        elif event == cv2.EVENT_RBUTTONUP:
            selected['pan'] = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            if selected['mode'] == 'polygon': selected['pending'].append(pt)
            elif selected['mode'] in ('rectangle', 'freehand'):
                selected['start'] = pt; selected['trail'] = [pt]
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON and selected['start']:
            if selected['mode'] == 'freehand': selected['trail'].append(pt)
        elif event == cv2.EVENT_LBUTTONUP and selected['start']:
            layer = np.zeros_like(mask)
            if selected['mode'] == 'rectangle':
                cv2.rectangle(layer, selected['start'], pt, 255, thickness=-1)
            elif selected['mode'] == 'freehand':
                pts = selected['trail'] + [pt]
                if len(pts) > 2: cv2.fillPoly(layer, [np.asarray(pts, np.int32)], 255)
            points = [selected['start'],pt] if selected['mode']=='rectangle' else selected['trail']+[pt]
            commit(layer, {'mode':selected['mode'], 'points_xy':[list(map(int,p)) for p in points]})
            selected['start'] = None; selected['trail'] = []

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, win_w, win_h)
    cv2.setMouseCallback(title, handler)
    try:
        while True:
            clamp()
            src = np.dstack([gray, gray, gray])
            src[mask > 0] = (0.6 * src[mask > 0] + 0.4 * np.asarray([0, 205, 255])).astype(np.uint8)
            x0, y0 = int(view['x']), int(view['y'])
            roi_w = max(1, int(np.ceil(win_w / view['scale'])))
            roi_h = max(1, int(np.ceil(win_h / view['scale'])))
            frame = cv2.resize(src[y0:min(h,y0+roi_h), x0:min(w,x0+roi_w)], (win_w, win_h), interpolation=cv2.INTER_LINEAR)
            points = [(int(round((px-view['x'])*view['scale'])), int(round((py-view['y'])*view['scale']))) for px,py in selected['pending']]
            for start,end in zip(points,points[1:]): cv2.line(frame,start,end,(0,0,255),1)
            for pt in points: cv2.circle(frame,pt,3,(0,0,255),-1)
            cv2.putText(frame, f"{selected['mode']}: P/R/F | Z undo | C clear | Enter save | Esc cancel", (6,20), cv2.FONT_HERSHEY_SIMPLEX,.40,(50,255,50),1)
            cv2.imshow(title, frame)
            key = cv2.waitKey(20) & 0xff
            if key in (13, 10):
                if selected['pending']: close_polygon()
                return (mask.astype(bool), list(shapes)) if return_shapes else mask.astype(bool)
            if key in (27, ord('q')): raise RuntimeError('Annotation editing cancelled; no changes saved')
            if key == ord('p'): selected['mode'] = 'polygon'; selected['pending'].clear()
            if key == ord('r'): selected['mode'] = 'rectangle'; selected['pending'].clear()
            if key == ord('f'): selected['mode'] = 'freehand'; selected['pending'].clear()
            if key == ord('z') and strokes:
                old_mask, old_shapes = strokes.pop()
                mask[:] = old_mask; shapes[:] = old_shapes
            if key == ord('c'):
                strokes.append((mask.copy(), list(shapes)))
                mask[:] = 0; shapes.clear(); selected['pending'].clear()
    finally:
        cv2.destroyWindow(title)




def edit_tissue_additions(image, automatic_tissue_mask, *, starting_mask=None,
                          starting_shapes=None, title='Add tissue (polygon)',
                          return_shapes=False):
    """Original S1_N3 tissue-addition editor: **polygon only**.

    Green = automatically detected tissue; yellow = manual additions.

    Controls (matching the original S1_N3 notebook exactly):
      Left-click: add a polygon vertex; right-click / Enter: CLOSE polygon.
      Mouse wheel / +/-: zoom; middle-drag or H+left-drag: pan;
      P: polygon mode; 0: fit image; U: undo pending/last NEW polygon;
      R: clear ALL additions (including previously saved additions);
      S: save; Q/Esc: cancel, leaving saved files unchanged.

    This editor changes a new *annotation mask*, NEVER the IF pixels or the
    automatic tissue mask. Reusing a saved annotation does not open a window.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError('Install opencv-python (GUI build) to edit tissue polygons') from exc
    from .visualization import robust_display

    raw = as_2d(image)
    automatic = as_2d(automatic_tissue_mask).astype(bool)
    if automatic.shape != raw.shape:
        raise ValueError('Automatic tissue mask and IF must have identical 2D shapes')
    if starting_mask is None:
        saved_additions = np.zeros(raw.shape, dtype=np.uint8)
    else:
        previous = as_2d(starting_mask)
        if previous.shape != raw.shape:
            raise ValueError('Starting polygon annotation and IF shape mismatch')
        saved_additions = (previous > 0).astype(np.uint8) * 255

    gray = np.rint(np.clip(robust_display(raw), 0, 1) * 255).astype(np.uint8)
    height, width = gray.shape
    auto_u8 = automatic.astype(np.uint8) * 255
    fit = min(1.0, 1400.0 / width, 900.0 / height)
    win_w = max(1, int(round(width * fit)))
    win_h = max(1, int(round(height * fit)))
    view = {'scale': fit, 'x': 0.0, 'y': 0.0, 'panning': False, 'last': None}
    state = {'mode': 'polygon', 'pending': [], 'polygons': [], 'cursor': None}
    existing_shapes = [] if starting_shapes is None else [dict(x) for x in starting_shapes]
    drawn_mask = saved_additions.copy()

    def clamp_view():
        view['x'] = float(np.clip(view['x'], 0,
                            max(0, width - win_w / view['scale'])))
        view['y'] = float(np.clip(view['y'], 0,
                            max(0, height - win_h / view['scale'])))

    def original_xy(x, y):
        return (int(np.clip(np.floor(view['x'] + x / view['scale']), 0, width - 1)),
                int(np.clip(np.floor(view['y'] + y / view['scale']), 0, height - 1)))

    def screen_points(points):
        coords = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return np.rint((coords - [view['x'], view['y']]) * view['scale']).astype(np.int32)

    def zoom_at(factor, x, y):
        old = view['scale']
        new = float(np.clip(old * factor, fit, max(5.0, fit * 12)))
        if new == old:
            return
        px = view['x'] + x / old
        py = view['y'] + y / old
        view['scale'] = new
        view['x'] = px - x / new
        view['y'] = py - y / new
        clamp_view()

    def refresh_mask():
        drawn_mask[:] = saved_additions
        for vertices in state['polygons']:
            cv2.fillPoly(drawn_mask, [np.asarray(vertices, np.int32)], 255)

    def finish_polygon():
        if len(state['pending']) >= 3:
            state['polygons'].append(state['pending'].copy())
            refresh_mask()
        state['pending'].clear()

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xFFFF
            if delta >= 32768:
                delta -= 65536
            if delta:
                zoom_at(1.25 if delta > 0 else 0.8, x, y)
            return
        if event == cv2.EVENT_MBUTTONDOWN or (state['mode'] == 'pan' and event == cv2.EVENT_LBUTTONDOWN):
            view['panning'] = True
            view['last'] = (x, y)
            return
        if event == cv2.EVENT_MBUTTONUP or (state['mode'] == 'pan' and event == cv2.EVENT_LBUTTONUP):
            view['panning'] = False
            view['last'] = None
            return
        if view['panning']:
            if event == cv2.EVENT_MOUSEMOVE and view['last'] is not None:
                dx, dy = x - view['last'][0], y - view['last'][1]
                view['x'] -= dx / view['scale']
                view['y'] -= dy / view['scale']
                clamp_view()
                view['last'] = (x, y)
            return
        state['cursor'] = original_xy(x, y)
        if state['mode'] == 'polygon':
            if event == cv2.EVENT_LBUTTONDOWN:
                state['pending'].append(state['cursor'])
            elif event == cv2.EVENT_RBUTTONDOWN:
                finish_polygon()

    window = title
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, on_mouse)
    except cv2.error as exc:
        raise RuntimeError('OpenCV GUI could not start. Use a local desktop and opencv-python, not opencv-python-headless.') from exc

    print('TISSUE ADDITION — POLYGON ONLY')
    print('LEFT-CLICK vertices | RIGHT-CLICK / ENTER finish polygon')
    print('WHEEL / + / - zoom | MIDDLE-DRAG / H pan | P polygon | 0 fit')
    print('U undo NEW polygon | R clear ALL additions | S save | Q/Esc cancel')
    saved = False
    try:
        while True:
            scale = view['scale']
            mat = np.asarray([[scale, 0, -view['x'] * scale],
                              [0, scale, -view['y'] * scale]], np.float32)
            viewport = cv2.warpAffine(gray, mat, (win_w, win_h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
            auto_view = cv2.warpAffine(auto_u8, mat, (win_w, win_h),
                                       flags=cv2.INTER_NEAREST)
            addition_view = cv2.warpAffine(drawn_mask, mat, (win_w, win_h),
                                           flags=cv2.INTER_NEAREST)
            frame = cv2.cvtColor(viewport, cv2.COLOR_GRAY2BGR)
            auto_pixels = auto_view > 0
            frame[auto_pixels] = (0.60 * frame[auto_pixels] +
                                  0.40 * np.asarray([0, 160, 0])).astype(np.uint8)
            add_pixels = addition_view > 0
            frame[add_pixels] = (0.40 * frame[add_pixels] +
                                 0.60 * np.asarray([0, 255, 255])).astype(np.uint8)
            if state['pending']:
                pts = screen_points(state['pending'])
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, (255, 255, 0), 2)
                for pt in pts:
                    cv2.circle(frame, tuple(pt), 3, (255, 255, 0), -1)
                if state['cursor'] is not None:
                    cursor = tuple(screen_points([state['cursor']])[0])
                    cv2.line(frame, tuple(pts[-1]), cursor, (255, 255, 0), 1)
            cv2.putText(frame,
                        f"{state['mode'].upper()} | {scale / fit:.1f}x | Green=auto Yellow=added",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            cv2.imshow(window, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and state['mode'] == 'polygon':
                finish_polygon()  # NOTE: Enter finishes, it does NOT save.
            elif key == ord('p'):
                state['mode'] = 'polygon'
            elif key == ord('h'):
                state['mode'] = 'pan'
            elif key in (ord('+'), ord('=')):
                zoom_at(1.25, win_w / 2, win_h / 2)
            elif key in (ord('-'), ord('_')):
                zoom_at(0.8, win_w / 2, win_h / 2)
            elif key == ord('0'):
                view.update({'scale': fit, 'x': 0.0, 'y': 0.0})
            elif key == ord('u'):
                if state['pending']:
                    state['pending'].clear()
                elif state['polygons']:
                    state['polygons'].pop()
                    refresh_mask()
            elif key == ord('r'):
                saved_additions.fill(0)
                existing_shapes.clear()
                state['pending'].clear()
                state['polygons'].clear()
                refresh_mask()
            elif key == ord('s'):
                if state['pending']:
                    print('Finish current polygon with Enter/right-click before saving.')
                    continue
                saved = True
                break
            elif key in (ord('q'), 27):
                break
    finally:
        cv2.destroyWindow(window)
    if not saved:
        raise RuntimeError('Tissue polygon editing cancelled. No annotation files were changed.')
    all_shapes = existing_shapes + [
        {'mode': 'polygon', 'points_xy': [list(map(int, point)) for point in points]}
        for points in state['polygons']
    ]
    output_mask = drawn_mask > 0
    return (output_mask, all_shapes) if return_shapes else output_mask


def edit_spatial_exclusions(image, *, starting_mask=None, starting_shapes=None,
                            mode='freehand', title='Manual spatial exclusions',
                            return_shapes=False):
    """Interactive original S1_N3 spatial-exclusion editor (freehand/box/polygon).

    This editor deliberately has DIFFERENT controls from tissue additions:
      F freehand (initial mode), B box, P polygon, H pan;
      left-drag to draw freehand/box; polygon left-click vertices;
      right-click / Enter closes the pending polygon (does not save);
      wheel or +/- zoom, middle-drag or H+left-drag pan, 0 fit;
      U undo, R reset all (can undo), S SAVE, Q/Esc CANCEL.

    The original fluorescence pixels are used for display only. This returns
    an annotation mask; files are written only by get_or_draw_annotation AFTER
    the user explicitly presses S. Cancelling preserves all saved files.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError('Install opencv-python with GUI support for exclusions') from exc
    from .visualization import robust_display

    if mode not in ('freehand', 'box', 'polygon', 'pan'):
        raise ValueError("mode must be one of 'freehand', 'box', 'polygon', 'pan'")
    raw = as_2d(image)
    display_8bit = np.rint(np.clip(robust_display(raw), 0, 1) * 255).astype(np.uint8)
    height, width = display_8bit.shape
    if starting_mask is None:
        drawn_mask = np.zeros(raw.shape, dtype=np.uint8)
    else:
        if as_2d(starting_mask).shape != raw.shape:
            raise ValueError('Starting exclusion mask and IF image must match')
        drawn_mask = (as_2d(starting_mask) > 0).astype(np.uint8) * 255
    import copy
    shapes = copy.deepcopy(starting_shapes or [])
    undo = []
    fit = min(1.0, 1400.0 / width, 900.0 / height)
    win_w = max(1, int(round(width * fit)))
    win_h = max(1, int(round(height * fit)))
    view = {'scale': fit, 'offset_x': 0.0, 'offset_y': 0.0,
            'panning': False, 'last_pan': None}
    state = {'mode': mode, 'drawing': False, 'points': [], 'start': None,
             'cursor': None}

    def clamp_view():
        visible_w = win_w / view['scale']
        visible_h = win_h / view['scale']
        view['offset_x'] = float(np.clip(view['offset_x'], 0,
                                           max(0.0, width - visible_w)))
        view['offset_y'] = float(np.clip(view['offset_y'], 0,
                                           max(0.0, height - visible_h)))

    def original_xy(x, y):
        ox = int(np.floor(view['offset_x'] + x / view['scale']))
        oy = int(np.floor(view['offset_y'] + y / view['scale']))
        return (int(np.clip(ox, 0, width - 1)), int(np.clip(oy, 0, height - 1)))

    def screen_poly(points):
        coords = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        offset = np.asarray([view['offset_x'], view['offset_y']])
        return np.rint((coords - offset) * view['scale']).astype(np.int32)

    def zoom_at(factor, x, y):
        if state['drawing']:
            return
        before = view['scale']
        after = float(np.clip(before * factor, fit, max(8.0, fit)))
        if before == after:
            return
        ox = view['offset_x'] + x / before
        oy = view['offset_y'] + y / before
        view['scale'] = after
        view['offset_x'] = ox - x / after
        view['offset_y'] = oy - y / after
        clamp_view()

    def snapshot():
        undo.append((drawn_mask.copy(), copy.deepcopy(shapes)))

    def commit_polygon(shape_mode='polygon'):
        if len(state['points']) >= 3:
            snapshot()
            vertices = np.asarray(state['points'], dtype=np.int32)
            cv2.fillPoly(drawn_mask, [vertices], 255)
            shapes.append({'mode': shape_mode,
                           'points_xy': [list(map(int, p)) for p in state['points']]})
        state['points'].clear()

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xFFFF
            if delta >= 32768:
                delta -= 65536
            if delta:
                zoom_at(1.25 if delta > 0 else (1 / 1.25), x, y)
            return
        if event == cv2.EVENT_MBUTTONDOWN and not state['drawing']:
            view['panning'] = True
            view['last_pan'] = (x, y)
            return
        if event == cv2.EVENT_MBUTTONUP:
            view['panning'] = False
            view['last_pan'] = None
            return
        if state['mode'] == 'pan':
            if event == cv2.EVENT_LBUTTONDOWN:
                view['panning'] = True
                view['last_pan'] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                view['panning'] = False
                view['last_pan'] = None
        if view['panning']:
            if event == cv2.EVENT_MOUSEMOVE and view['last_pan'] is not None:
                dx = x - view['last_pan'][0]
                dy = y - view['last_pan'][1]
                view['offset_x'] -= dx / view['scale']
                view['offset_y'] -= dy / view['scale']
                clamp_view()
                view['last_pan'] = (x, y)
            return
        if state['mode'] == 'pan':
            return

        state['cursor'] = original_xy(x, y)
        if state['mode'] == 'polygon':
            if event == cv2.EVENT_LBUTTONDOWN:
                state['points'].append(state['cursor'])
            elif event == cv2.EVENT_RBUTTONDOWN:
                commit_polygon()
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            state['drawing'] = True
            state['start'] = state['cursor']
            state['points'] = [state['cursor']]
        elif event == cv2.EVENT_MOUSEMOVE and state['drawing']:
            if state['mode'] == 'freehand':
                state['points'].append(state['cursor'])
        elif event == cv2.EVENT_LBUTTONUP and state['drawing']:
            state['drawing'] = False
            if state['mode'] == 'box':
                x0, y0 = state['start']
                x1, y1 = state['cursor']
                if abs(x1-x0) >= 1 and abs(y1-y0) >= 1:
                    snapshot()
                    cv2.rectangle(drawn_mask, (x0, y0), (x1, y1), 255, -1)
                    shapes.append({'mode':'box','points_xy':[[x0,y0],[x1,y1]]})
            elif state['mode'] == 'freehand':
                state['points'].append(state['cursor'])
                commit_polygon('freehand')
            state['points'].clear()
            state['start'] = None

    window = title
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, on_mouse)
    except cv2.error as exc:
        raise RuntimeError('OpenCV GUI could not start; use desktop opencv-python, not headless.') from exc

    print('MANUAL SPATIAL EXCLUSIONS — original S1_N3 controls')
    print('F=freehand | B=box | P=polygon | H=pan')
    print('Wheel or +/-=zoom | Middle-drag=pan | 0=fit')
    print('Polygon: Enter/right-click=close | U=undo | R=reset | S=save | Q=cancel')
    saved = False
    try:
        while True:
            scale = view['scale']
            matrix = np.asarray([[scale, 0, -view['offset_x']*scale],
                                 [0, scale, -view['offset_y']*scale]], dtype=np.float32)
            grayscale = cv2.warpAffine(display_8bit, matrix, (win_w, win_h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            shown_mask = cv2.warpAffine(drawn_mask, matrix, (win_w, win_h),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            preview = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
            highlighted = preview.copy()
            highlighted[shown_mask > 0] = (0, 0, 255)
            preview = cv2.addWeighted(preview, .65, highlighted, .35, 0)
            if state['points']:
                pts = screen_poly(state['points'])
                if len(pts) >= 2:
                    cv2.polylines(preview, [pts], False, (0, 255, 255), 2)
                if state['mode'] == 'polygon' and state['cursor'] is not None:
                    cv2.line(preview, tuple(pts[-1]),
                             tuple(screen_poly([state['cursor']])[0]), (0,255,255), 1)
                elif state['mode'] == 'box' and state['drawing'] and state['cursor']:
                    cv2.rectangle(preview, tuple(screen_poly([state['start']])[0]),
                                  tuple(screen_poly([state['cursor']])[0]), (0,255,255), 2)
            cv2.putText(preview, f"Mode: {state['mode'].upper()} | Zoom: {scale/fit:.1f}x | Red=excluded",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,240,255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('f'),ord('b'),ord('p'),ord('h')):
                state['mode'] = {ord('f'):'freehand',ord('b'):'box',
                                 ord('p'):'polygon',ord('h'):'pan'}[key]
                state['points'].clear()
                state['drawing'] = False
                state['start'] = None
            elif key in (ord('+'),ord('=')):
                zoom_at(1.25,win_w/2,win_h/2)
            elif key in (ord('-'),ord('_')):
                zoom_at(1/1.25,win_w/2,win_h/2)
            elif key == ord('0'):
                view.update({'scale':fit,'offset_x':0.0,'offset_y':0.0})
            elif key in (10,13) and state['mode'] == 'polygon':
                commit_polygon()  # ENTER closes polygon; it does NOT save.
            elif key == ord('u'):
                if state['points']:
                    state['points'].clear()
                elif undo:
                    old_mask, old_shapes = undo.pop()
                    drawn_mask[:] = old_mask
                    shapes[:] = old_shapes
            elif key == ord('r'):
                snapshot()
                drawn_mask.fill(0)
                shapes.clear()
                state['points'].clear()
            elif key == ord('s'):
                if state['drawing'] or state['points']:
                    print('Finish current selection before saving')
                    continue
                saved = True
                break
            elif key in (ord('q'),27):
                break
    finally:
        cv2.destroyWindow(window)
    if not saved:
        raise RuntimeError('Manual exclusion editing cancelled; no output files changed')
    result = drawn_mask > 0
    return (result, shapes) if return_shapes else result


def get_or_draw_spatial_exclusions(image, reference, *, save_dir, sample_name='sample',
                                   redraw=False, title=None):
    """First run draws original-style exclusions; later runs reuse this folder only."""
    return get_or_draw_annotation(
        image, reference, save_dir=save_dir, sample_name=sample_name,
        kind='exclusion', mode='freehand', redraw=redraw,
        title=title or f'{sample_name}: manual exclusions')


def apply_spatial_exclusions(tissue_analysis_mask, candidate_mask, exclusion_mask,
                             *, buffer_um=0.0, spacing_xy_um=None,
                             reference=None, save_dir=None, sample_name='sample'):
    """Exclude from BOTH analysis tissue and positive candidates (original behavior).

    Call on UNEXCLUDED input masks, e.g. `analysis_mask_before_exclusion`
    and `triangle_before_exclusion`, to avoid cumulative effects on reruns.
    The returned masks use the unmodified aligned IF grid and retain original
    foreground geometry; optional buffer is in physical micrometres.
    """
    from scipy import ndimage as ndi
    analysis = as_2d(tissue_analysis_mask).astype(bool)
    candidates = as_2d(candidate_mask).astype(bool)
    drawn = as_2d(exclusion_mask).astype(bool)
    if analysis.shape != candidates.shape or analysis.shape != drawn.shape:
        raise ValueError('Analysis tissue, candidate mask and exclusion must match the IF grid')
    if not np.isfinite(buffer_um) or buffer_um < 0:
        raise ValueError('buffer_um must be finite and nonnegative')
    if np.any(candidates & ~analysis):
        raise ValueError('Input candidates must lie inside the pre-exclusion analysis tissue')
    effective = drawn.copy()
    if buffer_um > 0 and drawn.any():
        if spacing_xy_um is None:
            raise ValueError('X/Y spacing required when exclusion buffer is nonzero')
        sx, sy = map(float, spacing_xy_um)
        if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
            raise ValueError('Positive finite X/Y spacing required')
        effective = ndi.distance_transform_edt(~drawn, sampling=(sy, sx)) <= buffer_um
    tissue_after = analysis & ~effective
    if not np.any(tissue_after):
        raise RuntimeError('Manual exclusion removed all analysis tissue')
    candidates_after = candidates & tissue_after
    s = Stage(save_dir, reference)
    s.mask('effective_exclusions', effective)
    s.mask('filtered_candidates', candidates_after)
    s.mask('tissue_analysis_after_exclusion', tissue_after)
    s.preview('effective_exclusions', effective)
    s.preview('filtered_candidates', candidates_after)
    s.preview('tissue_analysis_after_exclusion', tissue_after)
    s.mask(f'{sample_name}_exclusion_on_IF_grid', effective)
    s.mask(f'{sample_name}_triangle_after_exclusion', candidates_after)
    summary = {
        'sample_name': sample_name, 'buffer_um': buffer_um,
        'before_positive_pixels': int(candidates.sum()),
        'after_positive_pixels': int(candidates_after.sum()),
        'removed_positive_pixels': int((candidates & effective).sum()),
        'before_analysis_tissue_pixels': int(analysis.sum()),
        'after_analysis_tissue_pixels': int(tissue_after.sum()),
        'removed_analysis_tissue_pixels': int((analysis & effective).sum()),
    }
    s.json('summary', summary)
    return {'mask': candidates_after, 'candidate_mask': candidates_after,
            'tissue_analysis_mask': tissue_after, 'exclusion': effective,
            'drawn_exclusion': drawn, 'summary': summary}

def get_or_draw_annotation(image, reference, *, save_dir, sample_name='sample',
                           kind='exclusion', mode='polygon', redraw=False, title=None,
                           tissue_mask=None):
    """Draw a NEW annotation on the first run, then reuse it on subsequent runs.

    No legacy files or folders are searched. This workflow is intentionally
    self-contained in ``save_dir``; only the NRRD + JSON pair created here is
    eligible for automatic reuse.

    On first run, the OpenCV GUI is opened. For tissue_mask+kind='addition', press S to save (Enter closes the polygon).
    For exclusions, S also saves (Enter closes a polygon); F/B/P select tools.
    Esc cancels without creating/replacing saved files.
    On later runs, the saved mask is reused without a GUI. Set ``redraw=True`` to edit the existing
    mask; press R inside the editor to clear it and start from scratch.

    Reuse checks the sample, annotation kind, physical image grid AND a SHA256
    checksum of the source pixel array. This avoids silently reusing shapes
    drawn for a different IF image with identical geometry.

    Returns a dictionary with ``mask``, ``shapes``, ``status``, ``save_dir``.
    """
    if kind not in ('addition', 'exclusion'):
        raise ValueError("kind must be 'addition' or 'exclusion'")
    image_array = as_2d(image)
    if image_array.shape != as_2d(reference).shape:
        raise ValueError('Annotation source image and reference grid shape differ')
    fingerprint = hashlib.sha256(np.ascontiguousarray(image_array).tobytes()).hexdigest()
    directory = Path(save_dir)
    metadata_path = directory / 'annotation.json'
    nrrd_path = directory / 'annotation_mask.nrrd'
    if metadata_path.exists() != nrrd_path.exists():
        raise FileNotFoundError(
            'An annotation is partially saved (expected BOTH annotation.json and '
            f'annotation_mask.nrrd in {directory}). Fix that directory first.')

    previous = np.zeros(image_array.shape, dtype=bool)
    shapes = []
    saved = metadata_path.exists()
    if saved:
        # load_annotation checks sample, kind and full SimpleITK geometry.
        previous = load_annotation(directory, reference,
                                   sample_name=sample_name, kind=kind)
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('source_sha256') != fingerprint:
            raise ValueError(
                'Saved annotation is not confirmed for this exact IF pixel array. '
                f'Use a fresh stage folder to draw a new annotation: {directory}')
        shapes = metadata.get('shapes', [])
        if not redraw:
            return {'mask': previous, 'shapes': shapes,
                    'status': 'reused_saved_annotation', 'save_dir': directory}

    if kind == 'addition' and tissue_mask is not None:
        # Dedicated original S1_N3 interface, polygon-only with green/yellow overlays.
        if mode != 'polygon':
            raise ValueError('Tissue-addition editor only supports polygon mode')
        drawn, shapes = edit_tissue_additions(
            image_array, tissue_mask, starting_mask=previous,
            starting_shapes=shapes, title=title or f'{sample_name}: add tissue',
            return_shapes=True)
    elif kind == 'exclusion':
        # Original notebook exclusion editor, independent of tissue-addition GUI.
        drawn, shapes = edit_spatial_exclusions(
            image_array, starting_mask=previous, starting_shapes=shapes,
            mode=mode, title=title or f'{sample_name}: manual exclusions',
            return_shapes=True)
    else:
        # Keep legacy/general annotation API available for callers who use it.
        drawn, shapes = edit_annotation(
            image_array, starting_mask=previous, starting_shapes=shapes,
            mode=mode, title=title or f'{sample_name}: {kind}', return_shapes=True)
    drawn = as_2d(drawn).astype(bool)
    polygons_xy = [shape['points_xy'] for shape in shapes
                   if shape.get('mode') == 'polygon']
    save_annotation(drawn, reference, directory, sample_name=sample_name,
                    kind=kind, shapes=shapes, polygons_xy=polygons_xy,
                    source_sha256=fingerprint)
    if kind == 'exclusion':
        # Original-style *raw binary* PNG alongside canonical NRRD+JSON files.
        import cv2
        binary_png = directory / f'{sample_name}_drawn_exclusions.png'
        if not cv2.imwrite(str(binary_png), drawn.astype(np.uint8) * 255):
            raise IOError(f'Could not save raw exclusion mask: {binary_png}')
    return {'mask': drawn, 'shapes': shapes,
            'status': 'redrawn_annotation' if saved else 'new_annotation_drawn',
            'save_dir': directory}


def get_or_draw_tissue_additions(image, reference, automatic_tissue_mask, *, save_dir,
                                 sample_name='sample', redraw=False, title=None):
    """Polygon-only original S1_N3 tissue editor with safe first-draw/reuse.

    Parameters
    ----------
    image : 2D array or single-slice image
        Unchanged IF intensity image used only for display.
    reference : SimpleITK.Image
        Source grid used for geometry-preserving NRRD annotation export.
    automatic_tissue_mask : 2D bool array
        Initial tissue segmentation displayed in GREEN; never modified.
    save_dir : str or Path
        Own stage folder. First run draws; later runs reuse only this folder.
    redraw : bool
        Reopen the editor. Press R to clear all saved additions, S to save.

    Returns a dictionary with mask, shapes, status and save_dir.
    """
    return get_or_draw_annotation(
        image, reference,
        save_dir=save_dir, sample_name=sample_name, kind='addition',
        mode='polygon', redraw=redraw,
        title=title or f'{sample_name}: add tissue (polygon only)',
        tissue_mask=automatic_tissue_mask)
