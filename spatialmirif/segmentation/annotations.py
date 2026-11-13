"""General polygon/freehand/rectangle annotation with reusable geometry metadata.

Annotation masks are always stored separately from raw IF or automatic masks.
An optional interactive OpenCV editor runs locally (not in headless batch mode).
"""
from __future__ import annotations
import json
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
                    kind='exclusion', polygons_xy=None, shapes=None):
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
                          'shapes': shapes or []})
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


def load_legacy_annotation(png_path, metadata_path, reference, *, sample_name):
    """Reuse old S1_N3 PNG+metadata format from the original notebook."""
    import cv2
    png_path, metadata_path = Path(png_path), Path(metadata_path)
    if not png_path.exists() or not metadata_path.exists():
        raise FileNotFoundError('Existing PNG and metadata JSON are both required')
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    expected = {'sample_name': sample_name,
                'shape_yx': list(map(int, as_2d(reference).shape)),
                'spacing': grid_info(reference)['spacing'],
                'origin': grid_info(reference)['origin'],
                'direction': grid_info(reference)['direction']}
    if meta != expected:
        raise ValueError('Legacy annotation image/sample/geometry does not match')
    im = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if im is None or im.shape != as_2d(reference).shape:
        raise ValueError('Invalid legacy annotation PNG')
    return im > 0


def load_legacy_exclusion(png_path, metadata_path, reference, *, sample_name):
    """Strictly import the original notebook's saved exclusion PNG and provenance."""
    import cv2
    import hashlib
    png_path, metadata_path = Path(png_path), Path(metadata_path)
    if not png_path.exists() or not metadata_path.exists():
        raise FileNotFoundError('Both old exclusion PNG and metadata JSON are required')
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    g = grid_info(reference)
    raw = np.ascontiguousarray(as_2d(reference))
    expected = {'sample_name': str(sample_name),
                'image_shape_yx': list(map(int, raw.shape)),
                'aligned_IF_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                'image_spacing_xy': g['spacing'],
                'image_origin_xy': g['origin'],
                'image_direction': g['direction']}
    if meta != expected:
        raise ValueError('Legacy exclusion provenance differs from the selected IF image')
    image = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != raw.shape:
        raise ValueError('Saved exclusion mask shape mismatch')
    return image > 0


def apply_additions(automatic_mask, additions, *, reference=None, save_dir=None):
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


def edit_annotation(image, *, starting_mask=None, mode='polygon', title='IF annotation',
                    return_shapes=False):
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
    shapes = []
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
