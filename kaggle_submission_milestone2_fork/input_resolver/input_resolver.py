# >>> [calamitychasm] INPUT RESOLVER -- Kaggle mounts inputs in more than one layout.
# A GPU session of the same kernel gets datasets at /kaggle/input/datasets/<owner>/<slug> or at
# /kaggle/input/<slug>, the competition at /kaggle/input/competitions/<comp> or /kaggle/input/<comp>, and
# models at /kaggle/input/models/<owner>/<model>/... or /kaggle/input/<model>/... (JustAdev742 saw 5 of 9
# sessions on the older layout, 2026-10-07). /kaggle/input is read-only, so nothing is linked: the notebook
# resolves each input once and its path constants point at what this session actually has.
#   INPUT_RESOLVED <name> -> <path> (via <how>)     one line per input, greppable in the log
#   INPUT_MISSING <name>                            printed (with everything searched) before raising
import os as _ri_os

INPUT_ROOT = '/kaggle/input'          # tests point this at a temp dir
_RI_CANONICAL = '/kaggle/input'       # the prefix every path literal in the notebook starts with
_RI_MAX_DEPTH = 4                     # glob fallback: directories this far below INPUT_ROOT
RESOLVED_INPUTS = {}


def _ri_ci_walk(base, parts):
    """base/parts[0]/parts[1]/... matching each component case-insensitively; None if any is missing."""
    cur = base
    for part in parts:
        nxt = _ri_os.path.join(cur, part)
        if _ri_os.path.exists(nxt):
            cur = nxt
            continue
        try:
            names = _ri_os.listdir(cur)
        except OSError:
            return None
        hit = sorted(n for n in names if n.lower() == part.lower())
        if not hit:
            return None
        cur = _ri_os.path.join(cur, hit[0])
    return cur


def _ri_layouts(parts):
    """(label, relative parts) the same input may be mounted at, then (ident, rest) for the glob search."""
    alts, ident, rest = [('canonical', parts)], parts[-1], []
    if parts[0] == 'datasets' and len(parts) >= 3:      # datasets/<owner>/<slug>[/sub]
        alts.append(('older layout', parts[2:]))
        ident, rest = parts[2], parts[3:]
    elif parts[0] == 'competitions' and len(parts) >= 2:  # competitions/<comp>[/sub]
        alts.append(('older layout', parts[1:]))
        ident, rest = parts[1], parts[2:]
    elif parts[0] == 'models' and len(parts) >= 3:      # models/<owner>/<model>/<framework>/<instance>/<version>
        alts.append(('older layout', parts[2:]))
        ident, rest = parts[2], parts[3:]
    else:
        ident, rest = parts[0], parts[1:]
    return alts, ident, rest


def _ri_glob(ident, rest):
    """Directories named like ident up to _RI_MAX_DEPTH below INPUT_ROOT that contain the rest of the path."""
    found = []
    for cur, dirs, _files in _ri_os.walk(INPUT_ROOT, followlinks=True):
        depth = 0 if cur == INPUT_ROOT else len(_ri_os.path.relpath(cur, INPUT_ROOT).split(_ri_os.sep))
        if depth >= _RI_MAX_DEPTH:
            dirs[:] = []
        dirs.sort()
        for d in dirs:
            if d.lower() == ident.lower():
                hit = _ri_ci_walk(_ri_os.path.join(cur, d), rest)
                if hit is not None and _ri_os.path.isdir(hit):
                    found.append((depth, hit))
    found.sort()
    return found[0][1] if found else None


def _ri_listing():
    lines = []
    for cur, dirs, files in _ri_os.walk(INPUT_ROOT, followlinks=True):
        depth = 0 if cur == INPUT_ROOT else len(_ri_os.path.relpath(cur, INPUT_ROOT).split(_ri_os.sep))
        if depth >= 3:
            dirs[:] = []
        dirs.sort()
        lines.append(f'  {cur}  dirs={dirs[:8]} files={len(files)}')
        if len(lines) >= 60:
            break
    return lines or [f'  (no {INPUT_ROOT} mount is present)']


def resolve_input(name, path):
    """Return the existing directory for the canonical /kaggle/input path `path`, whichever layout this
    session mounted; print INPUT_RESOLVED, or print INPUT_MISSING with what was searched and raise."""
    if path != _RI_CANONICAL and not path.startswith(_RI_CANONICAL + '/'):
        raise ValueError(f'resolve_input({name!r}): {path!r} is not under {_RI_CANONICAL}')
    parts = [p for p in path[len(_RI_CANONICAL):].split('/') if p]
    alts, ident, rest = _ri_layouts(parts)
    searched = []
    for how, rel in alts:
        cand = _ri_ci_walk(INPUT_ROOT, rel)
        searched.append(_ri_os.path.join(INPUT_ROOT, *rel))
        if cand is not None and _ri_os.path.isdir(cand):
            RESOLVED_INPUTS[name] = cand
            print(f'INPUT_RESOLVED {name} -> {cand} (via {how})', flush=True)
            return cand
    cand = _ri_glob(ident, rest)
    searched.append(f'{INPUT_ROOT}/**/{ident}/' + '/'.join(rest) + f' (depth <= {_RI_MAX_DEPTH})')
    if cand is not None:
        RESOLVED_INPUTS[name] = cand
        print(f'INPUT_RESOLVED {name} -> {cand} (via glob)', flush=True)
        return cand
    print(f'INPUT_MISSING {name}: expected {path}', flush=True)
    for s in searched:
        print(f'  searched {s}', flush=True)
    print(f'  directories under {INPUT_ROOT}:', flush=True)
    for line in _ri_listing():
        print(line, flush=True)
    raise FileNotFoundError(f'INPUT_MISSING {name}: {path} (searched {searched})')
# <<< [calamitychasm] INPUT RESOLVER
