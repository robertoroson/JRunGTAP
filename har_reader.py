"""
har_reader.py  –  GEMPACK HAR binary reader (Python 3, little-endian)
Reads all numeric and string arrays from GTAP-format HAR files.

Usage:
    from har_reader import read_har
    d = read_har("gsdfdat.har")   # returns dict: name -> numpy array
    p = read_har("gsdfpar.har")

Arrays are shaped in COLUMN-MAJOR (Fortran) order, matching GEMPACK storage.
Supported types: RE (float32), 2R (float64), 2I (int32), 1C (string/12-char).
"""
import struct
import numpy as np


def read_har(fname):
    """Parse a GEMPACK HAR file and return {header_name: numpy_array}.

    Also populates the module-level `sets` dict with any named-set element
    labels encountered while reading, keyed by the 8-char set name."""

    def rrec(f):
        """Read one little-endian Fortran unformatted record."""
        hdr = f.read(4)
        if len(hdr) < 4:
            return None, 0
        n = struct.unpack('<i', hdr)[0]
        if n <= 0 or n > 200_000_000:
            return None, n
        data = f.read(n)
        f.read(4)          # trailing length word
        return data, n

    arrays = {}
    sets   = {}   # set_name -> [element, ...]

    with open(fname, 'rb') as f:
        while True:

            # ── Record 1: 4-byte header name ────────────────────────────
            rec1, n1 = rrec(f)
            if rec1 is None:
                break
            if n1 != 4:
                continue          # skip XXCR / LICN / DVER etc.
            try:
                name = rec1.decode('ascii').strip()
                if not all(c.isalnum() or c == '_' for c in name):
                    continue
            except Exception:
                continue

            # ── Record 2: descriptor ─────────────────────────────────────
            # Layout: [4-byte space pad][type 2B (RE/2R/2I/1C)][long name ...]
            rec2, n2 = rrec(f)
            if rec2 is None or n2 < 6:
                break
            try:
                rtype = rec2[4:6].decode('ascii').strip()
            except Exception:
                continue
            if rtype not in ('RE', '2R', '2I', '1C'):
                continue

            # ── Record 3: array element info ────────────────────────────
            # Layout: [4 pad][n_named_sets int][?][ndim int][name 12B][1][setname 8B]...
            # n_named_sets = number of distinct named sets (may be < ndim for bilateral arrays)
            # ndim          = actual array dimensions
            rec3, n3 = rrec(f)
            if rec3 is None or n3 < 16:
                break

            # ── 1C arrays: all data is in rec3 itself ─────────────────────
            # Layout: [4 pad][1][n_el][n_el][el0 w bytes][el1 w bytes]...
            # where w = (n3 - 16) / n_el
            if rtype == '1C':
                try:
                    n_el = struct.unpack_from('<i', rec3, 8)[0]
                    if 0 < n_el <= 200_000 and n3 > 16:
                        width = (n3 - 16) // n_el
                        if width > 0:
                            arr = np.array([
                                rec3[16 + i*width : 16 + (i+1)*width]
                                .decode('ascii', errors='replace').strip()
                                for i in range(n_el)
                            ])
                            arrays[name] = arr
                except Exception:
                    pass
                continue   # no further records for 1C headers

            # ── Numeric arrays: parse n_named / ndim from rec3 ───────────
            n_named = struct.unpack_from('<i', rec3, 4)[0]
            ndim    = struct.unpack_from('<i', rec3, 12)[0]
            if not (0 <= n_named <= 6) or not (0 <= ndim <= 6):
                continue

            # ── Records 4 … 3+n_named: one per named set (element names) ─
            for _ in range(n_named):
                r, rlen = rrec(f)
                if r is None:
                    break

            # ── Shape record ──────────────────────────────────────────────
            shape_rec, _ = rrec(f)
            if shape_rec is None:
                break
            if ndim == 0:
                shape = (1,)
            else:
                try:
                    shape = tuple(
                        struct.unpack_from('<i', shape_rec, 12 + i * 4)[0]
                        for i in range(ndim)
                    )
                    if any(s <= 0 or s > 200_000 for s in shape):
                        continue
                except Exception:
                    continue

            total = 1
            for s in shape:
                total *= s

            # ── Page-structure record (skip) ──────────────────────────────
            page_rec, _ = rrec(f)
            if page_rec is None:
                break

            # ── Data records (paginated for large arrays) ─────────────────
            # Between data chunks, GEMPACK inserts 64-byte page-header records.
            # Data record layout: [4-byte pad][4-byte int][payload]
            chunks = []
            collected = 0
            while collected < total:
                drec, dlen = rrec(f)
                if drec is None:
                    break
                # 64-byte records between data chunks are page headers — skip
                if dlen == 64:
                    continue
                payload = drec[8:]   # skip 4-byte pad + 4-byte header int
                if rtype == 'RE':
                    n_el = len(payload) // 4
                    vals = np.frombuffer(payload[: n_el * 4],
                                         dtype='<f4').astype(np.float64)
                elif rtype == '2R':
                    n_el = len(payload) // 8
                    vals = np.frombuffer(payload[: n_el * 8], dtype='<f8')
                elif rtype == '2I':
                    n_el = len(payload) // 4
                    vals = np.frombuffer(payload[: n_el * 4], dtype='<i4')
                else:
                    break
                if len(vals) == 0:
                    break
                chunks.append(vals)
                collected += len(vals)

            if not chunks:
                continue
            try:
                flat = np.concatenate(chunks)[:total]
                # GEMPACK stores in column-major (Fortran) order
                arr  = flat.reshape(shape, order='F')
                arrays[name] = arr
            except Exception:
                continue

    return arrays, sets
