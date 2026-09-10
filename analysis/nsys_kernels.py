#!/usr/bin/env python3
"""Summarise CUDA kernel and memcpy activity from Nsight Systems SQLite exports.

Nsight writes a .nsys-rep; `nsys export --type sqlite report.nsys-rep` turns it into
a queryable database. This reads those databases and produces one row per kernel
launch with achieved bandwidth, so a set of runs can be compared directly instead of
opening each trace in the GUI.

    ./nsys_kernels.py ../transpose/traces/*.sqlite --bytes-moved 8000000 --peak-gbs 2039

A trace with runtime API calls but zero kernels means the launches failed — most
often a block larger than the 1024-thread limit. That case is reported, not skipped.
"""
import argparse, os, sqlite3, sys

KERNELS = """
SELECT s.value AS name,
       k.gridX, k.gridY, k.gridZ,
       k.blockX, k.blockY, k.blockZ,
       (k.end - k.start) / 1000.0 AS us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
ORDER BY k.start
"""
MEMCPY = """
SELECT copyKind, bytes, (end - start) / 1000.0 AS us
FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start
"""
COPY_KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD"}


def scalar(cur, sql, default=None):
    try:
        row = cur.execute(sql).fetchone()
        return row if row else default
    except sqlite3.Error:
        return default


def read(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    gpu = scalar(cur, "SELECT name, computeMajor, computeMinor, smCount FROM TARGET_INFO_GPU")
    try:
        kernels = cur.execute(KERNELS).fetchall()
    except sqlite3.Error:
        kernels = []
    try:
        memcpy = cur.execute(MEMCPY).fetchall()
    except sqlite3.Error:
        memcpy = []
    api = scalar(cur, "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME", (0,))[0]
    con.close()
    return gpu, kernels, memcpy, api


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--bytes-moved", type=float, default=None,
                    help="bytes a kernel reads+writes, for achieved-bandwidth columns")
    ap.add_argument("--peak-gbs", type=float, default=None, help="device peak HBM bandwidth, GB/s")
    ap.add_argument("--csv", help="also write rows here")
    a = ap.parse_args()

    rows, gpu_seen = [], None
    for path in a.traces:
        gpu, kernels, memcpy, api = read(path)
        gpu_seen = gpu_seen or gpu
        tag = os.path.basename(path).replace(".sqlite", "")

        if not kernels:
            print(f"\n## {tag}")
            print(f"   no kernels recorded, {api} runtime API calls -> launches failed")
            if api:
                print("   most likely cause: threads-per-block above the 1024 limit")
            continue

        print(f"\n## {tag}")
        hdr = f"   {'kernel':<26}{'grid':>12}{'block':>10}{'time(us)':>10}"
        if a.bytes_moved:
            hdr += f"{'GB/s':>9}"
            if a.peak_gbs:
                hdr += f"{'% peak':>8}"
        print(hdr)
        for name, gx, gy, gz, bx, by, bz, us in kernels:
            short = name.split("(")[0]
            grid, block = f"{gx}x{gy}", f"{bx}x{by}"
            line = f"   {short:<26}{grid:>12}{block:>10}{us:>10.2f}"
            bw = None
            if a.bytes_moved:
                bw = a.bytes_moved / (us * 1e-6) / 1e9
                line += f"{bw:>9.0f}"
                if a.peak_gbs:
                    line += f"{bw / a.peak_gbs * 100:>7.1f}%"
            print(line)
            rows.append([tag, short, grid, block, f"{us:.2f}", f"{bw:.1f}" if bw else ""])

        for kind, nbytes, us in memcpy:
            gbs = nbytes / (us * 1e-6) / 1e9
            print(f"   {'memcpy ' + COPY_KIND.get(kind, str(kind)):<26}"
                  f"{nbytes/1e6:>11.1f}MB{'':>10}{us:>10.2f}{gbs:>9.1f}")

    if gpu_seen:
        name, major, minor, sms = gpu_seen
        print(f"\nDevice: {name}, compute {major}.{minor}, {sms} SMs")

    if a.csv and rows:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["trace", "kernel", "grid", "block", "time_us", "achieved_gbs"])
            w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    sys.exit(main())
