"""CodeContests riegeli 最小解析器（严格按官方格式规范；只读，不执行任何提交代码）。

规范：https://raw.githubusercontent.com/google/riegeli/master/doc/riegeli_records_file_format.md
用法：
  python scripts/cc_riegeli_extract.py --probe <file.riegeli>        # 只走 chunk 结构，输出类型统计
  python scripts/cc_riegeli_extract.py --extract <file.riegeli> --out <out.jsonl>
字段表：contest_problem.proto（ContestProblem：name=1, description=2, tests=4/5/18,
source=6, difficulty=7, solutions=8, cf_contest_id=10, cf_index=12, cf_rating=14,
cf_tags=15, incorrect_solutions=19；Test{input=1,output=2}；Solution{language=1,solution=2}）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

KBS = 1 << 16
KBH = 24
KUH = KBS - KBH
KCH = 40

SOURCE = {0: "UNKNOWN", 1: "CODECHEF", 2: "CODEFORCES", 3: "HACKEREARTH", 4: "CODEJAM",
          6: "ATCODER", 7: "AIZU"}
DIFF = {0: "UNKNOWN", 1: "EASY", 2: "MEDIUM", 3: "HARD", 4: "HARDER", 5: "HARDEST", 6: "EXTERNAL"}
LANG = {0: "UNKNOWN", 1: "PYTHON2", 2: "CPP", 3: "PYTHON3", 4: "JAVA"}


def u64(b: bytes) -> int:
    return int.from_bytes(b, "little")


def varint(buf: bytes, off: int):
    x = 0
    shift = 0
    while True:
        b = buf[off]
        off += 1
        x |= (b & 0x7F) << shift
        if not (b & 0x80):
            return x, off
        shift += 7


def remaining_in_block(pos: int) -> int:
    return KBS - 1 - (pos + KBS - 1) % KBS


def sat_sub(a: int, b: int) -> int:
    return a - b if a > b else 0


def add_with_overhead(pos: int, size: int) -> int:
    n_ov = (size + (pos + KUH - 1) % KBS) // KUH
    return pos + size + n_ov * KBH


def round_up_boundary(pos: int) -> int:
    return pos + sat_sub(remaining_in_block(pos), KUH - 1)


class RiegeliFile:
    def __init__(self, path: Path):
        self.data = path.read_bytes()
        self.n = len(self.data)

    def gather(self, pos: int, size: int):
        out = []
        while size > 0:
            if pos % KBS == 0:
                pos += KBH  # every 64KiB multiple has a block header
            take = min(size, KBS - (pos % KBS))
            out.append(self.data[pos:pos + take])
            pos += take
            size -= take
        return b"".join(out), pos

    def chunks(self):
        pos = 0
        types = Counter()
        records_total = 0
        while pos < self.n:
            begin = pos
            hdr, pos2 = self.gather(begin, KCH)
            if len(hdr) < KCH:
                break
            data_size = u64(hdr[8:16])
            ctype = hdr[24]
            num_records = int.from_bytes(hdr[25:32], "little")
            data, pos3 = self.gather(pos2, data_size)
            cend = max(add_with_overhead(begin, KCH + data_size),
                       round_up_boundary(begin + num_records))
            types[chr(ctype)] += 1
            records_total += num_records
            yield {"begin": begin, "type": chr(ctype), "data_size": data_size,
                   "num_records": num_records, "data": data, "end": cend}
            pos = cend

    def decode_records(self, chunk):
        """仅支持简单块 'r'；返回 record 字节列表。"""
        if chunk["type"] != "r":
            return None
        data = chunk["data"]
        ctype = data[0]
        sizes_size, off = varint(data, 1)
        sizes_blob = data[off:off + sizes_size]
        values_blob = data[off + sizes_size:]

        def decomp(blob):
            if ctype == 0:
                return blob
            dlen, o = varint(blob, 0)
            comp = blob[o:]
            if ctype == 0x7A:
                import zstandard as zstd
                return zstd.ZstdDecompressor().decompress(comp, max_output_size=dlen)
            if ctype == 0x62:
                import brotli
                return brotli.decompress(comp)
            if ctype == 0x73:
                import snappy
                return snappy.decompress(comp)
            raise ValueError(f"unsupported compression {ctype:#x}")

        sizes = decomp(sizes_blob)
        values = decomp(values_blob)
        recs = []
        o = 0
        for _ in range(chunk["num_records"]):
            sz, o = varint(sizes, o)
            recs.append(values[:sz])
            values = values[sz:]
        return recs


# ---------------- protobuf wire 解析（仅所需字段） ----------------
def pb_fields(buf: bytes):
    """yield (field_no, wire_type, value_or_bytes)"""
    o = 0
    n = len(buf)
    while o < n:
        key, o = varint(buf, o)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, o = varint(buf, o)
            yield fno, wt, v
        elif wt == 2:
            ln, o = varint(buf, o)
            yield fno, wt, buf[o:o + ln]
            o += ln
        elif wt == 5:
            yield fno, wt, buf[o:o + 4]
            o += 4
        elif wt == 1:
            yield fno, wt, buf[o:o + 8]
            o += 8
        else:
            raise ValueError(f"wire type {wt}")


def text(b) -> str:
    return b.decode("utf-8", errors="replace")


def parse_problem(buf: bytes) -> dict:
    out = {"name": "", "description": "", "source": 0, "difficulty": 0,
           "cf_contest_id": None, "cf_index": "", "cf_rating": None, "cf_tags": [],
           "public_tests": 0, "private_tests": 0, "generated_tests": 0,
           "tests_nonempty": 0, "solutions": Counter(), "incorrect": Counter(),
           "solutions_n": 0, "incorrect_n": 0}
    for fno, wt, v in pb_fields(buf):
        if fno == 1 and wt == 2:
            out["name"] = text(v)
        elif fno == 2 and wt == 2:
            out["description"] = text(v)
        elif fno == 6 and wt == 0:
            out["source"] = v
        elif fno == 7 and wt == 0:
            out["difficulty"] = v
        elif fno == 10 and wt == 0:
            out["cf_contest_id"] = v
        elif fno == 12 and wt == 2:
            out["cf_index"] = text(v)
        elif fno == 14 and wt == 0:
            out["cf_rating"] = v
        elif fno == 15 and wt == 2:
            out["cf_tags"].append(text(v))
        elif fno in (4, 5, 18) and wt == 2:
            key = {4: "public_tests", 5: "private_tests", 18: "generated_tests"}[fno]
            out[key] += 1
            has_in = has_out = False
            for f2, w2, v2 in pb_fields(v):
                if f2 == 1 and w2 == 2 and v2:
                    has_in = True
                if f2 == 2 and w2 == 2 and v2:
                    has_out = True
            out["tests_nonempty"] += int(has_in and has_out)
        elif fno in (8, 19) and wt == 2:
            lang = 0
            sol = b""
            for f2, w2, v2 in pb_fields(v):
                if f2 == 1 and w2 == 0:
                    lang = v2
                elif f2 == 2 and w2 == 2:
                    sol = v2
            key = "solutions" if fno == 8 else "incorrect"
            if sol:
                out[key][lang] += 1
                if fno == 8:
                    out["solutions_n"] += 1
                else:
                    out["incorrect_n"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path)
    ap.add_argument("--extract", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    t0 = time.time()
    path = args.probe or args.extract
    rf = RiegeliFile(path)
    types = Counter()
    n_chunks = n_records = 0
    first_recs = []
    out_fh = (args.out or Path("/tmp/cc_extract.jsonl")).open("w", encoding="utf-8") \
        if args.extract else open("/dev/null", "w")
    try:
        for ch in rf.chunks():
            n_chunks += 1
            types[ch["type"]] += ch["num_records"] if ch["type"] in "rt" else 0
            types["chunk:" + ch["type"]] += 1
            n_records += ch["num_records"]
            if args.extract and ch["type"] == "r" and ch["num_records"]:
                recs = rf.decode_records(ch)
                for rec in recs:
                    p = parse_problem(rec)
                    desc = p.pop("description")
                    row = {
                        "name": p["name"], "description_sha256": hashlib.sha256(desc.encode()).hexdigest(),
                        "description_len": len(desc), "source": SOURCE.get(p["source"], str(p["source"])),
                        "difficulty": DIFF.get(p["difficulty"], str(p["difficulty"])),
                        "cf_contest_id": p["cf_contest_id"], "cf_index": p["cf_index"],
                        "cf_rating": p["cf_rating"], "cf_tags_n": len(p["cf_tags"]),
                        "public_tests": p["public_tests"], "private_tests": p["private_tests"],
                        "generated_tests": p["generated_tests"], "tests_nonempty": p["tests_nonempty"],
                        "has_tests": bool(p["public_tests"] + p["private_tests"] + p["generated_tests"]),
                        "solutions_n": p["solutions_n"], "incorrect_n": p["incorrect_n"],
                        "solutions_langs": {LANG.get(k, str(k)): v for k, v in p["solutions"].items()},
                        "incorrect_langs": {LANG.get(k, str(k)): v for k, v in p["incorrect"].items()},
                    }
                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif args.probe and ch["type"] == "r" and ch["num_records"] and len(first_recs) < 3:
                recs = rf.decode_records(ch)
                first_recs.extend(recs[:3 - len(first_recs)])
    finally:
        out_fh.close()
    report = {"file": str(path), "bytes": rf.n, "chunks": n_chunks, "records": n_records,
              "chunk_types": dict(types), "runtime_s": round(time.time() - t0, 1)}
    print(json.dumps(report, ensure_ascii=False))
    if args.probe:
        for i, rec in enumerate(first_recs):
            try:
                p = parse_problem(rec)
                print(f"rec{i}: name={p['name'][:40]!r} desc_len={len(p['description'])} "
                      f"source={SOURCE.get(p['source'])} solutions={p['solutions_n']} "
                      f"incorrect={p['incorrect_n']} tests={p['public_tests']}/{p['private_tests']}/{p['generated_tests']}")
            except Exception as e:
                print("rec parse error:", type(e).__name__, str(e)[:120])


if __name__ == "__main__":
    main()
