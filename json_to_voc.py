# import os
# import json
# import shutil
# from pathlib import Path
# from argparse import ArgumentParser
# from xml.etree.ElementTree import Element, SubElement, ElementTree


# # ---------------- VOC writer ----------------
# def _indent(elem, level=0):
#     i = "\n" + level * "  "
#     if len(elem):
#         if not elem.text or not elem.text.strip():
#             elem.text = i + "  "
#         for e in elem:
#             _indent(e, level + 1)
#         if not elem.tail or not elem.tail.strip():
#             elem.tail = i
#     else:
#         if level and (not elem.tail or not elem.tail.strip()):
#             elem.tail = i


# def write_voc_xml(xml_path: Path, image_path: Path, width: int, height: int, objects: list, folder: str = "imgs"):
#     ann = Element("annotation")
#     SubElement(ann, "folder").text = folder
#     SubElement(ann, "filename").text = image_path.name
#     SubElement(ann, "path").text = str(image_path)

#     source = SubElement(ann, "source")
#     SubElement(source, "database").text = "Unknown"

#     size = SubElement(ann, "size")
#     SubElement(size, "width").text = str(int(width))
#     SubElement(size, "height").text = str(int(height))
#     SubElement(size, "depth").text = "3"
#     SubElement(ann, "segmented").text = "0"

#     for obj in objects:
#         o = SubElement(ann, "object")
#         SubElement(o, "name").text = str(obj["name"])
#         SubElement(o, "pose").text = "Unspecified"
#         SubElement(o, "truncated").text = "0"
#         SubElement(o, "difficult").text = "0"

#         bnd = SubElement(o, "bndbox")
#         SubElement(bnd, "xmin").text = str(int(obj["xmin"]))
#         SubElement(bnd, "ymin").text = str(int(obj["ymin"]))
#         SubElement(bnd, "xmax").text = str(int(obj["xmax"]))
#         SubElement(bnd, "ymax").text = str(int(obj["ymax"]))

#     _indent(ann)
#     xml_path.parent.mkdir(parents=True, exist_ok=True)
#     ElementTree(ann).write(str(xml_path), encoding="utf-8", xml_declaration=False)


# # ---------------- helpers ----------------
# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# def clamp(v, lo, hi):
#     return max(lo, min(hi, v))


# def parse_labelme_json(jpath: Path):
#     """
#     返回: (img_w, img_h, img_path_in_json, objects, labels_seen)
#     objects: [{'name': '1', 'xmin':..., 'ymin':..., 'xmax':..., 'ymax':...}, ...]
#     """
#     data = json.loads(jpath.read_text(encoding="utf-8"))
#     w = int(data.get("imageWidth", 0))
#     h = int(data.get("imageHeight", 0))
#     img_path_in_json = data.get("imagePath", "")

#     shapes = data.get("shapes", []) or []
#     objects = []
#     labels_seen = []

#     for s in shapes:
#         label = str(s.get("label", "")).strip()
#         labels_seen.append(label)

#         stype = s.get("shape_type", "")
#         pts = s.get("points", None)

#         # 你的样例是 rectangle，points 两个角点
#         if stype != "rectangle" or not pts or len(pts) < 2:
#             continue

#         (x1, y1), (x2, y2) = pts[0], pts[1]
#         xmin = min(float(x1), float(x2))
#         ymin = min(float(y1), float(y2))
#         xmax = max(float(x1), float(x2))
#         ymax = max(float(y1), float(y2))

#         objects.append({
#             "name": label,
#             "xmin": xmin,
#             "ymin": ymin,
#             "xmax": xmax,
#             "ymax": ymax,
#         })

#     return w, h, img_path_in_json, objects, labels_seen


# def safe_delete(path: Path, dry_run: bool, log_lines: list, reason: str):
#     if dry_run:
#         log_lines.append(f"[DRY-RUN][DELETE] {path}  reason={reason}")
#         return
#     try:
#         path.unlink()
#         log_lines.append(f"[DELETE] {path}  reason={reason}")
#     except Exception as e:
#         log_lines.append(f"[DELETE-FAIL] {path}  reason={reason}  err={e}")


# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--img-dir", default="/root/autodl-tmp/racketpose2.0/table_tennis_PP_2.0/bad_cases_imgs_epoch2/imgs", type=str, help="bad 图片目录（递归找图片）")
#     parser.add_argument("--json-dir", default="/root/autodl-tmp/racketpose2.0/table_tennis_PP_2.0/bad_cases_imgs_epoch2/json_boxes", type=str, help="标注 json 目录（递归找 json）")
#     parser.add_argument("--out-xml-dir", default="/root/autodl-tmp/racketpose2.0/table_tennis_PP_2.0/bad_cases_imgs_epoch2/boxes", type=str, help="输出 VOC xml 根目录（默认：img-dir 同级 boxes/）")

#     parser.add_argument("--folder-name", default="imgs", type=str, help="VOC <folder> 字段写什么")
#     parser.add_argument("--strict-label", action="store_true", help="只要发现 label!=1 直接报错退出（推荐开）")
#     parser.add_argument("--dry-run", action="store_true", help="不真正删除文件，只输出日志预览")
#     parser.add_argument("--log", default="json_to_voc_cleanup.log", type=str)

#     args = parser.parse_args()

#     img_root = Path(args.img_dir).resolve()
#     json_root = Path(args.json_dir).resolve()
#     if not img_root.exists():
#         raise FileNotFoundError(f"img-dir not found: {img_root}")
#     if not json_root.exists():
#         raise FileNotFoundError(f"json-dir not found: {json_root}")

#     # 输出目录默认：img-dir 的同级 boxes/
#     if args.out_xml_dir is None:
#         out_root = img_root.parent / "boxes"
#     else:
#         out_root = Path(args.out_xml_dir).resolve()
#     out_root.mkdir(parents=True, exist_ok=True)

#     # 建 json 索引：stem -> json_path（假设一个 stem 只有一个 json）
#     json_map = {}
#     for jp in json_root.rglob("*.json"):
#         json_map[jp.stem] = jp

#     log_lines = []
#     bad_label_lines = []
#     converted = 0
#     deleted = 0

#     # 遍历所有图片
#     img_files = [p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
#     img_files.sort()

#     for img_path in img_files:
#         stem = img_path.stem
#         jpath = json_map.get(stem, None)

#         # 1) 没有 json -> 删除图片
#         if jpath is None:
#             safe_delete(img_path, args.dry_run, log_lines, "missing_json")
#             deleted += 1
#             continue

#         # 解析 json
#         try:
#             w, h, _, objects, labels_seen = parse_labelme_json(jpath)
#         except Exception as e:
#             safe_delete(img_path, args.dry_run, log_lines, f"json_parse_error({jpath}): {e}")
#             deleted += 1
#             continue

#         # 2) label 必须全部为 "1"
#         bad_labels = [lb for lb in labels_seen if lb != "1"]
#         if bad_labels:
#             line = f"{img_path}\tjson={jpath}\tlabels={sorted(set(labels_seen))}"
#             bad_label_lines.append(line)
#             log_lines.append(f"[BAD-LABEL] {line}")

#             if args.strict_label:
#                 # 直接中止，避免输出污染
#                 (Path(args.log).resolve()).write_text("\n".join(log_lines + ["", "=== BAD LABELS ==="] + bad_label_lines),
#                                                      encoding="utf-8")
#                 raise RuntimeError(f"Found label != '1'. Example: {line}\n(已写入日志 {args.log})")
#             else:
#                 # 不严格模式：删掉图片（你也可以改成仅记录不删）
#                 safe_delete(img_path, args.dry_run, log_lines, "label_not_1")
#                 deleted += 1
#                 continue

#         # 3) 没有 bbox -> 删除图片
#         if len(objects) == 0:
#             safe_delete(img_path, args.dry_run, log_lines, f"no_bbox_in_json({jpath.name})")
#             deleted += 1
#             continue

#         # 用 json 的 w/h；若缺失则尝试从图像读（可选，这里不引入 cv2，保守）
#         if w <= 0 or h <= 0:
#             # 如果你希望强制正确尺寸，可以改成用 cv2.imread 来读
#             log_lines.append(f"[WARN] imageWidth/Height missing in {jpath}, use fallback 0 => skip xml")
#             safe_delete(img_path, args.dry_run, log_lines, "missing_image_size_in_json")
#             deleted += 1
#             continue

#         # clamp bbox 到图像范围，转 int
#         voc_objects = []
#         for obj in objects:
#             xmin = clamp(int(round(obj["xmin"])), 0, w - 1)
#             ymin = clamp(int(round(obj["ymin"])), 0, h - 1)
#             xmax = clamp(int(round(obj["xmax"])), 0, w - 1)
#             ymax = clamp(int(round(obj["ymax"])), 0, h - 1)
#             if xmax <= xmin or ymax <= ymin:
#                 continue
#             voc_objects.append({
#                 "name": "1",
#                 "xmin": xmin,
#                 "ymin": ymin,
#                 "xmax": xmax,
#                 "ymax": ymax,
#             })

#         if len(voc_objects) == 0:
#             safe_delete(img_path, args.dry_run, log_lines, "all_boxes_invalid_after_clamp")
#             deleted += 1
#             continue

#         # 写 xml：保持子目录结构
#         rel = img_path.relative_to(img_root)
#         xml_path = (out_root / rel).with_suffix(".xml")
#         write_voc_xml(xml_path, img_path, w, h, voc_objects, folder=args.folder_name)

#         log_lines.append(f"[OK] {img_path} -> {xml_path}")
#         converted += 1

#     # 写日志
#     log_path = Path(args.log).resolve()
#     tail = [
#         "",
#         "=== SUMMARY ===",
#         f"img_root={img_root}",
#         f"json_root={json_root}",
#         f"out_root={out_root}",
#         f"converted={converted}",
#         f"deleted={deleted}",
#         f"dry_run={args.dry_run}",
#         f"strict_label={args.strict_label}",
#         "",
#         "=== BAD LABELS (if any) ===",
#         *bad_label_lines
#     ]
#     log_path.write_text("\n".join(log_lines + tail), encoding="utf-8")

#     print(f"[Done] converted={converted}, deleted={deleted}, out_xml={out_root}")
#     print(f"[Log] {log_path}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
from pathlib import Path
from argparse import ArgumentParser
from xml.etree.ElementTree import Element, SubElement, ElementTree


# ---------------- VOC writer ----------------
def _indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            _indent(e, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write_voc_xml(xml_path: Path, image_path: Path, width: int, height: int, objects: list, folder: str = "imgs"):
    ann = Element("annotation")
    SubElement(ann, "folder").text = folder
    SubElement(ann, "filename").text = image_path.name
    SubElement(ann, "path").text = str(image_path)

    source = SubElement(ann, "source")
    SubElement(source, "database").text = "Unknown"

    size = SubElement(ann, "size")
    SubElement(size, "width").text = str(int(width))
    SubElement(size, "height").text = str(int(height))
    SubElement(size, "depth").text = "3"
    SubElement(ann, "segmented").text = "0"

    for obj in objects:
        o = SubElement(ann, "object")
        SubElement(o, "name").text = str(obj["name"])
        SubElement(o, "pose").text = "Unspecified"
        SubElement(o, "truncated").text = "0"
        SubElement(o, "difficult").text = "0"

        bnd = SubElement(o, "bndbox")
        SubElement(bnd, "xmin").text = str(int(obj["xmin"]))
        SubElement(bnd, "ymin").text = str(int(obj["ymin"]))
        SubElement(bnd, "xmax").text = str(int(obj["xmax"]))
        SubElement(bnd, "ymax").text = str(int(obj["ymax"]))

    _indent(ann)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree(ann).write(str(xml_path), encoding="utf-8", xml_declaration=False)


# ---------------- helpers ----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_labelme_json(jpath: Path):
    """
    返回: (img_w, img_h, img_path_in_json, objects, labels_seen)
    objects: [{'name': '1', 'xmin':..., 'ymin':..., 'xmax':..., 'ymax':...}, ...]
    """
    data = json.loads(jpath.read_text(encoding="utf-8"))
    w = int(data.get("imageWidth", 0))
    h = int(data.get("imageHeight", 0))
    img_path_in_json = data.get("imagePath", "")

    shapes = data.get("shapes", []) or []
    objects = []
    labels_seen = []

    for s in shapes:
        label = str(s.get("label", "")).strip()
        labels_seen.append(label)

        stype = s.get("shape_type", "")
        pts = s.get("points", None)

        # rectangle: two points
        if stype != "rectangle" or not pts or len(pts) < 2:
            continue

        (x1, y1), (x2, y2) = pts[0], pts[1]
        xmin = min(float(x1), float(x2))
        ymin = min(float(y1), float(y2))
        xmax = max(float(x1), float(x2))
        ymax = max(float(y1), float(y2))

        objects.append({"name": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    return w, h, img_path_in_json, objects, labels_seen


def safe_delete(path: Path, dry_run: bool, log_lines: list, reason: str):
    if not path.exists():
        return
    if dry_run:
        log_lines.append(f"[DRY-RUN][DELETE] {path}  reason={reason}")
        return
    try:
        path.unlink()
        log_lines.append(f"[DELETE] {path}  reason={reason}")
    except Exception as e:
        log_lines.append(f"[DELETE-FAIL] {path}  reason={reason}  err={e}")


def load_labels_csv(labels_csv: Path):
    if not labels_csv.exists():
        return None, []
    with open(labels_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        header = r.fieldnames
    return header, rows


def write_labels_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    parser = ArgumentParser()
    parser.add_argument("--img-dir", default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/SP3.0_bad_manual_upper/imgs", type=str, help="bad 图片目录（递归找图片）")
    parser.add_argument("--json-dir", default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/SP3.0_bad_manual_upper/json_boxes", type=str, help="标注 json 目录（递归找 json）")
    parser.add_argument("--out-xml-dir", default="/root/autodl-tmp/racketpose2.0/table_tennis_SP_3.0/SP3.0_bad_manual_upper/boxes", type=str, help="输出 VOC xml 根目录（生成/覆盖）")
    parser.add_argument("--allow-labels", default="1", type=str,
                    help="合法label白名单，用逗号分隔，例如 '1' 或 '2' 或 '1,2'")

    
    # 新增：原始 xml 根目录（用于删除旧 xml；如果你 out-xml-dir 就是同一个，也没问题）
    parser.add_argument("--orig-xml-dir", default=None, type=str,
                        help="原本推理生成的 xml 根目录（用于删除缺 json 的那些 xml）。不填则默认等于 out-xml-dir")

    # 新增：labels 汇总 csv（用于删除缺 json 的那部分行）
    parser.add_argument("--labels-csv", default=None, type=str,
                        help="labels 的 all.csv 路径（可选）。提供后会输出 labels_kept.csv / labels_deleted.csv")

    parser.add_argument("--folder-name", default="imgs", type=str, help="VOC <folder> 字段写什么")
    parser.add_argument("--strict-label", action="store_true", help="只要发现 label!=1 直接报错退出（推荐开）")
    parser.add_argument("--dry-run", action="store_true", help="不真正删除文件，只输出日志预览")
    parser.add_argument("--log", default="SP3.0_upper_json_to_voc_cleanup.log", type=str)

    # 新增：输出 to_delete txt
    parser.add_argument("--to-delete-txt", default="SP3.0_upper_to_delete_imgs.txt", type=str,
                        help="输出需要删除的图片清单")
    parser.add_argument("--save-relpath", action="store_true",
                        help="to_delete.txt 中写相对 img-dir 的路径（否则写绝对路径）")

    args = parser.parse_args()

    img_root = Path(args.img_dir).resolve()
    json_root = Path(args.json_dir).resolve()
    out_root = Path(args.out_xml_dir).resolve()
    orig_xml_root = Path(args.orig_xml_dir).resolve() if args.orig_xml_dir else out_root

    if not img_root.exists():
        raise FileNotFoundError(f"img-dir not found: {img_root}")
    if not json_root.exists():
        raise FileNotFoundError(f"json-dir not found: {json_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    # json 索引：stem -> json_path
    json_map = {jp.stem: jp for jp in json_root.rglob("*.json")}

    # labels csv（可选）
    labels_csv_path = Path(args.labels_csv).resolve() if args.labels_csv else None
    labels_header, labels_rows = (None, [])
    labels_map = {}  # filename(str) -> row
    if labels_csv_path:
        labels_header, labels_rows = load_labels_csv(labels_csv_path)
        if labels_header and "filename" in labels_header:
            for row in labels_rows:
                key = str(row.get("filename", "")).replace("\\", "/")
                if key:
                    labels_map[key] = row
        else:
            print(f"[WARN] labels csv missing or no 'filename' column: {labels_csv_path}")
            labels_csv_path = None

    log_lines = []
    bad_label_lines = []
    to_delete_imgs = []        # image paths to delete (for txt)
    deleted_label_rows = []    # rows removed from labels
    kept_label_rows = []       # rows kept

    converted = 0
    deleted = 0

    img_files = sorted([p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXTS])

    def mark_delete(img_path: Path, reason: str):
        nonlocal deleted
        deleted += 1

        # 记录删除清单（相对/绝对）
        if args.save_relpath:
            try:
                to_delete_imgs.append(str(img_path.relative_to(img_root)).replace("\\", "/") + f"\t{reason}")
            except Exception:
                to_delete_imgs.append(str(img_path) + f"\t{reason}")
        else:
            to_delete_imgs.append(str(img_path) + f"\t{reason}")

        # 删除 image
        safe_delete(img_path, args.dry_run, log_lines, reason)

        # 删除对应的“原始 xml”（推理阶段生成的那个）
        try:
            rel = img_path.relative_to(img_root)
            xml_path = (orig_xml_root / rel).with_suffix(".xml")
            safe_delete(xml_path, args.dry_run, log_lines, f"{reason}:remove_xml")
        except Exception as e:
            log_lines.append(f"[WARN] cannot resolve xml for {img_path}: {e}")

        # labels 行标记删除（如果提供了 labels csv）
        if labels_csv_path:
            # 你的 labels 里 filename 通常是 "imgs/xxx.jpg" 或 "imgs\\xxx.jpg"
            # 这里同时尝试两种 key
            key1 = str(Path("imgs") / rel).replace("\\", "/")
            key2 = str(rel).replace("\\", "/")
            row = labels_map.get(key1) or labels_map.get(key2)
            if row:
                deleted_label_rows.append(row)

    # 遍历图片
    for img_path in img_files:
        stem = img_path.stem
        jpath = json_map.get(stem, None)

        # 1) 没有 json -> 删除 image + xml + label row
        if jpath is None:
            mark_delete(img_path, "missing_json")
            continue

        # 解析 json
        try:
            w, h, _, objects, labels_seen = parse_labelme_json(jpath)
        except Exception as e:
            mark_delete(img_path, f"json_parse_error({jpath.name})")
            continue

        # # 2) label 必须全部为 "1"
        # bad_labels = [lb for lb in labels_seen if lb != "1"]
        # if bad_labels:
        #     line = f"{img_path}\tjson={jpath}\tlabels={sorted(set(labels_seen))}"
        #     bad_label_lines.append(line)
        #     log_lines.append(f"[BAD-LABEL] {line}")

        #     if args.strict_label:
        #         Path(args.log).resolve().write_text(
        #             "\n".join(log_lines + ["", "=== BAD LABELS ==="] + bad_label_lines),
        #             encoding="utf-8"
        #         )
        #         raise RuntimeError(f"Found label != '1'. Example: {line}\n(已写入日志 {args.log})")
        #     else:
        #         mark_delete(img_path, "label_not_1")
        #         continue
        # 2) label 白名单：只允许 allow_labels 里的值
        allow = set([x.strip() for x in args.allow_labels.split(",") if x.strip() != ""])
        bad_labels = [lb for lb in labels_seen if lb not in allow]
        if bad_labels:
            line = f"{img_path}\tjson={jpath}\tlabels={sorted(set(labels_seen))}"
            bad_label_lines.append(line)
            log_lines.append(f"[BAD-LABEL] {line}")

            if args.strict_label:
                Path(args.log).resolve().write_text(
                    "\n".join(log_lines + ["", "=== BAD LABELS ==="] + bad_label_lines),
                    encoding="utf-8"
                )
                raise RuntimeError(f"Found label not in allow list={sorted(allow)}. Example: {line}")
            else:
                safe_delete(img_path, args.dry_run, log_lines, "label_not_allowed")
                deleted += 1
                continue


        # 3) 没有 bbox -> 删除
        if len(objects) == 0:
            mark_delete(img_path, f"no_bbox_in_json({jpath.name})")
            continue

        # 4) json 没尺寸 -> 删除
        if w <= 0 or h <= 0:
            mark_delete(img_path, f"missing_image_size_in_json({jpath.name})")
            continue

        # clamp bbox
        voc_objects = []
        for obj in objects:
            xmin = clamp(int(round(obj["xmin"])), 0, w - 1)
            ymin = clamp(int(round(obj["ymin"])), 0, h - 1)
            xmax = clamp(int(round(obj["xmax"])), 0, w - 1)
            ymax = clamp(int(round(obj["ymax"])), 0, h - 1)
            if xmax <= xmin or ymax <= ymin:
                continue
            voc_objects.append({"name": "1", "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

        if len(voc_objects) == 0:
            mark_delete(img_path, "all_boxes_invalid_after_clamp")
            continue

        # 写新 xml：保持结构
        rel = img_path.relative_to(img_root)
        xml_path = (out_root / rel).with_suffix(".xml")
        write_voc_xml(xml_path, img_path, w, h, voc_objects, folder=args.folder_name)
        log_lines.append(f"[OK] {img_path} -> {xml_path}")
        converted += 1

        # labels row 保留（可选）
        if labels_csv_path:
            key1 = str(Path("imgs") / rel).replace("\\", "/")
            key2 = str(rel).replace("\\", "/")
            row = labels_map.get(key1) or labels_map.get(key2)
            if row:
                kept_label_rows.append(row)

    # 写 to_delete txt
    del_txt = Path(args.to_delete_txt).resolve()
    del_txt.parent.mkdir(parents=True, exist_ok=True)
    del_txt.write_text("\n".join(to_delete_imgs), encoding="utf-8")

    # 输出 labels_kept / deleted（可选）
    if labels_csv_path and labels_header:
        kept_csv = labels_csv_path.parent / "labels_kept.csv"
        del_csv = labels_csv_path.parent / "labels_deleted.csv"
        write_labels_csv(kept_csv, labels_header, kept_label_rows)
        write_labels_csv(del_csv, labels_header, deleted_label_rows)
        log_lines.append(f"[LABELS] kept={len(kept_label_rows)} -> {kept_csv}")
        log_lines.append(f"[LABELS] deleted={len(deleted_label_rows)} -> {del_csv}")

    # 写日志
    log_path = Path(args.log).resolve()
    tail = [
        "",
        "=== SUMMARY ===",
        f"img_root={img_root}",
        f"json_root={json_root}",
        f"out_root={out_root}",
        f"orig_xml_root={orig_xml_root}",
        f"converted={converted}",
        f"deleted={deleted}",
        f"dry_run={args.dry_run}",
        f"strict_label={args.strict_label}",
        f"to_delete_txt={del_txt}",
        "",
        "=== BAD LABELS (if any) ===",
        *bad_label_lines
    ]
    log_path.write_text("\n".join(log_lines + tail), encoding="utf-8")

    print(f"[Done] converted={converted}, deleted={deleted}, out_xml={out_root}")
    print(f"[Out] to_delete_txt: {del_txt} (n={len(to_delete_imgs)})")
    if labels_csv_path:
        print(f"[Out] labels_kept/deleted saved under: {labels_csv_path.parent}")
    print(f"[Log] {log_path}")


if __name__ == "__main__":
    main()
