"""UA-DETRAC parsing, per-vehicle annotated occlusion and conversion (data/detrac.py, scripts/02c)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from confdet.data import detrac as dt  # noqa: E402

# frame 181 of MVI_39031 (vehicles 4, 5, 6 as shown in the Phase 3a report), plus two invented frames
XML = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_TEST">
 <sequence_attribute camera_state="stable" sence_weather="sunny"/>
 <ignored_region><box left="0" top="0" width="50" height="40"/></ignored_region>
 <frame density="3" num="1"><target_list>
  <target id="4"><box left="677.33" top="310.4" width="108.74" height="107.67"/>
   <attribute orientation="0" speed="0" trajectory_length="1" truncation_ratio="0" vehicle_type="van"/></target>
  <target id="5"><box left="433.07" top="112" width="21.26" height="22.33"/>
   <attribute orientation="0" speed="0" trajectory_length="1" truncation_ratio="0" vehicle_type="car"/>
   <occlusion><region_overlap left="433.07" top="112" width="18.06" height="9.53" occlusion_id="6" occlusion_status="1"/></occlusion></target>
  <target id="6"><box left="433.07" top="101.33" width="18.06" height="20.2"/>
   <attribute orientation="0" speed="0" trajectory_length="1" truncation_ratio="0.25" vehicle_type="car"/></target>
 </target_list></frame>
 <frame density="1" num="2"><target_list>
  <target id="7"><box left="100" top="100" width="100" height="100"/>
   <attribute orientation="0" speed="0" trajectory_length="1" truncation_ratio="0" vehicle_type="bus"/>
   <occlusion><region_overlap left="100" top="150" width="100" height="50" occlusion_id="8" occlusion_status="0"/>
              <region_overlap left="100" top="100" width="50" height="50" occlusion_id="0" occlusion_status="-1"/></occlusion></target>
  <target id="8"><box left="100" top="150" width="100" height="100"/>
   <attribute orientation="0" speed="0" trajectory_length="1" truncation_ratio="0" vehicle_type="others"/></target>
 </target_list></frame>
 <frame density="1" num="11"><target_list>
  <target id="9"><box left="10" top="10" width="20" height="20"/>
   <attribute orientation="0" speed="0" trajectory_length="1" truncation_ratio="0" vehicle_type="car"/></target>
 </target_list></frame>
</sequence>"""


def test_occlusion_readings_on_the_reported_example(tmp_path):
    p = tmp_path / "MVI_TEST.xml"
    p.write_text(XML)
    seq = dt.parse_sequence(p)
    assert seq["name"] == "MVI_TEST" and seq["ignored"] == [(0.0, 0.0, 50.0, 40.0)]
    occ = {o["track_id"]: o for o in dt.frame_occlusion(seq["frames"][1])}
    region = 18.06 * 9.53
    assert occ["6"]["occ_vehicle"] == pytest.approx(region / (18.06 * 20.2), abs=1e-6)   # 0.47: hidden by 5
    assert occ["5"]["occ_vehicle"] == 0.0 and occ["4"]["occ_vehicle"] == 0.0             # 5 is the occluder
    assert occ["5"]["occ_naive"] == pytest.approx(region / (21.26 * 22.33), abs=1e-6)    # naive: 0.36 on the carrier
    assert occ["6"]["occ_naive"] == 0.0 and occ["6"]["truncation"] == 0.25
    f2 = {o["track_id"]: o for o in dt.frame_occlusion(seq["frames"][2])}
    assert f2["7"]["occ_vehicle"] == pytest.approx(0.5) and f2["7"]["occ_nonvehicle"] == pytest.approx(0.25)
    assert f2["7"]["occ_naive"] == pytest.approx(0.5)                                    # code 0 counts in both readings


def test_every_tenth_frame_and_label_lines(tmp_path):
    p = tmp_path / "MVI_TEST.xml"
    p.write_text(XML)
    seq = dt.parse_sequence(p)
    assert dt.sampled_frames(seq["frames"], 10) == [1, 11]
    vs = seq["frames"][2]
    lines, kept = dt.yolo_label_lines(vs, seq["ignored"], 960, 540)
    assert kept == [0, 1] and len(lines) == 3
    assert lines[0].startswith("2 ") and lines[1].startswith("3 ") and lines[2].startswith("4 ")   # bus, others, ignore
    off = dt.Vehicle("x", (2000, 10, 2100, 50), "car", 0.0)                                        # outside the image
    l2, k2 = dt.yolo_label_lines([off] + vs, [], 960, 540)
    assert k2 == [1, 2] and len(l2) == 2


def test_converter_links_images_and_aligns_annotations(tmp_path):
    from PIL import Image
    raw = tmp_path / "raw"
    for split, seqname in (("Train", "MVI_A"), ("Test", "MVI_B")):
        d = raw / "DETRAC-Annos" / f"DETRAC-{split}-Annotations-XML"
        d.mkdir(parents=True)
        (d / f"{seqname}.xml").write_text(XML.replace("MVI_TEST", seqname))
        imgs = raw / "DETRAC-Images" / seqname
        imgs.mkdir(parents=True)
        for n in (1, 2, 11):
            Image.new("RGB", (960, 540)).save(imgs / f"img{n:05d}.jpg")
    spec = importlib.util.spec_from_file_location("c02c", REPO / "scripts" / "02c_convert_detrac.py")
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log = type("L", (), {"info": staticmethod(lambda *_: None)})()
    out = tmp_path / "proc" / "ua_detrac"
    s = mod.convert(raw, out, 10, log)
    assert s["calib"]["images"] == 2 and s["test"]["images"] == 2                       # frames 1 and 11 of each
    import os
    a = out / "images" / "calib" / "MVI_A_img00001.jpg"
    assert os.stat(a).st_ino == os.stat(raw / "DETRAC-Images" / "MVI_A" / "img00001.jpg").st_ino   # hard link
    ann = json.loads((out / "annotations_calib.json").read_text())
    lbl = (out / "labels" / "calib" / "MVI_A_img00001.txt").read_text().splitlines()
    vehicle_lines = [l for l in lbl if not l.startswith("4 ")]
    assert len(ann["MVI_A_img00001.jpg"]) == len(vehicle_lines) == 3
    m = json.loads((out / "manifest.json").read_text())
    assert {im["group"] for im in m["images"] if im["role"] == "test"} == {"seq:MVI_B"}
    with pytest.raises(SystemExit):
        mod.convert(raw, out, 10, log)                                                   # never overwrites


def test_annotations_align_with_tables_and_group_bootstrap():
    from confdet.evaluation.labeling import build_tables
    from confdet.evaluation.records import ImageRecord
    # labels: car, others, bus, ignore -> objects kept after the class map: car, bus
    r = ImageRecord("f.jpg", 960, 540, np.array([[0, 0, 50, 50], [100, 0, 150, 50], [200, 0, 250, 50], [0, 400, 100, 500]], float),
                    np.array([0, 3, 2, 4]), np.zeros(4), np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
    t = build_tables([r], class_agnostic=True, cand_class_map=np.zeros(1, int),
                     gt_class_map=np.array([0, 1, 2, -1, -1]))
    ann = {"f.jpg": [{"occ_vehicle": 0.1}, {"occ_vehicle": 0.2}, {"occ_vehicle": 0.3}]}
    assert dt.annotations_for_tables(ann, [r], t, "occ_vehicle").tolist() == [0.1, 0.3]
    with pytest.raises(ValueError):
        dt.annotations_for_tables({"f.jpg": [{"occ_vehicle": 0.1}]}, [r], t, "occ_vehicle")
    w = dt.group_bootstrap_image_weights(["a", "a", "b", "c", "c", "c"], 200, seed=1)
    assert w.shape == (200, 6) and (w[:, 0] == w[:, 1]).all() and (w[:, 3] == w[:, 5]).all()
    assert np.allclose(w[:, [0, 2, 3]].sum(axis=1), 3)                    # three sequences drawn per replicate


def test_script_20_end_to_end_on_a_fake_detrac(tmp_path):
    """Convert a fake UA-DETRAC, fake the cached predictions of one detector, run scripts/20."""
    import subprocess
    import yaml
    from PIL import Image
    from confdet.evaluation.labeling import build_tables
    from confdet.evaluation.records import ImageRecord, RecordSet, save_records
    from confdet.evaluation.simulate import simulate_records
    from confdet.pipeline import fit_posthoc
    from confdet.utils.io_utils import save_results
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    for split, seqs in (("Train", ["MVI_A", "MVI_B", "MVI_C"]), ("Test", ["MVI_D", "MVI_E", "MVI_F"])):
        d = raw / "DETRAC-Annos" / f"DETRAC-{split}-Annotations-XML"
        d.mkdir(parents=True)
        for s in seqs:
            frames = []
            for n in range(1, 201, 10):
                tg = []
                for k in range(6):
                    x, y = rng.uniform(0, 800), rng.uniform(0, 400)
                    occl = (f'<occlusion><region_overlap left="{x:.1f}" top="{y + 40:.1f}" width="60" height="20" '
                            f'occlusion_id="{k + 1}" occlusion_status="0"/></occlusion>') if k % 2 else ""
                    tg.append(f'<target id="{k}"><box left="{x:.1f}" top="{y:.1f}" width="80" height="60"/>'
                              f'<attribute orientation="0" speed="0" trajectory_length="1" truncation_ratio="0" '
                              f'vehicle_type="{["car", "van", "bus"][k % 3]}"/>{occl}</target>')
                frames.append(f'<frame density="6" num="{n}"><target_list>{"".join(tg)}</target_list></frame>')
            (d / f"{s}.xml").write_text(f'<sequence name="{s}"><ignored_region/>{"".join(frames)}</sequence>')
            (raw / "DETRAC-Images" / s).mkdir(parents=True)
            for n in range(1, 201, 10):
                Image.new("RGB", (960, 540)).save(raw / "DETRAC-Images" / s / f"img{n:05d}.jpg")
    cfg = yaml.safe_load((REPO / "configs" / "experiment_detrac.yaml").read_text("utf-8"))
    cfg["project"].update({"processed_dir": str(tmp_path / "proc"), "output_dir": str(tmp_path / "out"),
                           "runs_dir": str(tmp_path / "runs")})
    cfg["datasets"]["ua_detrac"]["root"] = str(raw)
    cfg["evaluation"]["bootstrap_replicates"] = 40
    cfg["posthoc"]["min_group_size"] = 30
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run = lambda s, *a: subprocess.run([sys.executable, str(REPO / "scripts" / s), "--config", str(cpath), *a],  # noqa: E731
                                       capture_output=True, text=True, cwd=REPO)
    r = run("02c_convert_detrac.py", "--every", "10")
    assert r.returncode == 0, r.stderr[-2000:]
    bmd_names = list(cfg["datasets"]["bmd45"]["class_map"])
    out = tmp_path / "out" / "bmd_dev" / "det"
    sim = simulate_records(400, seed=3)
    save_records(out / "predictions" / "bmd45__calib.npz",
                 RecordSet(sim, {"dataset": "bmd45", "role": "calib", "dataset_names": bmd_names, "detector_names": bmd_names}))
    t = build_tables(sim, class_agnostic=True, context_conf=0.3)
    save_results(fit_posthoc(cfg, t, 42, tau=0.3, mode="class_agnostic"), out / "class_agnostic" / "calibration.json")
    for role in ("calib", "test"):
        recs = []
        for img in sorted((tmp_path / "proc" / "bmd_dev" / "ua_detrac" / "images" / role).iterdir()):
            rows = [list(map(float, l.split())) for l in (img.parents[2] / "labels" / role / (img.stem + ".txt")).read_text().splitlines()]
            g = np.array([[(cx - w / 2) * 960, (cy - h / 2) * 540, (cx + w / 2) * 960, (cy + h / 2) * 540] for _, cx, cy, w, h in rows]).reshape(-1, 4)
            gc = np.array([int(c) for c, *_ in rows], int)
            keep = rng.random(len(g)) < 0.85
            cand = np.vstack([g[keep] + rng.normal(0, 3, (keep.sum(), 4)), rng.uniform(0, 500, (3, 4)) + [0, 0, 60, 60]])
            conf = np.r_[rng.uniform(0.3, 0.99, keep.sum()), rng.uniform(0.001, 0.4, 3)]
            recs.append(ImageRecord(img.name, 960, 540, g, gc, occlusion_severity(g), cand, conf, np.zeros(len(conf), int)))
        save_records(out / "predictions" / f"ua_detrac__{role}.npz",
                     RecordSet(recs, {"dataset": "ua_detrac", "role": role, "dataset_names": list(dt.CLASSES),
                                      "detector_names": bmd_names}))
    r = run("20_detrac.py", "--detectors", "det")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads((out / "class_agnostic" / "results" / "ua_detrac_transfer.json").read_text())
    blk = res["by_alpha"]["0.1"]
    assert {"frozen:mondrian_gilg_iou_aware", "recalibrated:mondrian_confidence", "frozen:global_confidence"} <= set(blk)
    assert 0.6 < res["recall_at_frozen_threshold"] <= 1.0 and res["n_sequences"] == 3
    pv = json.loads((tmp_path / "out" / "bmd_dev" / "comparison" / "class_agnostic" / "ua_detrac_proxy_validation.json").read_text())
    assert pv["n_vehicles"] == res["n_vehicles"] and "o_hat_per_detector" in pv


def occlusion_severity(g):
    from confdet.data.occlusion import occlusion_severity_array
    return occlusion_severity_array(g)


def test_interim_report_copies_numbers_and_marks_missing_sections(tmp_path):
    import subprocess
    from confdet.utils.io_utils import site_paths
    queue = site_paths()["queue"]
    if not Path(queue).exists():
        return                                   # no job queue on this site (e.g. Beocat): nothing to report on
    out = tmp_path / "r.md"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "21_interim_report.py"), "--queue",
                        queue, "--out", str(out)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    text = out.read_text(encoding="utf-8")
    assert "## a) NMS pilot" in text and "## Queue status" in text
    for section in ("b) Evidential", "c) UA-DETRAC", "evidential: rarity diagnostic (calib)"):
        assert section in text
    # sections whose result file does not exist yet say so instead of inventing numbers
    if not (REPO / "outputs" / "bmd_dev" / "yolo26_evid_pilot_b" / "class_agnostic" / "results" / "evidential_pilot.json").exists():
        assert "evidential_pilot.json not found" in text
