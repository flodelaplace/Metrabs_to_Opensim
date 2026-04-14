"""
Calibrate the bml_movi_87 marker positions by aggregating MarkerPlacer outputs
from multiple subjects.

Workflow:
  1. Run the pipeline with --ik on multiple test videos
  2. Each run produces a *_markers.xml in its output folder
  3. This script reads all of them, computes per-marker statistics,
     and optionally generates a new calibrated marker XML

Usage:
  python tools/calibrate_markers.py output/                  # report only
  python tools/calibrate_markers.py output/ --apply          # write calibrated XML
  python tools/calibrate_markers.py output/ --threshold 30   # offset > 30mm only
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from lxml import etree


REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_XML = REPO_ROOT / 'opensim_setup' / 'Markers_bml_movi_87.xml'
OUTPUT_XML = REPO_ROOT / 'opensim_setup' / 'Markers_bml_movi_87_calibrated.xml'


def parse_markers_xml(path):
    """Return {marker_name: (parent_body, np.array([x,y,z]), fixed_flag)}."""
    tree = etree.parse(str(path))
    markers = {}
    for m in tree.getroot().iter('Marker'):
        name = m.get('name')
        if name is None:
            continue
        parent = m.find('socket_parent_frame')
        loc = m.find('location')
        fixed = m.find('fixed')
        if parent is None or loc is None:
            continue
        coords = np.array([float(x) for x in loc.text.split()])
        is_fixed = (fixed is not None and fixed.text.strip().lower() == 'true')
        markers[name] = (parent.text.strip(), coords, is_fixed)
    return markers


def find_run_marker_files(output_dir):
    """Find all MarkerPlacer output marker XML files in subfolders."""
    output_dir = Path(output_dir)
    files = []
    for f in output_dir.rglob('*_markers.xml'):
        if 'opensim_setup' in f.parts:
            continue
        files.append(f)
    return sorted(files)


def aggregate_marker_positions(marker_files):
    """Load all marker files and return {marker_name: list of (file, position)}."""
    data = defaultdict(list)
    for f in marker_files:
        try:
            markers = parse_markers_xml(f)
        except Exception as e:
            print(f"  /!\\ Failed to parse {f}: {e}")
            continue
        for name, (parent, coords, fixed) in markers.items():
            data[name].append((f.parent.name, parent, coords, fixed))
    return data


def write_calibrated_xml(calibrated, template_path, output_path):
    """Write a new marker XML with calibrated positions."""
    tree = etree.parse(str(template_path))
    for m in tree.getroot().iter('Marker'):
        name = m.get('name')
        if name not in calibrated:
            continue
        new_coords = calibrated[name][1]
        loc = m.find('location')
        if loc is not None:
            loc.text = f"{new_coords[0]} {new_coords[1]} {new_coords[2]}"
    tree.write(str(output_path), pretty_print=True, xml_declaration=True, encoding='UTF-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_dir',
                        help='Directory containing pipeline runs (e.g. output/)')
    parser.add_argument('--apply', action='store_true',
                        help='Write calibrated XML to opensim_setup/')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Only update markers with offset > threshold mm (default: 10)')
    parser.add_argument('--min_subjects', type=int, default=3,
                        help='Minimum number of subjects to update a marker (default: 3)')
    args = parser.parse_args()

    if not ORIGINAL_XML.exists():
        print(f"Error: original XML not found at {ORIGINAL_XML}")
        sys.exit(1)
    original = parse_markers_xml(ORIGINAL_XML)
    print(f"Loaded {len(original)} markers from {ORIGINAL_XML.name}\n")

    marker_files = find_run_marker_files(args.output_dir)
    if not marker_files:
        print(f"No *_markers.xml files found in {args.output_dir}")
        print("Hint: run the pipeline with --ik on multiple videos first.")
        sys.exit(1)

    print(f"Found {len(marker_files)} subject marker files:")
    for f in marker_files:
        print(f"  {f.relative_to(args.output_dir)}")
    print()

    data = aggregate_marker_positions(marker_files)

    print("=" * 95)
    print(f"{'Marker':<22} {'Body':<22} {'N':<4} {'Original (mm)':<22} "
          f"{'Calibrated mean (mm)':<22} {'|d| mm':<10} {'s mm':<10} {'Status'}")
    print("=" * 95)

    calibrated = {}
    summary = {"updated": [], "skipped_small": [], "skipped_few_subjects": [],
               "fixed_in_xml": []}

    for name, original_data in original.items():
        orig_parent, orig_coords, orig_fixed = original_data
        runs = data.get(name, [])
        n = len(runs)

        if n == 0:
            continue

        coords_arr = np.array([r[2] for r in runs])
        mean_pos = coords_arr.mean(axis=0)
        std_pos = coords_arr.std(axis=0)
        offset = mean_pos - orig_coords
        offset_mm = offset * 1000
        offset_mag_mm = np.linalg.norm(offset_mm)
        std_mm = np.linalg.norm(std_pos * 1000)

        if orig_fixed:
            status = "FIXED (no change)"
            summary["fixed_in_xml"].append(name)
            calibrated[name] = (orig_parent, orig_coords)
        elif n < args.min_subjects:
            status = f"too few subjects ({n} < {args.min_subjects})"
            summary["skipped_few_subjects"].append(name)
            calibrated[name] = (orig_parent, orig_coords)
        elif offset_mag_mm < args.threshold:
            status = "ok (< threshold)"
            summary["skipped_small"].append(name)
            calibrated[name] = (orig_parent, orig_coords)
        else:
            status = "*** UPDATED ***"
            summary["updated"].append((name, offset_mag_mm, n))
            calibrated[name] = (orig_parent, mean_pos)

        orig_str = f"({orig_coords[0]*1000:6.0f},{orig_coords[1]*1000:6.0f},{orig_coords[2]*1000:6.0f})"
        new_str = f"({mean_pos[0]*1000:6.0f},{mean_pos[1]*1000:6.0f},{mean_pos[2]*1000:6.0f})"
        body_short = orig_parent.replace('/bodyset/', '')
        print(f"{name:<22} {body_short:<22} {n:<4} {orig_str:<22} {new_str:<22} "
              f"{offset_mag_mm:<10.1f} {std_mm:<10.1f} {status}")

    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(f"  Total markers analyzed: {len(original)}")
    print(f"  Updated (offset > {args.threshold}mm, >={args.min_subjects} subjects): "
          f"{len(summary['updated'])}")
    print(f"  Skipped (offset < {args.threshold}mm): {len(summary['skipped_small'])}")
    print(f"  Skipped (too few subjects): {len(summary['skipped_few_subjects'])}")
    print(f"  Skipped (fixed=true): {len(summary['fixed_in_xml'])}")

    if summary["updated"]:
        print(f"\nTop offsets:")
        for name, mag, n in sorted(summary["updated"], key=lambda x: -x[1])[:15]:
            print(f"  {name:<22} offset={mag:>6.1f}mm  ({n} subjects)")

    if args.apply:
        write_calibrated_xml(calibrated, ORIGINAL_XML, OUTPUT_XML)
        print(f"\nCalibrated XML written to: {OUTPUT_XML}")
    else:
        print(f"\n(Pass --apply to write calibrated XML)")


if __name__ == '__main__':
    main()
