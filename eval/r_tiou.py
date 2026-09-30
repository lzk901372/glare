"""R-tIoU: mean temporal IoU over all matched events in the evaluation set."""

from common import emit_result, iter_annotation_pairs, make_parser, match_events, run_cli


def compute_r_tiou(pairs) -> float:
    total_iou, match_count = 0.0, 0
    for _, gt, gen in pairs:
        for match in match_events(gt.events, gen.events):
            total_iou += match.tiou
            match_count += 1
    # Appendix D.3 explicitly assigns zero if there are no matches.
    return total_iou / match_count if match_count else 0.0


def main():
    args = make_parser(__doc__).parse_args()
    emit_result("R-tIoU", compute_r_tiou(iter_annotation_pairs(args.gt_csv_dir, args.gen_csv_dir)))


if __name__ == "__main__":
    run_cli(main)
