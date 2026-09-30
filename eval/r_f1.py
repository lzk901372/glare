"""R-F1: dataset-level event counts, followed by an equal mean of six classes."""

from common import REACTIONS, emit_result, iter_annotation_pairs, make_parser, match_events, run_cli


def compute_r_f1(pairs) -> float:
    # Confirmed interpretation of the ambiguous wording in Appendix D.3:
    # aggregate TP/FP/FN over ALL clips per class, then take the six-class mean.
    # Do not average per-video F1, pool classes into micro-F1, or divide by Nk.
    tp = dict.fromkeys(REACTIONS, 0)
    fp = dict.fromkeys(REACTIONS, 0)
    fn = dict.fromkeys(REACTIONS, 0)
    for _, gt, gen in pairs:
        for event in gt.events:
            fn[event.reaction] += 1
        for event in gen.events:
            fp[event.reaction] += 1
        for match in match_events(gt.events, gen.events):
            reaction = match.ground_truth.reaction
            tp[reaction] += 1
            fp[reaction] -= 1
            fn[reaction] -= 1
    class_scores = []
    for reaction in REACTIONS:
        denominator = 2 * tp[reaction] + fp[reaction] + fn[reaction]
        class_scores.append(2 * tp[reaction] / denominator if denominator else 0.0)
    # All six classes participate, including absent classes (F1 = 0).
    return sum(class_scores) / len(REACTIONS)


def main():
    args = make_parser(__doc__).parse_args()
    emit_result("R-F1", compute_r_f1(iter_annotation_pairs(args.gt_csv_dir, args.gen_csv_dir)))


if __name__ == "__main__":
    run_cli(main)
