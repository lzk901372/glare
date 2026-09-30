"""R-ATD: mean asymmetric timing penalty over all matches, reported times 100."""

import sys

from common import emit_result, iter_annotation_pairs, make_parser, match_events, run_cli


ALPHA = 2.0
BETA = 0.5
# USER-CONFIRMED reporting convention: multiply the formula result by 100.
# Eq. (7)/(17)/(18) themselves contain NO factor of 100; the paper does not
# explicitly specify a scale for its tables. This factor is our agreed output
# convention, not a claim that a hidden implementation detail was recovered.
REPORT_SCALE = 100.0


def pair_penalty(ground_truth, prediction) -> float:
    duration = ground_truth.end - ground_truth.start
    deviations = (
        (prediction.start - ground_truth.start) / duration,
        (prediction.end - ground_truth.end) / duration,
        ((prediction.end - prediction.start) - duration) / duration,
    )
    # The THREE penalties are summed per pair, not averaged by three.
    return sum((ALPHA if delta < 0 else BETA) * abs(delta) for delta in deviations)


def compute_r_atd(pairs) -> float | None:
    total_penalty, match_count = 0.0, 0
    for _, gt, gen in pairs:
        for match in match_events(gt.events, gen.events):
            total_penalty += pair_penalty(match.ground_truth, match.prediction)
            match_count += 1
    return REPORT_SCALE * total_penalty / match_count if match_count else None


def main():
    args = make_parser(__doc__).parse_args()
    result = compute_r_atd(iter_annotation_pairs(args.gt_csv_dir, args.gen_csv_dir))
    if result is None:
        print("R-ATD is null: no matched events (Appendix D.3).", file=sys.stderr)
    emit_result("R-ATD", result)


if __name__ == "__main__":
    run_cli(main)
