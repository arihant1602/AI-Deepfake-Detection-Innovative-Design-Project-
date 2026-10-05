# Benchmark results

Generated 2026-10-05 21:21 by `benchmarks/run_benchmark.py`.

## Gate 2 - enrolled-reference PRNU (decision: PCE >= 60)

### usb-5986-216a-1-3 1280x720 (2 genuine sessions, leave-one-session-out enrollment)

| Stream | n | PCE min | PCE median | PCE max | accepted |
|---|---|---|---|---|---|
| genuine (other sessions) | 2 | 3418.21 | 3785.36 | 4152.51 | 100.0% |
| impostor - fake: face-swap (DF40) (center crop) | 1 | 1.63 | 1.63 | 1.63 | 0/1 |
| impostor - fake: talking-head (DF40) (center crop) | 5 | 0.24 | 1.42 | 3.28 | 0/5 |
| impostor - fake: text-to-video (center crop) | 10 | 1.16 | 2.5 | 4.89 | 0/10 |
| impostor - real: phone native (VISION) (center crop) | 20 | 0.38 | 3.13 | 16.9 | 0/20 |
| replay of genuine capture via x264-crf23 | 1 | 175.82 | 175.82 | 175.82 | 100% |
| replay of genuine capture via x264-crf32 | 1 | 16.33 | 16.33 | 16.33 | 0% |
| replay of genuine capture via mp4v | 1 | 385.43 | 385.43 | 385.43 | 100% |
| photo re-animation of a genuine frame | 1 | 1.0 | 1.0 | 1.0 | 0% |
| static replay of a genuine frame | 1 | 552.15 | 552.15 | 552.15 | 100% |

Impostor accept rate: 0.00% of 36 clips; latency per test 852.72 ms (median).

### usb-5986-216a-1-3 640x480 (8 genuine sessions, leave-one-session-out enrollment)

| Stream | n | PCE min | PCE median | PCE max | accepted |
|---|---|---|---|---|---|
| genuine (other sessions) | 56 | 392.26 | 2982.9 | 21227.37 | 100.0% |
| impostor - fake: face-swap (DF40) (center crop) | 8 | 0.83 | 1.91 | 10.0 | 0/8 |
| impostor - fake: talking-head (DF40) (center crop) | 6 | 0.37 | 1.2 | 4.04 | 0/6 |
| impostor - fake: text-to-video (center crop) | 16 | 1.09 | 2.91 | 10.08 | 0/16 |
| impostor - real: phone WhatsApp (VISION) (center crop) | 12 | 1.51 | 4.0 | 7.08 | 0/12 |
| impostor - real: phone native (VISION) (center crop) | 22 | 0.31 | 3.25 | 13.49 | 0/22 |
| replay of genuine capture via x264-crf23 | 7 | 920.47 | 1494.85 | 7201.94 | 100% |
| replay of genuine capture via x264-crf32 | 7 | 183.82 | 456.45 | 1823.8 | 100% |
| replay of genuine capture via mp4v | 7 | 1816.35 | 2541.35 | 12467.11 | 100% |
| photo re-animation of a genuine frame | 7 | 2.37 | 9.34 | 25.24 | 0% |
| static replay of a genuine frame | 7 | 734.15 | 1454.24 | 9422.06 | 100% |

Impostor accept rate: 0.00% of 64 clips; latency per test 191.81 ms (median).

## Gate 2 - blind motion-gated PRNU (advisory)

N = 45 frames

| Group | PRESENT | ABSENT | INCONCLUSIVE |
|---|---|---|---|
| attack: photo re-animation | 0 | 3 | 0 |
| attack: static replay | 0 | 0 | 3 |
| fake: face-swap (DF40) | 11 | 2 | 15 |
| fake: reenactment (DF40) | 0 | 3 | 5 |
| fake: talking-head (DF40) | 2 | 1 | 9 |
| fake: text-to-video | 3 | 4 | 9 |
| real: phone WhatsApp (VISION) | 0 | 6 | 6 |
| real: phone native (VISION) | 1 | 16 | 5 |
| real: webcam (this machine) | 2 | 0 | 9 |

N = 90 frames

| Group | PRESENT | ABSENT | INCONCLUSIVE |
|---|---|---|---|
| attack: photo re-animation | 0 | 1 | 2 |
| attack: static replay | 0 | 0 | 3 |
| fake: face-swap (DF40) | 16 | 4 | 8 |
| fake: reenactment (DF40) | 0 | 3 | 5 |
| fake: talking-head (DF40) | 4 | 1 | 7 |
| fake: text-to-video | 7 | 3 | 6 |
| real: phone WhatsApp (VISION) | 0 | 9 | 3 |
| real: phone native (VISION) | 4 | 16 | 2 |
| real: webcam (this machine) | 2 | 0 | 9 |

## Gate 3 - temporal consistency (unchanged module), N = 90

| Group | flagged | passed | abstained (<8 face frames) |
|---|---|---|---|
| fake: face-swap (DF40) | 0 | 28 | 0 |
| fake: reenactment (DF40) | 0 | 8 | 0 |
| fake: talking-head (DF40) | 0 | 12 | 0 |
| fake: text-to-video | 2 | 7 | 7 |
| real: phone WhatsApp (VISION) | 0 | 3 | 9 |
| real: phone native (VISION) | 3 | 11 | 8 |
| real: webcam (this machine) | 0 | 2 | 9 |

## Gate 1 - active sensor challenge

Null (no response) false-pass rate: 0 / 32700 trials over 109 real luma traces (95% upper bound 9.17e-05); null correlation p99 = 0.593, max = 0.831 (threshold 0.9).

Live on /dev/video0: 20/20 genuine passes (corr 0.91-1.0, effect 48.75 grey levels, 1583.85 ms); replayed frames rejected 11/11.

Gate 1 passive probe latency: median 9.71 ms (min 9.41, max 12.19).
