# Subject00 Condition Slot Binding

Task: `AAAI27-SUBJECT00-DATA-PREPARATION-GENERATION-READY-001`

The front-axis calibration is `view_orientation=(camera_azimuth+90 degrees) mod 360`. Pose 0 is frozen for every garment and every slot. Each selected camera is strict-train, has a valid RGB/mask pair, uses the frozen shared calibration/SMPL-X files, and is within the 22.5-degree yaw tolerance.

| Slot | Semantic view | Target degrees | Camera | Pose | Error degrees | Split |
|---|---|---:|---:|---:|---:|---|
| slot_00 | front | 0 | 17 | 0 | 0.622 | STRICT_TRAIN |
| slot_01 | front_left_three_quarter | 45 | 21 | 0 | 5.519 | STRICT_TRAIN |
| slot_02 | front_right_three_quarter | 315 | 14 | 0 | 0.826 | STRICT_TRAIN |
| slot_03 | left | 90 | 23 | 0 | 8.561 | STRICT_TRAIN |
| slot_04 | right | 270 | 11 | 0 | 10.194 | STRICT_TRAIN |
| slot_05 | back_left_three_quarter | 135 | 2 | 0 | 1.461 | STRICT_TRAIN |
| slot_06 | back_right_three_quarter | 225 | 9 | 0 | 5.911 | STRICT_TRAIN |
| slot_07 | back | 180 | 5 | 0 | 1.795 | STRICT_TRAIN |

The same eight source conditions are reused for O01, O03, and O04, yielding 24 complete source bindings. Cardinal slots 00/03/04/07 are both `GARMENT_REFERENCE` and `TEACHER_TARGET`, for 12 reference bindings. This does not claim generated pixels: accepted Teacher targets and references remain `MATERIALIZATION_PENDING`, with actual count zero.

Held-out cameras `[0,4,8,12,16,20]`, held-out poses, buffer-only poses, and ambiguous assets are forbidden. Current split leakage is zero.
