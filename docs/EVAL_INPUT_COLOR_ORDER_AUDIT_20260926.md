# D1/D3 online evaluation color-order audit

The offline D1 generator decodes frames with `cv2.IMREAD_COLOR` and computes
appearance masks with `cv2.COLOR_BGR2GRAY` / `cv2.COLOR_BGR2HSV`. RoboTwin live
observations are RGB. Before this fix, `OnlineDepthProcessor._d1` passed live
RGB directly to `corrupt_frame`, so the appearance-dependent D1 noise did not
match the D1 data used for training. D3 uses D1 first and is also affected.

## Scope

- Affected: deployment D1 or D3 in evaluations importing
  `pi05_dpv_online_depth.py`, regardless of whether the trained checkpoint is
  D0, D1, or D3. Old scores and partial progress remain archived but must not
  be used for a matched train/deploy depth-method ranking.
- Not affected by this specific bug: deployment D0; offline D1/D3 generation;
  trained checkpoints; RGB-only evaluations. Other evaluation issues require
  separate audits.
- AutoDL old Randomized matrix `stack_blocks_two_random_matrix_20260925`:
  PI05 D3-trained x D1/D3 deployment (2 cells), ACT3 D0/D1/D3-trained x
  D1/D3 deployment (6 cells). The old queue is paused; corrected cells restart
  from seed zero in `stack_blocks_two_random_matrix_colorfix_20260926`.
- AutoDL old Easy ACT3 batch `act3_depth_matrix_resume_20260925`:
  completed D0-trained x D1/D3 deployment and D1/D3-trained cross-depth
  D1/D3 deployments (4 historical result directories). The documented
  D0-trained pair is queued for correction; D1/D3-trained cells overlap the
  corrected 56 Easy matrix.
- AutoDL old PI05 batch `pi05_dpv_autodl_20260925`:
  D0/D1-trained x Randomized D1/D3 deployment (4 cells), plus D3-trained x
  Easy D1/D3 deployment (2 cells). The former four are queued after the first
  corrected Randomized matrix; the latter two overlap the corrected 56 D3
  Easy window.
- 56 old Easy matrix `pi05_dpv_d0_d1_x_d0_d1_d3_official_20260921`:
  PI05 D0/D1-trained x D1/D3 deployment (4 cells).
- 56 old Easy matrix `act3_d1_d3_x_d0_d1_d3_official_20260921`:
  ACT3 D1/D3-trained x D1/D3 deployment (4 cells).
- 56 old resumable window `pi05_dpv_window_20260924`:
  PI05 D3-trained x Easy D1/D3 deployment (2 partial cells).

## Corrected runs

- AutoDL: `evaluations/stack_blocks_two_random_matrix_colorfix_20260926`
  has 8 independent, resumable cells.
- AutoDL: `evaluations/stack_blocks_two_doc_colorfix_20260926` waits for the
  first corrected Randomized matrix, then reruns 4 PI05 Randomized D1/D3 and
  2 ACT3 Easy D1/D3 cells that appear in the deployment-results document.
- 56: `evaluations/stack_blocks_two_easy_colorfix_20260926` has 8 Easy cells
  in fresh state directories.
- 56: `evaluations/pi05_dpv_d3_easy_colorfix_20260926` waits for the first
  56 Easy matrix, then starts 2 independently resumable PI05 D3 cells.
- Both servers passed a three-camera RGB-to-BGR regression test. The corrected
  online module SHA256 is
  `db73171a128eadf73ed289688ecd96f633bca2ef3613ea4d0dd21451b2e59aee`.
- D3 still feeds the original RGB to LingBot-Depth; only D1's appearance-noise
  function receives BGR. Old source files are retained with the
  `.pre_rgb_bgr_fix` suffix.

Do not resume old D1/D3 progress in a corrected run or compare old and new
results as if they had identical input definitions. Training is scheduled
after evaluation verification.
