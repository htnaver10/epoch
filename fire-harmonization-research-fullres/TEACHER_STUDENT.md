# VIIRS teacher to MODIS-only student

This experiment adds a VIIRS thermal-image teacher to the existing MODIS student.
It preserves the completed supervised experiment, its source code and artifacts.
The current experiment is `teacher_student_v1`.

Run from the full-resolution research folder:

```powershell
.\run-teacher-student.cmd
```

The same command resumes completed downloads, paired preparation and epoch-boundary
checkpoints. Completed local data do not require another Earthdata login. Individual
stages are `inventory`, `download`, `prepare`, `teacher`, `cache`, `student`, and
`evaluate`, for example `.\run-teacher-student.cmd teacher`. Configuration and
training sources are frozen; changes require a new experiment version.

## What each network sees

| Network | Image inputs | Supervision | Required at inference |
|---|---|---|---|
| VIIRS teacher | Native I04/I05 thermal radiances plus two validity masks | Confident VIIRS fire/clear query labels | VIIRS imagery |
| MODIS student | The original MODIS thermal, reflective and geometry inputs plus query coordinates | The original VIIRS labels, native auxiliary labels and frozen teacher predictions | MODIS inputs only |

The teacher has 488,145 parameters. It uses a spatial CNN on one unpadded native
VIIRS crop at a time, accumulating gradients over an effective batch. Processing
each crop at its real shape avoids changing spatial normalization with padding.
Its output head starts at the supported training-label prevalence.

The student's 1,982,050-parameter `NativeFireNet` architecture is unchanged. It is
initialized afresh with the same seed and training-only priors as the supervised
baseline. No teacher network, VIIRS image tensor or teacher soft-target file is
loaded by ordinary student inference.

## Data and calibration

- The parent dataset and split assignments are frozen by SHA-256. All 1,100 parent
  tiles are retained: 844 training, 172 validation and 84 test. Their fire-reference
  counts remain 1,913, 1,757 and 95 respectively.
- Seventeen exact VNP02IMG collection-002 imagery companions are selected by
  acquisition timestamp and locked NASA granule ID. Actual new transfers total
  3.593 GiB, within the 16 GiB budget. Downloads retain a 40 GiB free-space reserve.
- VIIRS thermal calibration follows CF scaling: `radiance = SI * scale_factor +
  add_offset`, using the actual file attributes. Fill/special values, saturation,
  bowtie deletion, missing data and advertised anomalous quality flags are masked.
  These inputs are thermal imagery; the VIIRS fire mask is never an input channel.
- Each VIIRS crop must exactly reproduce the frozen student's saved latitude and
  longitude arrays. The native imagery and geolocation dimensions and acquisition
  times must agree. No interpolation of the VIIRS teacher image is used.
- Teacher radiance normalizers are fitted on training imagery only, after
  `log1p` transformation. Invalid pixels have zero normalized value and an explicit
  false validity flag. Label weights require at least one valid thermal channel.
  The actual paired audit retains 100% of the parent's known label exposure.

## Training and distillation objective

1. Fit the VIIRS teacher for 30 epochs with masked, exposure-normalized binary
   cross entropy. Select the best checkpoint by development-validation log loss.
2. Check that the selected teacher's validation Brier score beats always-clear
   and its average precision exceeds both 0.01 and five times reference prevalence.
   This gate is not evidence of verified wildfire truth or broad generalization.
3. Freeze the teacher (`eval`, no parameter gradients). Cache raw teacher logits
   for exactly the training tiles. No validation/test soft-target cache is allowed.
   The cache binds teacher, imagery, student tile and manifest hashes.
4. Train the MODIS student for 30 epochs with:

```text
L = sum(w_hard * BCE(student_logit, VIIRS_label)) / sum(w_hard)
  + 0.2 * sum(w_native * BCE(native_logit, native_label)) / sum(w_native)
  + 0.5 * T^2 * sum(w_soft * KL(Bernoulli(p_teacher) || Bernoulli(p_student)))
                 / sum(w_soft)

T = 2
p_teacher = sigmoid(detached_teacher_logit / T)
p_student = sigmoid(student_logit / T)
```

The KL includes both fire and clear probabilities. Each term has its own actual
exposure denominator across the accumulated optimizer window. Hard-label
supervision retains full weight. Unknown or unsupported pixels receive zero
distillation weight. Zero-support terms are omitted safely. Nonfinite losses and
gradients refuse the optimizer update.

5. Select the student using hard-label development-validation log loss. Freeze
   its intercept adjustment and MODIS-matched false-positive cutoff on validation
   before collecting its test predictions. Compare raw and adjusted policies
   with regular MODIS and the previous supervised model on identical queries.

**The 2024 test benchmark was examined in the earlier run.** This is a controlled
comparison on a reused benchmark, not a new blind test. The single development
cohort and correlated query labels remain limitations. No post-test threshold
retuning or claim of operational wildfire usefulness is made.

## Outputs

```text
data/manifests/teacher_student_v1_pairs.jsonl
data/manifests/teacher_student_v1_teacher_stats.json
data/manifests/teacher_student_v1_soft_targets.json
models/teacher_student_v1_teacher/best.pt
models/teacher_student_v1_teacher/last.pt
models/teacher_student_v1_student/best.pt
models/teacher_student_v1_student/last.pt
models/teacher_student_v1_student/student_inference.pt
results/teacher_student_v1/teacher_quality.json
results/teacher_student_v1/operating_policy.json
results/teacher_student_v1/EVALUATION.md
results/teacher_student_v1/evaluation.json
```

`student_inference.pt` omits teacher weights, soft targets and optimizer state. It
loads with the existing MODIS-only checkpoint loader and `run.py infer` command.
Resume state is kept separately in `last.pt`; an interrupted checkpoint/log commit
can restore its missing epoch-log entry without duplicating an epoch.

## Evidence and sources

All 81 software tests passed, including CF radiance scaling, special-value and QA
masking, exact coordinate alignment, train-only statistics, source isolation,
temperature-scaled KL gradients, teacher detachment, unequal-exposure accumulation,
cache tampering, unknown-label masks, checkpoint/history recovery, a CPU end-to-end
teacher/cache/student/export round trip, and a CUDA/BF16 student update.
The actual-data teacher profile passed on the largest admitted query crop.

[Test log](results/teacher_student_v1/software_tests.log),
[paired data audit](results/teacher_student_v1/paired_data_audit.json),
[GPU profile](results/teacher_student_v1/teacher_gpu_profile.json),
[experiment configuration](configs/teacher_student_v1.json).

Product documentation: [NASA VNP02IMG](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP02IMG/),
[NASA VIIRS L1B file specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/VIIRS/1/VNP02IMG.fs).
Loss semantics: [PyTorch KL divergence](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html).
