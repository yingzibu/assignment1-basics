# CS336 Experiment Log

Date: 2026-10-15

## Status

Logging infrastructure is implemented. Model-training experiments have not been run in this conversation. All experiment rows below are planned; loss and timing fields remain blank until measured.

## Files and measurements

Each run uses a separate directory, for example `artifacts/experiments/baseline`.

- `config.json`: model, optimizer, data, batch, schedule, and seed settings.
- `metrics.jsonl`: train/validation split, optimizer update, loss, cumulative elapsed seconds, UTC timestamp, and optional learning rate/throughput.
- `summary.json`: status, last logged update, elapsed time, and best logged validation loss.
- `latest.pt` and `best.pt`: training checkpoints, including elapsed seconds and random states.
- Resume archives the previous metrics file as `metrics.before_resume_<timestamp>.jsonl`, then retains records through the restored checkpoint update.

Elapsed time starts when the logger is initialized, accumulates across active training/validation sessions, and excludes offline downtime. A logging step counts one optimizer update; gradient accumulation does not increase that counter until the optimizer updates.

Legacy records without `elapsed_seconds` can appear in the update plot. Their earlier wall times cannot be recovered; they are omitted from the time plot.

## Reference model settings

The handout's TinyStories reference settings are: vocabulary 10000; context length 256; d_model 512; d_ff 1344; 4 layers; 16 heads; RoPE theta 10000. Set d_ff explicitly to 1344 to use this profile.

The earlier small training example uses d_model 256, d_ff 704, 4 layers, and 8 heads. Record the selected profile and actual configuration for each experiment.

With gradient accumulation, tokens processed = batch_size × grad_accum_steps × completed_updates × context_length. Keep the comparison budget and validation sampling procedure fixed and record the chosen budget below.

## Planned experiment register

| ID | Question | Changed setting | Controls | Seed | Status | Best validation loss | Best update | Elapsed seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Establish the selected reference configuration | None | Same tokenizer, data, validation windows, and token budget | 42 | Not run | — | — | — |
| lr_low | Compare convergence at a lower learning rate | max_lr = 1e-4 | Same as baseline; record min_lr and warmup | 42 | Not run | — | — | — |
| lr_mid | Compare the initial learning-rate choice | max_lr = 3e-4 | Same as baseline; record min_lr and warmup | 42 | Not run | — | — | — |
| lr_high | Inspect stability at a larger learning rate | max_lr = 1e-3 | Same as baseline; record min_lr and warmup | 42 | Not run | — | — | — |
| batch_variant | Compare effective batch sizes | Choose batch size and accumulation | Same token budget; retune learning rate if needed | 42 | Not run | — | — | — |
| ablation | Test one architectural component | Specify one change | Same remaining configuration and evaluation | 42 | Not run | — | — | — |

## Per-experiment entry

Copy this entry for every actual run.

- Run ID and directory:
- Date, device, and relevant software versions:
- Hypothesis:
- Changed variable:
- Model profile and configuration:
- Tokenizer/data paths or dataset identifiers:
- Microbatch size, accumulation, and effective batch size:
- Target updates and total token budget:
- Learning rate, minimum rate, warmup, and cosine endpoint:
- AdamW betas, epsilon, weight decay, and gradient clipping:
- Seed:
- Validation protocol:
- Exact command:
- Checkpoint and resume history:
- Status: planned / running / completed / interrupted / diverged
- Best logged validation loss and update:
- Final logged validation loss:
- Elapsed seconds and throughput:
- Curves against updates and wall time:
- Observations:
- Conclusion supported by this run:
- Next experiment:

## Plotting and summaries

After actual runs produce metrics:

```bash
uv run --with matplotlib python -m cs336_basics.experiment_log \
  artifacts/experiments/baseline artifacts/experiments/lr_high \
  --output artifacts/experiments/lr_comparison
```

This produces `lr_comparison.png` and `lr_comparison.pdf`, each containing loss-versus-updates and loss-versus-wall-time panels.

```bash
uv run python -m cs336_basics.experiment_log \
  artifacts/experiments/baseline --summary
```

Fill the results table from actual measurements. The code-verification fixtures are synthetic and do not constitute language-model experiments.

## Infrastructure checks

Passed checks using a controlled clock and synthetic records: JSONL fields, configuration persistence, summary generation, cumulative timing, resume backups and rollback, invalid-record handling, incomplete final lines, headless PNG/PDF rendering, command-line summaries, and training-script syntax/interface wiring.

PyTorch is unavailable in the verification environment. Training, forward/backward execution, and checkpoint round trips have not been performed.

