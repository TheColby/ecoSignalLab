# Validation Harness

`esl validate` runs dataset-level regression and quality checks and writes machine-readable reports.

Command:

```bash
esl validate input_dir --out validation_out --rules rules.json
```

Outputs:
- `validation_out/validation_report.json`
- `validation_out/validation_summary.csv`

## Rules Format

Rules may be JSON or YAML.

```json
{
  "metric_thresholds": {
    "snr_db": { "min": 6.0 },
    "clipping_ratio": { "max": 0.001 }
  },
  "validity_flags": {
    "clipping": false,
    "ir_tail_low_snr": false
  }
}
```

Semantics:
- `metric_thresholds.<metric>.min`: fail if metric mean is below this value.
- `metric_thresholds.<metric>.max`: fail if metric mean is above this value.
- `validity_flags.<flag>`: fail if emitted validity flag does not equal expected value.

## Report Fields

`validation_report.json` includes:
- `files_checked`
- `files_passed`
- `files_failed`
- `rules` snapshot
- `file_reports` with per-file failures and selected metric means

`validation_summary.csv` includes one row per check:
- file path
- check type (`metric` or `validity_flag`)
- expected bounds/flag
- status (`pass`/`fail`)
- failure message
