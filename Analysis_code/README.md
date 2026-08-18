# Analysis code

This folder contains analysis and presentation code. Its strict input is
`../data_after_all_processed/`; it does not read original or intermediate data.

## Script

- `run_analysis.py`: writes descriptive CSV tables to `../Output/Table/` and PNG
  figures to `../Output/Figure/`. It reports the provider clock and the unconfirmed
  +3-hour Ticket clock separately, and it does not claim that either clock is correct.

Run from the project root:

```bash
/Users/pirin/miniconda3/bin/python Analysis_code/run_analysis.py
```

Publication figures inherited from the earlier pipeline are retained in
`Output/Figure/`; their historical producing code is documented under
`Archive/legacy_pipeline/`.
