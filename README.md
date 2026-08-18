# NetMob 2026 data challenge project

This repository now follows a staged, contamination-resistant workflow. Generated
data never overwrite an earlier layer.

```text
data_original
    -> Data_cleaning_code -> data_after_cleaning
    -> Data_matching_code -> data_after_all_processed
    -> Analysis_code      -> Output/Table and Output/Figure
                                      -> Output/overleaf -> Output/Results_pdf
```

## Directory roles

- `data_original/`: unchanged files from the NetMob challenge download.
- `supplementary_material/`: the official `data_description.pdf` and its note.
- `Data_cleaning_code/`: source inspection and conservative cleaning code.
- `data_after_cleaning/`: cleaned CSV copies plus QA reports.
- `Data_matching_code/`: route, map-matching, supply/demand, phase, and index code.
- `data_after_all_processed/`: final reusable CSV feature tables.
- `Analysis_code/`: tables and figures; reads only final processed data.
- `Documentation/`: method notes, research plan, verified literature, and personal
  oral-presentation notes.
- `Output/`: tables, figures, LaTeX sources, and compiled result PDFs.
- `Archive/legacy_pipeline/`: frozen earlier code, outputs, reports, and vehicle-ID
  matching audit notebooks, with a file-by-file explanation.

## Rebuild order

From this project root, run:

```bash
/Users/pirin/miniconda3/bin/python Data_cleaning_code/run_cleaning.py
/Users/pirin/miniconda3/bin/python Data_matching_code/build_processed_data.py
/Users/pirin/miniconda3/bin/python Analysis_code/run_analysis.py
```

All derived data are CSV. The original Ticket timestamp is preserved, while the
empirically suggested +3-hour version is a separate, explicitly unconfirmed column.
Known anomalous Mobility dates are retained with quality flags rather than silently
removed.

## Vehicle matching status

The archived robustness audit finds strong candidate signal but no defensible complete
vehicle crosswalk. Ticket and Mobility vehicle identifiers have zero exact overlap;
route and time can rank candidates, but stability and one-to-one uniqueness are too
weak without an external identifier. See
`Archive/legacy_pipeline/vehicle_matching_audit/04_final_audit_report.ipynb`.

## Documentation and historical results

Research notes are collected under `Documentation/`. Historical Week 1–2 results and
the reports produced by the earlier mixed pipeline are retained under
`Archive/legacy_pipeline/`.
