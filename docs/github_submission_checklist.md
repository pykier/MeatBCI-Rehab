# GitHub Submission Checklist

## Include

- `metabci/`
  - Platform dataset, algorithm, brainstim and brainflow extensions.
- `demos/rehab_mi/`
  - Competition launchers, assets, config and demo README.
- `tests/`
  - Non-hardware tests and import checks.
- `docs/`
  - Scoring mapping, hardware setup, dataset card and project test report.
- `README.md`
  - Top-level reproduction and scoring evidence.
- `requirements.txt`, `setup.py`, `pyproject.toml`, `LICENSE`.

## Exclude

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `my_data/` unless a tiny anonymized sample is intentionally added.
- `checkpoints/`
- `log.txt` and runtime logs.
- Raw private EEG data without anonymization and release approval.
- Large trained models unless they are intentionally published through Release
  assets or Git LFS.
- Local Word drafts, screenshots and WeChat temporary files.

## Recommended Repository Layout

```text
MetaBCI-Rehab/
  metabci/
  demos/
    rehab_mi/
      assets/
      experiment_config.json
      README.md
  docs/
    scoring_mapping.md
    project_test_report_template.md
    dataset_card.md
    hardware_setup.md
    github_submission_checklist.md
  tests/
  README.md
  requirements.txt
  setup.py
```

## Submission Steps

1. Run `git status --short` and remove unintended untracked files.
2. Confirm `.gitignore` excludes runtime data, logs, environments and caches.
3. Run at least the import/smoke tests that do not require hardware.
4. Run hardware preflight and save the terminal log.
5. Run one offline pipeline and save the terminal log.
6. Run one online VR closed-loop demo and save the prediction log.
7. Fill `docs/project_test_report_template.md` with actual screenshots and logs.
8. Add the full dataset link or GitHub Release link.
9. Tag the final submission version.

## Minimum Evidence For Reviewers

- A command that checks EEG and robot hardware.
- A command that collects offline data and trains a model.
- A command that starts the online VR closed loop.
- A scoring table mapping each score item to a code path.
- A dataset card with subject count and event definitions.
- A video showing EEG stream, VR page, stimulus window and robot feedback.
