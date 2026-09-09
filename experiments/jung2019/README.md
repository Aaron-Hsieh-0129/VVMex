# Jung 2019 experiments

Experiment-specific assets live here:

- `configs/`: reproducible CASE 1 template and the user's `Jung_fig6.json` (all settings, including its external output path, are preserved; that config was not launched).
- `tools_jung2019/`: configure/launch, diagnostics, archive comparison and refinement analysis.
- `tests/`: shared-model integration/rejection test, CTest registration and independent L2-analysis tests.
- `rll-section42-*.md`: scientific protocol, results and handoff.
- Named run directories: preserved configurations, provenance, output and plots.
- `testing_output/`: new integration-test evidence, separated by build directory name.

Run from the repository root:

```sh
python experiments/jung2019/tools_jung2019/experiment.py --help
python -m unittest discover -s experiments/jung2019/tests -p test_convergence.py -v
ctest --test-dir build -R '^test_jung2019_shared_model$' --output-on-failure
```

Regenerate existing CMake build trees after relocation (`cmake --preset blaze -DBUILD_TESTS=ON`, or `blaze-cpu`). The existing CTest name and backend/resource settings are retained. Shared runtime implementation stays in `src`; generic RLL operator tests remain in `tests/unit`. Historical run provenance and previous build-directory test results are not rewritten or moved.

The repository ignores `experiments/`; the small source/configuration/documentation files here are explicitly tracked, while generated results remain ignored. Never add the entire directory with `git add -f`.
