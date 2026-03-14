# Releasing `realtimestt-xyz`

## Goals

- Ship immutable release artifacts for `RealtimeVoiceChat` and other downstream consumers.
- Keep the import path stable as `RealtimeSTT`.
- Track `whisper.cpp` intentionally through a pinned submodule revision.

## Release Flow

1. Update the package version in `pyproject.toml` and `setup.py`.
2. Update the changelog or release notes context in `README.md` if needed.
3. Sync the vendored upstream revision when desired:
   ```bash
   git submodule update --init --recursive
   git -C third_party/whisper.cpp fetch origin
   git -C third_party/whisper.cpp checkout <upstream-commit>
   git add third_party/whisper.cpp
   git commit -m "chore: bump whisper.cpp"
   ```
4. Verify locally:
   ```bash
   python3 -m unittest tests.unit.test_audio_recorder_contract tests.unit.test_whisper_cpp_support
   python3 -m build --sdist
   ```
5. Tag the release:
   ```bash
   git tag v<version>
   git push origin master --tags
   ```
6. GitHub Actions will:
   - build an sdist
   - build macOS and Linux wheels
   - attach all artifacts to a GitHub Release
   - publish to PyPI via Trusted Publishing if the repository is configured as a trusted publisher for `realtimestt-xyz`

## Downstream Consumption

- Preferred once PyPI is live:
  ```txt
  realtimestt-xyz==<version>
  ```
- Canonical fallback before or alongside PyPI:
  ```txt
  realtimestt-xyz @ https://github.com/BarakXYZ/RealtimeSTT/releases/download/v<version>/realtimestt_xyz-<version>.tar.gz
  ```

The package name is distinct from upstream, but downstream code still imports:

```python
from RealtimeSTT import AudioToTextRecorder
```
