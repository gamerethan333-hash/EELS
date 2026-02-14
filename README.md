# EELS Spectrum Image Analysis Script

This repository provides `eels_analysis.py`, a command-line tool to analyze and plot EELS spectrum images.

## Supported formats
- `*.npy` — 3D array `(ny, nx, nE)`
- `*.npz` — must contain `data`; optional `energy`
- `*.dm3` / `*.dm4` — via HyperSpy + RosettaSciIO

## Install dependencies
```bash
pip install numpy matplotlib hyperspy rosettasciio
```

## Run
```bash
python eels_analysis.py your_file.dm4 \
  --bg-start 240 --bg-end 270 \
  --int-start 280 --int-end 320 \
  --output-dir outputs
```

The script writes:
- `outputs/average_spectrum.png`
- `outputs/integrated_edge_map.png`

## Windows Git troubleshooting (from your screenshot)
Use these exact commands in **Git Bash** or **PowerShell** (not `C:\Windows\System32` CMD with malformed text):

```bash
cd /c/path/to/your/repo
# or in PowerShell:
# Set-Location C:\path\to\your\repo

git remote set-url origin https://github.com/gamerethan333-hash/EELS.git
git push -u origin work
```

Common issues shown in the screenshot:
- `系统找不到指定的路径` (`path not found`): you were in `C:\Windows\System32`, not the repo folder.
- `git\\` 不是内部或外部命令: command had an extra backslash/backtick character; type plain `git ...`.
- `git push -u origin work /workspace/EELS`: extra argument appended; correct form is only `git push -u origin work`.
