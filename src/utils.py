import os
import pandas as pd
from pathlib import Path

def create_directories(paths=None):
	"""Create necessary directories for processed data, figures, tables and models.

	If `paths` is provided it should be an iterable of paths. Otherwise common
	folders are created relative to the repository root.
	"""
	default_dirs = [
		os.path.join("data", "processed"),
		os.path.join("reports", "figures"),
		os.path.join("reports", "tables"),
		os.path.join("models"),
	]
	dirs = paths or default_dirs
	for d in dirs:
		Path(d).mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, filename: str):
	"""Save dataframe to reports/tables as CSV.

	filename may be a bare filename or a path under `reports/tables`.
	"""
	target = Path(filename)
	if not target.parent or str(target.parent) == ".":
		target = Path("reports") / "tables" / filename
	target.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(target, index=False)
	return str(target)

