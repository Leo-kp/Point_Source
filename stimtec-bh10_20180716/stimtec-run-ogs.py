import os
import ogstools as ot
from pathlib import Path
###########################################################

# This finds the folder where THIS script is saved
script_dir = Path(__file__).parent.resolve()

# 5: run simulation OGS
# Point to the file relative to the script location
prj_file = script_dir / "stimtec-bh10_20180716.prj"

out_dir = script_dir / "_out"
out_dir.mkdir(parents=True, exist_ok=True)

# Use the full path for the input_file
model = ot.Project(input_file=str(prj_file), output_file=f"{out_dir}/modified.prj")
model.write_input()
model.run_model(logfile=f"{out_dir}/out.txt", args=f"-o {out_dir} -m {script_dir} -s {script_dir}")

# #5: run simulation OGS
# prj_file = "stimtec-bh10_20180716.prj"
# out_dir = Path(os.environ.get("OGS_TESTRUNNER_OUT_DIR", "_out"))
# out_dir.mkdir(parents=True, exist_ok=True)
# model = ot.Project(input_file=prj_file, output_file=f"{out_dir}/modified.prj")
# model.write_input()
# model.run_model(logfile=f"{out_dir}/out.txt", args=f"-o {out_dir} -m . -s .")
