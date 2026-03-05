import os
import ogstools as ot
from pathlib import Path
###########################################################
#5: run simulation OGS
prj_file = "stimtec-bh10_20180716.prj"
out_dir = Path(os.environ.get("OGS_TESTRUNNER_OUT_DIR", "_out"))
out_dir.mkdir(parents=True, exist_ok=True)
model = ot.Project(input_file=prj_file, output_file=f"{out_dir}/modified.prj")
model.write_input()
model.run_model(logfile=f"{out_dir}/out.txt", args=f"-o {out_dir} -m . -s .")
