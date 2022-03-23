from jux import jux
import os
import sys

fn = sys.argv[1]
if os.path.exists(fn):
    print(os.path.basename(fn))
    lc = jux.Lightcurve(fn)
    model_zip = lc.main()
    print(lc.flare_details)
else:
    print("File doesn't exist!")
