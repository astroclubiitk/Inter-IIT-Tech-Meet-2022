from jux import jux
import os
import sys
import warnings

warnings.filterwarnings("ignore")

try:
    path_to_lc = sys.argv[1]
    path_to_pickle = sys.argv[2]
except:
    print("Provide cmd arguments")
if os.path.exists(path_to_lc):
    print(os.path.basename(path_to_lc))
    lc = jux.Lightcurve(path_to_lc)
    outlier_details = lc.main(True, path_to_pickle)
    print(outlier_details)
else:
    print("File doesn't exist!")
