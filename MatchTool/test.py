import os
import glob
import subprocess
BASE_DIR = "tests"
test_dirs = os.listdir(BASE_DIR)

# python file since I do not want to do file stuff in cpp
write = False

for test in test_dirs:
    print("running test", test)
    src = glob.glob(f"{BASE_DIR}/{test}/Src*")[0]
    dst = glob.glob(f"{BASE_DIR}/{test}/Dst*")[0]
    # result = glob.glob(f"{BASE_DIR}/{test}/result")
    if write:
        result_path = f"{BASE_DIR}/{test}/result"
        with open(result_path, 'w') as f:
            subprocess.run(["out/main", src, dst], stdout=f)
    else:
        result_path = f"{BASE_DIR}/{test}/result"
        with open(result_path, 'r') as f:
            expected = f.read()
            proc = subprocess.run(["out/main", src, dst], stdout=subprocess.PIPE)
            if proc.stdout.decode() != expected:
                print("TEST", test, "FAILED!")
        

