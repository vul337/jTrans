import os
import subprocess
import multiprocessing
import time
from util.pairdata import pairdata

ida_path = "./ida/idat64"
work_dir = os.path.abspath('.')
dataset_dir = './dataset/'
strip_path = "./dataset_strip/"
script_path = f"./process.py"
SAVE_ROOT = "./extract"

def getTarget(path, prefixfilter=None):
    target = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if prefixfilter is None:
                target.append(os.path.join(root, file))
            else:
                for prefix in prefixfilter:
                    if file.startswith(prefix):
                        target.append(os.path.join(root, file))
    return target

if __name__ == '__main__':
    # prefixfilter = ['libcap-git-setcap']
    start = time.time()
    target_list = getTarget(dataset_dir)

    pool = multiprocessing.Pool(processes=8)
    for target in target_list:
        filename = target.split('/')[-1]
        filename_strip = filename  + '.strip'
        ida_input = os.path.join(strip_path, filename_strip)
        os.system(f"strip -s {target} -o {ida_input}")
        print(f"strip -s {target} -o {ida_input}")

        cmd_str = f'{ida_path} -Llog/{filename}.log -c -A -S{script_path} -oidb/{filename}.idb {ida_input}'
        print(cmd_str)
        cmd = [ida_path, f'-Llog/{filename}.log', '-c', '-A', f'-S{script_path}', f'-oidb/{filename}.idb', f'{ida_input}']
        pool.apply_async(subprocess.call, args=(cmd,))
    pool.close()
    pool.join()
    print('[*] Features Extracting Done')
    pairdata(SAVE_ROOT)
    end = time.time()
    print(f"[*] Time Cost: {end - start} seconds")
