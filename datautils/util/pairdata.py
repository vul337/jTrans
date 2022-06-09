import os
from collections import defaultdict
from tqdm import tqdm
from shutil import move
import pickle
from functools import reduce
import networkx as nx

def pairdata(data_dir):
    def get_prefix(path): # get proj name
        l = path.split('-')
        prefix = '-'.join(l[:-2])
        return prefix.split('/')[-1]

    proj2file = defaultdict(list) # proj to filename list
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in tqdm(files):
            pickle_path = os.path.join(root, name)
            prefix = get_prefix(pickle_path)
            proj2file[prefix].append(name)

    for proj, filelist in proj2file.items():
        if not os.path.exists(os.path.join(data_dir, proj)):
            os.mkdir(os.path.join(data_dir, proj))

        binary_func_list = []
        pkl_list = []
        for name in filelist:
            src = os.path.join(data_dir, name)
            dst = os.path.join(data_dir, proj, name)
            pkl = pickle.load(open(src, 'rb'))
            pkl_list.append(pkl)
            func_list = []
            for func_name in pkl:
                func_list.append(func_name)
            print(name, len(func_list))
            binary_func_list.append(func_list)
            move(src, dst) # move file into proj dir
        
        final_index = reduce(lambda x,y : set(x) & set(y), binary_func_list)
        print('all', len(final_index))

        saved_index = defaultdict(list)
        for func_name in final_index:
            for pkl in pkl_list:
                saved_index[func_name].append(pkl[func_name])

        saved_pickle_name = os.path.join(data_dir, proj, 'saved_index.pkl') # pari data
        pickle.dump(dict(saved_index), open(saved_pickle_name, 'wb'))