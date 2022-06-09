import idc
import idautils
import idaapi
import pickle
import binaryai
import networkx as nx
from util.base import Binarybase

SAVEROOT = "./extract" # dir of pickle files saved by IDA
DATAROOT = "./dataset" # dir of binaries (not stripped)

class BinaryData(Binarybase):
    def __init__(self, unstrip_path):
        super(BinaryData, self).__init__(unstrip_path)
        self.fix_up()
    
    def fix_up(self):
        for addr in self.addr2name:
            # incase some functions' instructions are not recognized by IDA
            idc.create_insn(addr)  
            idc.add_func(addr) 

    def get_asm(self, func):
        instGenerator = idautils.FuncItems(func)
        asm_list = []
        for inst in instGenerator:
            asm_list.append(idc.GetDisasm(inst))
        return asm_list

    def get_rawbytes(self, func):
        instGenerator = idautils.FuncItems(func)
        rawbytes_list = b""
        for inst in instGenerator:
            rawbytes_list += idc.get_bytes(inst, idc.get_item_size(inst))
        return rawbytes_list

    def get_cfg(self, func):

        def get_attr(block, func_addr_set):
            asm,raw=[],b""
            curr_addr = block.start_ea
            if curr_addr not in func_addr_set:
                return -1
            # print(f"[*] cur: {hex(curr_addr)}, block_end: {hex(block.end_ea)}")
            while curr_addr <= block.end_ea:
                asm.append(idc.GetDisasm(curr_addr))
                raw+=idc.get_bytes(curr_addr, idc.get_item_size(curr_addr))
                curr_addr = idc.next_head(curr_addr, block.end_ea)
            return asm, raw

        nx_graph = nx.DiGraph()
        flowchart = idaapi.FlowChart(idaapi.get_func(func), flags=idaapi.FC_PREDS)
        func_addr_set = set([addr for addr in idautils.FuncItems(func)])
        for block in flowchart:
            # Make sure all nodes are added (including edge-less nodes)
            attr = get_attr(block, func_addr_set)
            if attr == -1:
                continue
            nx_graph.add_node(block.start_ea, asm=attr[0], raw=attr[1])
            # print(f"[*] bb: {hex(block.start_ea)}, asm: {attr[0]}")
            for pred in block.preds():
                if pred.start_ea not in func_addr_set:
                    continue
                nx_graph.add_edge(pred.start_ea, block.start_ea)
            for succ in block.succs():
                if succ.start_ea not in func_addr_set:
                    continue
                nx_graph.add_edge(block.start_ea, succ.start_ea)
        return nx_graph  

    def get_binai_feature(self, func):
        return binaryai.ida.get_func_feature(func)

    def extract_all(self):
        for func in idautils.Functions():
            if idc.get_segm_name(func) in ['.plt','extern','.init','.fini']:
                continue
            print("[+] %s" % idc.get_func_name(func))
            asm_list = self.get_asm(func)
            rawbytes_list = self.get_rawbytes(func)
            cfg = self.get_cfg(func)
            bai_feature = self.get_binai_feature(func)
            yield (self.addr2name[func], func, asm_list, rawbytes_list, cfg, bai_feature)

if __name__ == "__main__":
    import os
    from collections import defaultdict

    assert os.path.exists(DATAROOT), "DATAROOT does not exist"
    assert os.path.exists(SAVEROOT), "SAVEROOT does not exist"

    binary_abs_path = idc.get_input_file_path()
    filename = binary_abs_path.split('/')[-1][:-6]
    unstrip_path = os.path.join(DATAROOT, filename)
    idc.auto_wait()
    binary_data = BinaryData(unstrip_path)

    saved_dict = defaultdict(lambda: list)
    saved_path = os.path.join(SAVEROOT, filename + "_extract.pkl") # unpair data
    with open(saved_path, 'wb') as f:
        for func_name, func, asm_list, rawbytes_list, cfg, bai_feature in binary_data.extract_all():
            saved_dict[func_name] = [func, asm_list, rawbytes_list, cfg, bai_feature]
        pickle.dump(dict(saved_dict), f)
    idc.qexit(0) # exit IDA