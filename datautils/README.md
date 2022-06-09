# Prerequests
- linux ida
- IDA Python (python3) with networkx, pyelftools, binaryai
```bash
python3 -m pip install pyelftools binaryai networkx
```

# Quick Start

## Directory Description
- dataset (original binaries)
- dataset_strip (temp directory for strip binary)
- extract (extracted feature)
- ida (linux ida directory)
- idb (idb files for orignial binaries)
- log (processing log)
- util (scripts utilities)
    - base.py (binary process base class)
    - pairdata (pair the groudtruth for functions with different optimization)
- process.py (IDA Python scripts for extrating features of binaries)
- playdata.py (play with the extracted features)
- run.py (parallel run)

# Usage
## Extracting features for binary similarity task
- copy all the compiled binaries with symbol table to dataset/
- change config.py for the suitable parameters
- run the following commands
```bash
./ida/idapyswitch # switch to system python3
python3 run.py
```
    
## Use the extracted features 
- Have a look at util/playdata.py
- There are two types of processed datasets, one for unsupervised learning (unpair_data) and another for supervised learning (pair_data), which are stored in .pickle files
- unpair data
    ```python
    unpair_data = {
        'foo': [
                0x400000, # function_addr
                ['sub rbp, rsp', 'ret'], # asm_list
                b"\x48\x29\xe5\xc3", # raw bytes
                cfg, # networkx DiGraph
                binaryai_feature
            ],
        'bar': [
            ...
        ]
    }
    # cfg traverse node
    def traverse_cfg_node(self, cfg):
        for node in cfg.nodes():
            yield cfg.nodes[node]['asm'], cfg.nodes[node]['raw']
    
    # cfg create code
    def get_cfg(self, func):

        def get_attr(block):
            asm,raw=[],b""
            curr_addr = block.start_ea
            while curr_addr < block.end_ea:
                asm.append(idc.GetDisasm(curr_addr))
                raw+=idc.get_bytes(curr_addr, idc.get_item_size(curr_addr))
                curr_addr = idc.next_head(curr_addr, block.end_ea)
            return asm, raw

        nx_graph = nx.DiGraph()
        flowchart = idaapi.FlowChart(idaapi.get_func(func), flags=idaapi.FC_PREDS)
        for block in flowchart:
            # Make sure all nodes are added (including edge-less nodes)
            attr = get_attr(block)
            nx_graph.add_node(block.start_ea, asm=attr[0], raw=attr[1])

            for pred in block.preds():
                nx_graph.add_edge(pred.start_ea, block.start_ea)
            for succ in block.succs():
                nx_graph.add_edge(block.start_ea, succ.start_ea)
        return nx_graph 
    ```
- pair data
    the pair data is organized by the groudtruth (paired functions compiled by diferent optimization)
    ```python
    pair_data = {
            'foo': [
                unpair_foo_O0, # unpair_func_foo_O0
                unpair_foo_O1, # unpair_func_foo_O1
                unpair_foo_O2, # unpair_func_foo_O2
                ...
            ],
            'bar': [
                ...
            ]
        }
    ```
