import pickle
import re
import os
import readidadata

def tokenizer():
    with open('token_ida.pkl','rb') as f:
        token_id=pickle.load(f)
        f.close()
    return token_id

def seq_to_token(token_id,seq,UNK):
    ret=[]
    cnt=0
    for str in seq:
        re=token_id.get(str)
        if re!=None:
            ret.append(re)
        else:
            cnt+=1
            ret.append(UNK)       
    return ret

def normalize(opcode):
    opcode = opcode.replace(' - ', ' + ')
    opcode = re.sub(r'0x[0-9a-f]+', 'CONST', opcode)
    opcode = re.sub(r'\*[0-9]', '*CONST', opcode)
    opcode = re.sub(r' [0-9]', ' CONST', opcode)
    return opcode

def save_tokens(prefixs):
    token_id={}
    cnts=1
    file_cnt=0
    binlist=[]
    nowdir='../largedata/ourclean'
    docs = os.listdir(nowdir)
    for i in docs:
        pth=os.path.join(nowdir,i)
        for fi in os.listdir(pth):
            idx=fi.find('.')
            fd=False
            for pre in prefixs:
                if fi.startswith(pre):
                    fd=True
            if fd and fi.endswith('.nod'):
                fn=os.path.join(pth,fi[0:idx])
                binlist.append(fn+'.nod')
                print(fn)
    for fi in binlist:
        fii=open(fi,'rb')
        try:
	        asm_seq=pickle.load(fii)
        except:
            fii.close()
            continue
        else:
            fii.close()
            for bbid,addr,bb in asm_seq:
                    for addr,instructions in bb:
                        operator,operand1,operand2,operand3,annotation=readidadata.parse_asm(instructions)
                        if operator!=None:
                            if token_id.get(operator)==None:
                                print(operator,cnts," from ",hex(addr),instructions)
                                token_id[operator]=cnts
                                cnts+=1

                        if operand1!=None:
                            if not operand1.startswith('hex') and token_id.get(operand1)==None:
                                print(operand1,cnts," from ",hex(addr),instructions)
                                token_id[operand1]=cnts
                                cnts+=1

                        if operand2!=None:
                            if token_id.get(operand2)==None:
                                print(operand2,cnts," from ",hex(addr),instructions)
                                token_id[operand2]=cnts
                                cnts+=1

                        if operand3!=None:
                            if token_id.get(operand3)==None:
                                print(operand3,cnts," from ",hex(addr),instructions)
                                token_id[operand3]=cnts
                                cnts+=1
            file_cnt+=1
            print("finnish ",fi,file_cnt/len(binlist))
    print("token_number: ",cnts)
    with open('token_ida.pkl','wb') as f:
        pickle.dump(token_id,f)
        f.close()
    return token_id
if __name__ == "__main__":
    fi=open("vocab.txt","wb")
    tokens=save_tokens(['proxmark3','pythonqt','pizmidi','plasma','qbs','qcad','sc3','vice','virtualgl','vtk','onics','odr','opencolorio','owncloud','sagemath','usd','lua','lxc'])
    '''
    output=[0 in range(100000)]
    for i in tokens:
        output[tokens[i]]=i
    for i in range(len(tokens)):
        print(output[i],file=fi)

    fi.close()
'''
