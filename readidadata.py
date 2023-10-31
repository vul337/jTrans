import pickle
import networkx
import time
def parse_operand(operator,location,operand1):
    operand1=operand1.strip(' ')
    operand1=operand1.replace('ptr ','')
    operand1=operand1.replace('offset ','')
    operand1=operand1.replace('xmmword ','')
    operand1=operand1.replace('dword ','')
    operand1=operand1.replace('qword ','')
    operand1=operand1.replace('word ','')
    operand1=operand1.replace('byte ','')
    operand1=operand1.replace('short ','')
    operand1=operand1.replace('-','+')

    if operand1[0:3]=='cs:' :
        operand1='cs:xxx'
        return operand1
    if operand1[0:3]=='ss:' :
        operand1='ss:xxx'
        return operand1
    if operand1[0:3]=='fs:' :
        operand1='fs:xxx'
        return operand1
    if operand1[0:3]=='ds:' :
        operand1='ds:xxx'
        return operand1
    if operand1[0:3]=='es:' :
        operand1='es:xxx'
        return operand1
    if operand1[0:3]=='gs:' :
        operand1='gs:xxx'
        return operand1
    if operator[0]=='j' and not isregister(operand1):
        if operand1[0:4]=='loc_' or operand1[0:7]=='locret_' or operand1[0:4]=='sub_' :
            operand1='hex_'+operand1[operand1.find('_')+1:]
            return operand1
        else:
            #print("JUMP ",operand1)
            operand1='UNK_ADDR'
            return operand1

    if operand1[0:4]=='loc_' :
        operand1='loc_xxx'
        return operand1
    if operand1[0:4]=='off_' :
        operand1='off_xxx'
        return operand1
    if operand1[0:4]=='unk_' :
        operand1='unk_xxx'
        return operand1
    if operand1[0:6]=='locret' :
        operand1='locretxxx'
        return operand1
    if operand1[0:4]=='sub_' :
        operand1='sub_xxx'
        return operand1
    if operand1[0:4]=='arg_' :
        operand1='arg_xxx'
        return operand1
    if operand1[0:4]=='def_' :
        operand1='def_xxx'
        return operand1
    if operand1[0:4]=='var_' :
        operand1='var_xxx'
        return operand1
    if operand1[0]=='(' and operand1[-1]==')':
        operand1='CONST'
        return operand1
    if operator=='lea' and location==2:
        if not ishexnumber(operand1) and not isaddr(operand1):  #handle some address constants
            operand1='GLOBAL_VAR'
            return operand1

    if operator=='call' and location==1:
        if len(operand1)>3:
            operand1='callfunc_xxx'
            return operand1

    if operator=='extrn':
        operand1='extrn_xxx'
        return operand1
    if ishexnumber(operand1):
        operand1='CONST'
        return operand1
    elif ispurenumber(operand1):
        operand1='CONST'
        return operand1
    if isaddr(operand1):
        params=operand1[1:-1].split('+')
        for i in range(len(params)):
            if ishexnumber(params[i]):
                params[i]='CONST'
            elif ispurenumber(params[i]):
                params[i]='CONST'
            elif params[i][0:4]=='var_':
                params[i]='var_xxx'
            elif params[i][0:4]=='arg_':
                params[i]='arg_xxx'
            elif not isregister(params[i]):
                if params[i].find('*')==-1:
                    params[i]='CONST_VAR'
        s1='+'
        operand1='['+s1.join(params)+']'
        return operand1

    if not isregister(operand1) and len(operand1)>4:
        operand1='CONST'
        return operand1
    return operand1
def parse_asm(code):   #handle ida code to better quality code for NLP model
    annotation=None
    operator,operand=None,None
    operand1,operand2,operand3=None,None,None
    if code.find(';')!=-1:
        id=code.find(';')
        annotation=code[id+1:]
        code=code[0:id]
    if code.find(' ')!=-1:
        id=code.find(' ')
        operand=code[id+1:]
        operator=code[0:id]
    else:
        operator=code
    if operand!=None:
        if operand.find(',')!=-1:
            strs=operand.split(',')
            if len(strs)==2:
                operand1,operand2=strs[0],strs[1]
            else:
                operand1,operand2,operand3=strs[0],strs[1],strs[2]
        else:
            operand1=operand
            operand2=None
    if operand1!=None:
        operand1=parse_operand(operator,1,operand1)
    if operand2!=None:
        operand2=parse_operand(operator,2,operand2)
    if operand3!=None:
        operand3=parse_operand(operator,3,operand3)
    return operator,operand1,operand2,operand3,annotation
def isregister(x):
    registers=['rax','rbx','rcx','rdx','esi','edi','rbp','rsp','r8','r9','r10','r11','r12','r13','r14','r15']
    return x in registers
def ispurenumber(number):
    if len(number)==1 and str.isdigit(number):
        return True
    return False
def isaddr(number):
    return number[0]=='[' and number[-1]==']'
def ishexnumber(number):
    if number[-1]=='h':
        for i in range(len(number)-1):
            if str.isdigit(number[i]) or (number[i] >='A' and number[i]<='F'):
                continue
            else:
                return False
    else:
        return False
    return True

