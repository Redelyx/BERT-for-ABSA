import json
import os
import sklearn.metrics
import numpy as np
from pprint import pprint
import argparse
import time
import math
import random
import xml.etree.ElementTree as ET
from subprocess import check_output

def label_rest_xml(fn, output_fn, corpus, label):
    dom=ET.parse(fn)
    root=dom.getroot()
    pred_y=[]
    for zx, sent in enumerate(root.iter("sentence") ) :
        tokens=corpus[zx]
        lb=label[zx]
        opins=ET.Element("Opinions")
        token_idx, pt, tag_on=0, 0, False
        start, end=-1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx<len(tokens) and pt>=len(tokens[token_idx] ):
                pt=0
                token_idx+=1

            if token_idx<len(tokens) and lb[token_idx]==1 and pt==0 and c!=' ':
                if tag_on:
                    end=ix
                    tag_on=False
                    opin=ET.Element("Opinion")
                    opin.attrib['target']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and lb[token_idx]==2 and pt==0 and c!=' ' and not tag_on:
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and (lb[token_idx]==0 or lb[token_idx]==1) and tag_on and pt==0:
                end=ix
                tag_on=False 
                opin=ET.Element("Opinion")
                opin.attrib['target']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            elif token_idx>=len(tokens) and tag_on:
                end=ix
                tag_on=False 
                opin=ET.Element("Opinion")
                opin.attrib['target']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            if c==' ':
                pass
            elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                pt+=2
            else:
                pt+=1
        if tag_on:
            tag_on=False
            end=len(sent.find('text').text)
            opin=ET.Element("Opinion")
            opin.attrib['target']=sent.find('text').text[start:end]
            opin.attrib['from']=str(start)
            opin.attrib['to']=str(end)
            opins.append(opin)
        sent.append(opins )
    dom.write(output_fn)

def label_laptop_xml(fn, output_fn, corpus, label):
    dom=ET.parse(fn)
    root=dom.getroot()
    pred_y=[]
    for zx, sent in enumerate(root.iter("sentence") ):
        tokens=corpus[zx]
        lb=label[zx]
        opins=ET.Element("aspectTerms")
        token_idx, pt, tag_on=0, 0, False
        start, end=-1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx<len(tokens) and pt>=len(tokens[token_idx] ):
                pt=0
                token_idx+=1

            if token_idx<len(tokens) and lb[token_idx]==1 and pt==0 and c!=' ':
                if tag_on:
                    end=ix
                    tag_on=False
                    opin=ET.Element("aspectTerm")
                    opin.attrib['term']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and lb[token_idx]==2 and pt==0 and c!=' ' and not tag_on:
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and (lb[token_idx]==0 or lb[token_idx]==1) and tag_on and pt==0:
                end=ix
                tag_on=False 
                opin=ET.Element("aspectTerm")
                opin.attrib['term']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            elif token_idx>=len(tokens) and tag_on:
                end=ix
                tag_on=False 
                opin=ET.Element("aspectTerm")
                opin.attrib['term']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            if c==' ' or ord(c)==160:
                pass
            elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                pt+=2
            else:
                pt+=1
        if tag_on:
            tag_on=False
            end=len(sent.find('text').text)
            opin=ET.Element("aspectTerm")
            opin.attrib['term']=sent.find('text').text[start:end]
            opin.attrib['from']=str(start)
            opin.attrib['to']=str(end)
            opins.append(opin)
        sent.append(opins )
    dom.write(output_fn)

def evaluate_ae_main(pred_json):    

    if 'rest' in pred_json:
        command="java -cp eval/A.jar absa16.Do Eval -prd ae/official_data/rest_pred.xml -gld ae/official_data/EN_REST_SB1_TEST.xml.gold -evs 2 -phs A -sbt SB1"
        template="ae/official_data/EN_REST_SB1_TEST.xml.A"
    elif 'laptop' in pred_json:
        command="java -cp eval/eval.jar Main.Aspects ae/official_data/laptop_pred.xml ae/official_data/Laptops_Test_Gold.xml"
        template="ae/official_data/Laptops_Test_Data_PhaseA.xml"

    with open(pred_json) as f:
        pred_json1=json.load(f)    
    y_pred=[]
    for ix, logit in enumerate(pred_json1["logits"]):
        pred=[0]*len(pred_json1["raw_X"][ix])
        for jx, idx in enumerate(pred_json1["idx_map"][ix]):
            lb=np.argmax(logit[jx])
            if lb==1: #B
                pred[idx]=1
            elif lb==2: #I
                if pred[idx]==0: #only when O->I (I->I and B->I ignored)
                    pred[idx]=2
        y_pred.append(pred)
    
    if 'REST' in command:
        command=command.split()
        label_rest_xml(template, command[6], pred_json1["raw_X"], y_pred)
        acc=check_output(command ).split()
        return float(acc[9][10:])
    elif 'Laptops' in command:
        command=command.split()
        label_laptop_xml(template, command[4], pred_json1["raw_X"], y_pred)
        acc=check_output(command ).split()
        return float(acc[15])

def evaluate(tasks, berts, domains, runs=10):
    for task in tasks:
        for bert in berts:
            for domain in domains:            
                scores=[]
                for run in range(1, runs+1):
                    DATA_DIR=os.path.join(task, domain)
                    OUTPUT_DIR=os.path.join("run", bert+"_"+task, domain, str(run) )
                    if os.path.exists(os.path.join(OUTPUT_DIR, "predictions.json") ):
                        if "rrc" in task:
                            ret = os.system('python evaluate-v1.1.py $DATA_DIR/test.json $OUTPUT_DIR/predictions.json')
                            score=json.loads(ret[0])
                            scores.append([score["exact_match"], score["f1"] ] )
                        elif task == "ae":
                            PATH = os.path.join(OUTPUT_DIR, "predictions.json")
                            ret = evaluate_ae_main(PATH)
                            scores.append(float(ret)*100 )
                        elif "asc" in task:
                            with open(os.path.join(OUTPUT_DIR, "predictions.json") ) as f:
                                results=json.load(f)
                            y_true=results['label_ids']
                            y_pred=[np.argmax(logit) for logit in results['logits'] ]
                            p_macro, r_macro, f_macro, _=sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
                            f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
                            scores.append([100*sklearn.metrics.accuracy_score(y_true, y_pred), 100*f_macro ] )
                        else:
                            raise Exception("unknown task")
                scores=np.array(scores)
                m=scores.mean(axis=0)
                
                if len(scores.shape)>1:
                    for iz, score in enumerate(m):
                        print(task, ":", bert, domain, "metric", iz, round(score, 2))
                        pprint(scores)
                else:
                    print(task, ":", bert, domain, ", accuracy:", round(m,2))
                    pprint(scores)
                print

#if __name__ == "__main__":    
    
 #   evaluate(tasks, berts, domains, runs)