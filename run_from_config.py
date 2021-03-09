import sys
import os
import config as c
import eval.eval_prova as eval
import torch

print("Running " + sys.argv[0])

#print(f"Using cuda {c.cuda}")
if torch.cuda.is_available():
    cuda_id = torch.cuda.current_device()
    print(f"Using gpu: {torch.cuda.get_device_name(cuda_id)}")
else:
    print("Using cpu")
    
data_dir = f"{c.task}/{c.domain}"
print(data_dir)

import torch
version = torch.__version__
print(f"using pytorch {version}")

for run in range(1, int(c.runs)+1):
    output_dir = f"run/{c.run_dir}/{c.domain}/{run}"
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        print(f"Directory {output_dir} already exists")
    except OSError:
        print(f"Creation of the directory {output_dir} failed")
    else:
        print(f"Directory {output_dir} created.")

    print(f"--- Run {run}/{c.runs} ---")

#--- train and validation ---
    if os.path.isfile(f"{output_dir}/valid.json"):
        print(f"file {output_dir}/valid.json exists! Skipped...")
    else:
        os.system(f'python src/run_{c.task}.py --bert_model {c.bert} --do_train --do_valid --max_seq_length 100 --train_batch_size 3 --learning_rate 3e-5 --num_train_epochs 3 --output_dir {output_dir} --data_dir {data_dir} --seed {run}')
		
#--- evaluation ---
    if os.path.isfile(f"{output_dir}/predictions.json"):
        print(f"file {output_dir}/predictions.json exists! Skipped...")
    else:
        os.system(f'python src/run_{c.task}.py --bert_model {c.bert} --do_eval --max_seq_length 100 --output_dir {output_dir} --data_dir {data_dir} --seed {run}')

    if (os.path.isfile(f"{output_dir}/predictions.json") and os.path.isfile(f"{output_dir}/model.pt")):
        os.remove(f"{output_dir}/model.pt")
    else:
        print()	

print("***** Done! *****")	
#--- result ---
if c.eval == "y":
    eval.evaluate(c.tasks, c.berts, c.domains, c.runs)
else:
    print("Please write 'cd ..' then 'python result.py' to see the results")