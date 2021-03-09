# Adversarial Training for Aspect-Based Sentiment Analysis with BERT
Code for "[Adversarial Training for Aspect-Based Sentiment Analysis with BERT](https://arxiv.org/pdf/2001.11316)".

We have used the codebase from the following paper and improved upon their results by applying adversarial training.
"[BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://www.aclweb.org/anthology/N19-1242.pdf)".

## ABSA Tasks
We focus on two major tasks in Aspect-Based Sentiment Analysis (ABSA).

Aspect Extraction (AE): given a review sentence ("The retina display is great."), find aspects("retina display");

Aspect Sentiment Classification (ASC): given an aspect ("retina display") and a review sentence ("The retina display is great."), detect the polarity of that aspect (positive).

## Running (see below to run all the code in python)

Place laptop and restaurant post-trained BERTs into ```pt_model/laptop_pt``` and ```pt_model/rest_pt```, respectively. The post-trained Laptop weights can be download [here](https://drive.google.com/file/d/1io-_zVW3sE6AbKgHZND4Snwh-wi32L4K/view?usp=sharing) and restaurant [here](https://drive.google.com/file/d/1TYk7zOoVEO8Isa6iP0cNtdDFAUlpnTyz/view?usp=sharing).

Execute the following command to run the model for Aspect Extraction task:

```bash run_absa.sh ae laptop_pt laptop pt_ae 9 0```

Here, ```laptop_pt``` is the post-trained weights for laptop, ```laptop``` is the domain, ```pt_ae``` is the fine-tuned folder in ```run/```, ```9``` means run 9 times and ```0``` means use gpu-0.

Similarly,
```
bash run_absa.sh ae rest_pt rest pt_ae 9 0
bash run_absa.sh asc laptop_pt laptop pt_asc 9 0
bash run_absa.sh asc rest_pt rest pt_asc 9 0
```
### Evaluation
Evaluation wrapper code has been written in ipython notebook ```eval/eval.ipynb```. 
AE ```eval/evaluate_ae.py``` additionally needs Java JRE/JDK to be installed.

Open ```result.ipynb``` and check the results.


## Run in Python (iPython Notebook not needed)

After  placing  laptop and restaurant post-trained BERTs into the right folders, edit ```src/modelconfig.py```:
edit  ```../pt_model/laptop_pt/``` to ```pt_model/laptop_pt/``` and ```../pt_model/rest_pt/``` to ```pt_model/rest_pt/```.

Now you can start the training with ```run_from_config.py```. You don't need to add any argument.
To edit your preferences open ```config.py``` and insert your values.
If the value "eval" is set to "y", the evaluation will start after the training is over. Otherwise you can run the evaluation running ```result.py```.

## Citation

```
@misc{karimi2020adversarial,
    title={Adversarial Training for Aspect-Based Sentiment Analysis with BERT},
    author={Akbar Karimi and Leonardo Rossi and Andrea Prati and Katharina Full},
    year={2020},
    eprint={2001.11316},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
