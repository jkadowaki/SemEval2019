#  Temporal Ensembles of Fined-Tuned BERT Models for Offensive Language Identification
### by Jennifer Kadowaki

-

### Task: Identifying Offensive and Abusive Language in Tweets 
*With the advent of major social media platforms, growing concerns surround online user safety and experience. We participated in the SemEval-2019’s OffensEval shared task, which uses the Offensive Language Identification Dataset (OLID; Zampieri et al., 2019) to identify offensive and abusive language in Tweets.*

**Annotated Examples of Offensive or Abusive Language**:

    1. @USER @USER The prison system is so fucked.  Why are they still getting away with what is potentially murder with intent if the prisoners die in the hurricane? They did this in Louisiana and like 500 inmates went missing
    
    2. @USER @USER @USER @USER LOL!!!   Throwing the BULLSHIT Flag on such nonsense!!  #PutUpOrShutUp   #Kavanaugh   #MAGA   #CallTheVoteAlready URL
    
    3. The only thing the Democrats have is lying and stalling to stop Trump from being #President.  What have they done for you lately. #Trump #Kavanaugh #MAGA #DEMSUCK
    
**Annotated Examples of Non-offensive Language**:

    1. @USER @USER She is useless.  Attempts to do the right thing but never follows through.
    
    2. PLEASE vote NO on Kavanaugh. He is not fit for SCOTUS and allegations about women and shady financials should disqualify him.    #RuleOfLaw matters. #MeToo #CountryOverParty #WithdrawKavanaugh  #StopKavanaugh
    
    3. @USER @USER @USER @USER did Twitter silence alex jones in retaliation of him asking Twitter jack questions  @USER @USER @USER @USER URL 

-

## Fine-tuning BERT Models

### Using Ocelote High Performance Computing Cluster

To log into Ocelote: ```ssh username@hpc.arizona.edu```

You will be asked for a password and two-factor authentication. (Follow these [instructions](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-on-ubuntu-1604) to bypass this step on future logins.) Next, specify the cluster: ```ocelote```

Your ```$HOME``` and PBS directories are limited to 15GB in total. Use ```/extra/username``` or ```/xdisk/username``` storage to prevent disk space errors. 

```
USER=$(whoami)
cd /extra/$USER
```

### Download Data & Scripts

```
git clone https://github.com/jkadowaki/SemEval2019.git
cd SemEval2019
```

### Preprocess Data
We have included the preprocessed the data files in the ```/eval_data``` directory on the [GitHub repository](https://github.com/jkadowaki/SemEval2019). However, if you would like to preprocess the data on a local machine before scp-ing the data, ensure you have the directory ```OLIDv1.0``` containing the [Offensive Language Identification Dataset](https://competitions.codalab.org/competitions/20011#participate) and run:

```
python code/prepare_data.py
```

To secure copy the data and scripts to the HPC from your local machine:

```
scp -r local_directory username@filexfer.hpc.arizona.edu:/extra/username/SemEval2019/
```

### Download Singularity Images
The [UA-hpc-containers](https://www.singularity-hub.org/collections/2226) Singularity Hub contains a repository of Singularity images and recipes for the [AutoMATES team](https://ml4ai.github.io/automates/).

You can also easily download a number of [Singularity containers](https://public.confluence.arizona.edu/display/UAHPC/GPU+Nodes#GPUNodes-NVIDIAGPUCloudContainerRegistry) built by the HPC team.


* The Singularity image for the model was built to run CUDA and Tensorflow on NVIDIA GPUs in [Ocelote](https://docs.hpc.arizona.edu/display/UAHPC/Ocelote+Quick+Start). We'll be running this model directly from `/unsupported/singularity/nvidia/nvidia-tensorflow.18.09-py3.simg`.


* If you are using PyTorch's implementation of BERT, the container can be found here: `/unsupported/singularity/nvidia/nvidia-pytorch.18.09-py3.simg`.

_Further Reading_: [Quick Start Guide to Singularity](https://singularity.lbl.gov/quickstart#download-pre-built-images)


### Download the Trained BERT Model
To download the trained model to the HPC:

```
cd /extra/$USER/SemEval2019/
git clone https://github.com/google-research/bert.git
```

To download the cased, BERT-base model:

```
wget -P bert/ https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
```


### Portable Batch System (PBS) Scripts
The PBS scripts are not required to run from your home directory.  

 

Ensure to change the e-mail address ```username@email.arizona.edu``` in the script to receive updates when the job terminates or is aborted.

### Submitting a Job on the HPC
From the directory containing ```script.pbs```, submit the job to Ocelote's PSB scheduler:

```
qsub script.pbs
```

Once the job has been submitted to the scheduler, you can check the status of your jobs via

```
qstat -u $USER
```

You can also peek at the progress of your script with

```
qpeek <jobid>
```

or interactively with [Open OnDemand](https://ood.hpc.arizona.edu/pun/sys/dashboard/apps/show/activejobs). The total compute time for testing is <2 minutes with exclusive access to 28 cores on 1 GPU on Ocelote.

-
## Predicting with BERT

### Scripts & Data
```
```

### Portable Batch System (PBS) Scripts
```
qsub predict_script.pbs
```
-
## Helpful HPC Debugging Strategies:
We fine-tune BERT for 100 epochs for every fold. After every training epoch, we save a Tensorflow model checkpoint, which is used to create predictions. As a result, we generate nearly ~150 GB of BERT models at the end of the 100 epochs. To ensure you are below the required disk space requirements, check your memory usage:

```
uquota
```

Occasionally, if your run crashes, the HPC will fail to deliver your output and error files associated with the job you submitted. This typically occurs if your job has an infinite loop which generates output text that exceeds 15 GB (the disk storage allocation for your home directory). These failed deliveries reside in the ```/tmp``` directory and will count against your disk storage allocation, preventing you from storing more data or submitting more jobs. To search for these files:

```
module load unsupported
module load ferng/find-pbs-files
pbs-files $USER
```
You will need to manually delete these files.


Lastly, when debugging, it maybe helpful to run the code interactively:

```
qsub -I script.pbs
```
Once you secure an interactive node, you will be kicked off the interactive node if the system detects more than 15 minutes of inactivity.
