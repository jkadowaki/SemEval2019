#  Temporal Ensembles of Fined-Tuned BERT Models for Offensive Language Identification
## by Jennifer Kadowaki

---

## Task: Identifying Offensive and Abusive Language in Tweets 
*With the advent of major social media platforms, growing concerns surround online user safety and experience. We participated in the SemEval-2019’s OffensEval shared task, which uses the Offensive Language Identification Dataset (OLID; Zampieri et al., 2019) to identify offensive and abusive language in Tweets.*

**Annotated Examples of Offensive or Abusive Language**:
    1. @USER @USER The prison system is so fucked.  Why are they still getting away with what is potentially murder with intent if the prisoners die in the hurricane? They did this in Louisiana and like 500 inmates went missing
    2. @USER @USER @USER @USER LOL!!!   Throwing the BULLSHIT Flag on such nonsense!!  #PutUpOrShutUp   #Kavanaugh   #MAGA   #CallTheVoteAlready URL
    3. The only thing the Democrats have is lying and stalling to stop Trump from being #President.  What have they done for you lately. #Trump #Kavanaugh #MAGA #DEMSUCK
    
**Annotated Examples of Non-offensive Language**:
    1. @USER @USER She is useless.  Attempts to do the right thing but never follows through.
    2. PLEASE vote NO on Kavanaugh. He is not fit for SCOTUS and allegations about women and shady financials should disqualify him.    #RuleOfLaw matters. #MeToo #CountryOverParty #WithdrawKavanaugh  #StopKavanaugh
    3. @USER @USER @USER @USER did Twitter silence alex jones in retaliation of him asking Twitter jack questions  @USER @USER @USER @USER URL
    
---

## Replicate the Model Architecture
### Use Ocelote High Performance Computing Cluster


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

# TBD!
