
# Code Scout (python repo)

## About The Project

This is the python side of CodeScout. There is also a Scala project.

In here, there are the following components:

- The performance testing tool was developed to be able to measure and compare various CodeScout approaches/prototypes. 
The tool aims to take suggestions from different prototypes and measures and to compare their results, thus offering support
in finding the best working solution for CodeScout. 
Currently, the performance measure tool consists of two separate apps:

  -  code_ranking_tier
  -  case_ranking

The corresponding .py files to run the apps can be found in the following folder: **src &rarr; apps** 


### Technology Stack

- [Python 3.10](https://www.python.org/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [PyCharm IDE](https://www.jetbrains.com/pycharm/promo/?source=google&medium=cpc&campaign=14123077402&term=pycharm)

## Getting Started

### Prerequisites

Python 3, Docker Desktop and the PyCharm IDE must be installed locally => see Technology stack

### Setup

1. Duplicate the `.env-example` file and name the file `.env`
2. Insert all env variables into the file. You find them in KeePass
   1. All `BFS_CASES_DB_*` are found under `aimedic/AWS/databases` 
   2. `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` is your personal AWS Key
   3. `AWS_REGION` is **eu-central-1**
   4. The most recent `AIMEDIC_GROUPER_VERSION` you find on [AWS CodeArtifact](https://eu-central-1.console.aws.amazon.com/codesuite/codeartifact/d/264427866130/aimedic/r/aimedic/p/maven/ch.aimedic/aimedic-grouper_2.12/versions?region=eu-central-1&package-versions-meta=eyJmIjp7fSwicyI6e30sIm4iOjIwLCJpIjowfQ)

#### Specific setup for PyCharm 2022.2.4 on MacOs
These instructions may be needed for later versions, too.

To restore the old-looking interpreter window, one has to:
- Go to `Help` | `Find Action`, and type `registry`
  - Type `python.use.targets.api`, and untick the checkbox. 
- Now you can follow the instructions below. 

#### General setup for PyCharm
These instructions have been only tested on MacOS. They may need to be adapted for other OSs.

- Open the file `dockerfile.sh`, and click on the green arrow at the top of the file.
  - Wait for all the steps to be completed. Make sure that there are no errors.
- Go to `PyCharm` | `Preferences`.
  - In the sidebar, select `Project: code-scout-python`, then `Python Interpreter`.
  - On the right-hand side, click on the clockwork icon, then `Add...`.
  - In the new window, select `Docker` in the sidebar.
  - On the right-hand side, make sure to have selected `code-scout-python:latest` as `Image name`, and `python` as 
    `Python interpreter path`.
- Click on OK until you're back to the main window of PyCharm.
- Wait for all the background processes to finish.


### Input and Output

The case ranking and code ranking both require the same input files. 

- The DtoD revision data (currently named: <code>CodeScout_GroundTruthforPerformanceMeasuring.csv</code>) containing the following variables:
    - <code> CaseId </code>, <code> CW_new </code>, <code> CW_old </code>, <code>ICD_added</code>
- The code scout results are stored in a folder with an individual path carrying a name based on '<code>hospital_year</code>'. The results are stored
in a <code>.csv</code> containing the following variables: 
  - <code>CaseId</code>, <code>SuggestedCodeRankings</code>, <code>UpcodingConfidenceScore</code> 


The resources are currently saved under <code>performance-measuring</code> and get committed to bitbucket, 
which means that the developer does not need to set it up additionally. The exact location in S3 may change due 
to restructuring. 


## Project structure

    .
    ├── resources                   
    ├── src                      
      ├── apps                     # runnalbe applications
        ├── case_ranking_tier.py  
        ├── case_ranking_tier.py    
      ├── files.py                  
      ├── rankings.py
      ├── schema.py                  
      ├── utils.py
      ├── venn.py
    ├── test                 
    ├── Dockerfile
    └── README.md

   
## Maintainers

- Lirui Zhang
- Michael Kunz 





