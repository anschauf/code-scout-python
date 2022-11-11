
# Performance testing tool for CodeScout (CodeScout Performance App)

## About The Project

The CodeScout performance testing tool was developed to be able to measure and compare various CodeScout approaches/prototypes. 
The tool aims to take suggestions from different prototypes and measures and to compare their results, thus offering support
in finding the best working solution for CodeScout. 
Currently, the performance measure tool consists of two separate apps:

-  code_ranking_tier
-  case_ranking

The corresponding .py files to run the apps can be found in the following folder: **codescout-performance-app &rarr; src &rarr; apps** 


The detailed description of the CodeScout apps can be found here [RFC 0011 - Performance testing tool for Code Scout](https://www.notion.so/aimedic/RFC-0011-Performance-testing-tool-for-Code-Scout-554e9d35b96845afa42c70f7fe8ccef2)


### Technology Stack

- [Python 3.10.5](https://www.python.org/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [PyCharm IDE](https://www.jetbrains.com/pycharm/promo/?source=google&medium=cpc&campaign=14123077402&term=pycharm)


## Getting Started

#### Prerequisites

Python 3, Docker Desktop and the PyCharm IDE must be installed locally => see Technology stack

#### Setup

1. Duplicate the `.env-example` file and name the file `.env`
2. Insert all env variables into the file. You find them in KeePass
   1. All `BFS_CASES_DB_*` are found under `aimedic/AWS/databases` 
   2. `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` is your personal AWS Key
   3. `AWS_REGION` is **eu-central-1**
   4. The most recent `AIMEDIC_GROUPER_VERSION` you find on [AWS CodeArtifact](https://eu-central-1.console.aws.amazon.com/codesuite/codeartifact/d/264427866130/aimedic/r/aimedic/p/maven/ch.aimedic/aimedic-grouper_2.12/versions?region=eu-central-1&package-versions-meta=eyJmIjp7fSwicyI6e30sIm4iOjIwLCJpIjowfQ)
3. From the run configs click `Add configuration`. Under `Shell script` you should find run configs for `dockerfile.sh`. Click **OK** ro add the configs.
4. Run the shell script by hitting the green arrow button.
5. In the terminal it should now start a Docker build. Wait until it has finished buidling.
6. Once finished you see a localhost URL in the output. Open it and a jupyter notebook opens in your browser
7. You are ready to code =)

> It is important that you alway call the shell-script `dockerfile.sh` and not the Dockerfile itself.
> 
> As there the environment variables are not loaded in upfront and the building of the Dockerfile will fail.

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





