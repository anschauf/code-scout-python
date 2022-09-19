
<h1>Performance testing tool for CodeScout (CodeScout Performance App) </h1>

<h2>About The Project </h2>

The CodeScout performance testing tool was developed to be able to measure and compare various CodeScout approaches/prototypes. 
The tool aims to take suggestions from different prototypes and measures and to compare their results, thus offering support
in finding the best working solution for CodeScout. 
<p>
Currently, the performance measure tool consists of two seperate apps, which will be explained in further detail later on: 

<li>Code Ranking tiers</li>
<li>Case ranking tiers</li>

</p>

<h4>Technology Stack </h4>

Python 3.10.5, Docker Desktop, PyCharm IDE


<h2>Getting Started </h2>

<h4>Prerequisites</h4>

Python 3, Docker Desktop and the PyCharm IDE must be installed locally => see Technology stack

<h4>Installation (TO BE VERIFIED!) </h4>

1. Open the project folder in PyCharm 
2. In the run configurations at the top right, you need to launch the build of the docker container.
3. After the container is built successfully you need to set the python interpreter for PyCharm.
4. Go to the interpreter settings in PyCharm.
5. Click the gear icon and select add.
6. Select the image which you created in the previous step. Default: Remote Python 3.10.5 Docker (codescout-performance-app)

<h4>Setting up the Docker Environment and interpreter in the PyCharm IDE</h4>

For the apps to run, it is to be ensured that the docker environment is set up as follows in the PyCharm IDE 
<b>(Run > Edit configurations...)</b>:


<img src="resources/images/readme/run_debug_config_docker.png"/>

To run the case ranking and code ranking tier apps, the following configurations are needed in the PyCharm IDE 
(here shown in the case ranking tier) <b>(Run > Edit configurations...)</b>:

<img src="resources/images/readme/config_case_ranking.png"/>

The script path as well as the working directory are set up according to the individual local space of the user.
Another thing to ensure is that the <b>AWS Connection</b> is set to "Use the currently selected credential profile/region" (also in <b>(Run > Edit configurations...)</b>) is set up as follows:

<img src="resources/images/readme/aws_config.png"/>

Further, the AWS Connection setting (bottom right in PyCharm IDE) are to be set up as follows:

<img height="201" src="resources/images/readme/aws_connection_settings.png" width="206"/>


<h3>Input and Output</h3>

<li> case ranking tier: </li>
Data sets with the following: CaseId, CW_new, CWold....

<li> code ranking tier: </li>
Data sets with the following:.....









<p>Libraries:</p>
<li>awswrangler</li>
<li>Pandas, Numpy, Matplotlib</li>
<li>loguru</li>
<li>For the Venn Diagramm: pyvenn, Repository: https://github.com/tctianchi/pyvenn </li> 





