## Generalizable Bus Travel Times
#### Bus Travel Time Prediction with GTFS and GTFS-RT Data

### Introduction
This repository contains scripts and files used for a project which aims to predict bus travel times using General Transit Feed Specification (GTFS) and GTFS-Realtime (GTFS-RT) data collected from King County Metro in Seattle and AtB in Trondheim.

The paper focuses on developing machine learning models that can predict bus travel times more accurately using a combination of timetable, geographic, and real-time data. It involves the collection, processing, and analysis of transit data from the two agencies, and building models based on that data.

The source code files in the /src/ folder include scripts for scraping the data, and Jupyter notebooks for data processing, feature engineering, and model prototyping. The code is written in Python. Dependencies can be installed using the conda environment file. The scraping scripts require additional setup e.g., a linked AWS account.

### Installation
1. Clone Github repository
2. Create new conda environment from .yaml file
3. Activate environment
4. Run scripts from /src/ directory
