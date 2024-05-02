# PyTorch Implementation of Traffexplainer: A Framework towards GNN-based
Interpretable Traffic Prediction

Official PyTorch implementation for Traffexplainer: A Framework towards GNN-based
Interpretable Traffic Prediction.

## Requirements

- Python >= 3.8 (3.9 recommended)
- fastdtw==0.3.4
- gensim==4.3.2
- geopy==2.4.1
- hyperopt==0.2.7
- matplotlib==3.8.2
- networkx==3.2.1
- nltk==3.8.1
- numpy==1.26.4
- pandas==2.2.2
- progressbar33==2.4
- ray==2.20.0
- scikit_learn==1.3.2
- scipy==1.13.0
- statsmodels==0.14.2
- torch==2.1.2
- torchdiffeq==0.2.3
- torchtext==0.18.0
- tqdm==4.66.1

## Folder Structure

  ```
CauseFormer/
|-- README.md
|-- config
|-- data - original data from Amap API (deprecated)
|-- img - output images from notebooks
|-- lib
    |-- Bigscity-LibCity - LibCity: An Open Library for Urban Spatial-temporal Data Mining.
        |-- LICENSE.txt
        |-- cmd-batch.sh - batch script of cmd-explainer.sh
        |-- cmd-explainer.sh - script for interpret the trained prediction model
        |-- cmd-model.sh - script for training the GNN-based traffic prediction models
        |-- contribution_list.md
        |-- hyper_tune.py
        |-- libcity
        |-- raw_data
        |   |-- PEMSD4
        |   |-- PEMSD8
        |   `-- TGESHD - (SH_traffic) download from https://ieee-dataport.org/documents/shtraffic
        |-- readme.md
        |-- readme_zh.md
        |-- requirements.txt
        |-- run_hyper.py
        |-- run_model.py
        |-- test
        |-- test_model.py
        |-- unit_test.py
        |-- visualize.py
        |-- visualized_data
|-- requirements.txt
`-- src
    |-- POI_crawler.ipynb - construct SH_traffic data (deprecated)
    |-- compare_mask.ipynb - Interpretation Validation
    |-- data_atomizer.ipynb - build SH_traffic data (deprecated)
    |-- data_clean.ipynb - clean original data (deprecated)
    |-- data_crawler.ipynb - collect original data (deprecated)
    |-- data_visualization.ipynb - analysis original data (deprecated)
    |-- draw_util.py
    |-- graph_generator.ipynb - construct original data (deprecated)
    |-- main-resemble-analysis.ipynb - analysis original data (deprecated)
    |-- main-topology-graph.ipynb - construct original data (deprecated)
    |-- results_table.ipynb - Prediction Results
    |-- show_mask.ipynb - Analysis of Learned Masks
    `-- xml_parser.ipynb - construct original data (deprecated)
  ```

## Dataset

- PeMSD4&PeMSD8: [Caltrans Performance Measurement System (*PeMS*)](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source)
- SH_traffic: [SH_traffic](https://ieee-dataport.org/documents/shtraffic) **Note: `TGESHD` in the code all refer to `SH_traffic`**

## Usage

1. Use `lib/Bigscity-LibCity/cmd-model.sh` to train a GNN-based traffic prediction model

2. Add the meta data of the trained GNN in to `lib/Bigscity-LibCity/libcity/model/traffic_speed_prediction/Explainer.py` . For example:

   ```python
   cache_name_dict = {
       'mySTGCN':'./libcity/cache/96118/model_cache/mySTGCN_TGESHD.m',
       'HGCN':'./libcity/cache/44711/model_cache/HGCN_METR_LA.m',
       'TGCN':'./libcity/cache/81856/model_cache/TGCN_TGESHD.m',
       'DCRNN':'./libcity/cache/47320/model_cache/DCRNN_TGESHD.m',
       'MTGNN':'./libcity/cache/78181/model_cache/MTGNN_TGESHD.m'
   }
   ```

3. Use `lib/Bigscity-LibCity/cmd-explainer.sh` to generate explanation for the trained models.

4. Use notebooks for analysis.

## License

This project is licensed under the  GPL-3.0 License. See LICENSE for more details
This GNN training framework is based on [LibCity: An Open Library for Urban Spatial-temporal Data Mining](https://github.com/LibCity/Bigscity-LibCity).
