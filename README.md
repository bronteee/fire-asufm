# fire-asufm
This is the official repository for the paper Wildfire Spread Prediction in North America Using Satellite Imagery and Vision Transformer.

Project website: https://bronteee.github.io/

## Installation

To create a python environment with all the dependencies, run the following command:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Data and Pre-trained Models

The original Next Day Wildfire Spread dataset can be downloaded from [here](https://www.kaggle.com/fantineh/next-day-wildfire-spread). 

The extended 2012-2023 dataset can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/bronteli/next-day-wildfire-spread-north-america-2012-2023).

The pre-trained models can be downloaded from [here](https://www.kaggle.com/models/bronteli/attention-swin-u-net-with-focal-modulation-asufm).

## Training and Evaluation

To train the model, run the following command:
```
python main.py --seed <seed> --dir_checkpoint <checkpoint directory> --epochs <number of epochs> --batch_size <batch size> 
```
To evaluate the model, run the following command:
```
python evaluate.py --seed <seed> --load_model <path to model checkpoint>
```

## References and Acknowledgements
Our code is based on the following repositories, we thank the authors for their excellent contributions.

[Attention Swin U-net](https://github.com/NITR098/AttSwinUNet)

[Next Day Wildfire Spread](https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread)

[Focal Modulation Networks](https://github.com/microsoft/FocalNet)

[Face Parsing](https://github.com/Jo-dsa/SemanticSeg/tree/master)

## Citation

B. Li and R. Rad, "Wildfire Spread Prediction in North America Using Satellite Imagery and Vision Transformer," in 2024 IEEE Conference on Artificial Intelligence (CAI), Singapore, Singapore, 2024 pp. 1536-1541.
doi: 10.1109/CAI59869.2024.00278

keywords: {climate change;wildfires;satellite images;visualization;predictive models;environmental monitoring;remote sensing;deep learning;risk management}

Abstract: The recent surge in wildfire incidents, exacerbated by climate change, demands effective prediction and monitoring solutions. This study introduces a novel deep learning model, the Attention Swin U-net with Focal Modulation (ASUFM), designed to predict wildfire spread in North America using large-scale remote sensing data. The ASUFM model integrates spatial attention and focal modulation into the Swin U-net architecture, significantly enhancing predictive performance on the Next Day Wildfire Spread (NDWS) benchmark. Additionally, we have developed an expanded dataset that encompasses wildfire data across North America from 2012 to 2023, providing a more extensive basis for model training and validation. Our approach demonstrates state-of-the-art performance in wildfire spread prediction, offering a promising tool for risk mitigation and resource allocation in wildfire management. Related code is in our public repository https://github.com/bronteee/fire-asufm
url: https://doi.ieeecomputersociety.org/10.1109/CAI59869.2024.00278
