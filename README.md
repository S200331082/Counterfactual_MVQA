# Counterfactual Med_VQA

1. Use ```lib/utils/run.sh``` for creating dictionaries, ```img2idx.json```, resized images e.g. ```images64x64.pkl``` and other input files. For the glove embeddings, you can download the data from [here](https://1drv.ms/u/s!ApXgPqe9kykTgxHHWkVTHvyD7cg2?e=X9Zlf1).
2. Move data to ```./data/data_rad``` and ```./data/data_slake``` for RAD and SLAKE datasets, respectively.
3. Set the ```CLIP_PATH```, i.e. path of the fine-tuned PubMedCLIP in your config file of interest in ```configs```.
4. Run the experiments for original CLIP and PubMedCLIP via ```xxx.sh``` , respectively.


The key part code is under the ``causal_effect_intervention`` folder, you can embed it into your network
