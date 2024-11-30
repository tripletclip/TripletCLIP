# TripletCLIP

## Install
We advise you first create a virtual environment with:
```
conda create -n tripletclip python=3.12
conda activate tripletclip
```

Then, install the required dependencies from the requirements.txt file:
```
pip install -r requirements.txt
```

# Training TripletCLIP
To train the model with the configuration from the paper, you can run the following command:

```
python src/main.py
 --model_name 'ViT-B-32' \
 --lr 0.00005 \
 --data_dir '/path/to/data/tar/files' \
 --epochs 30 \
 --train
```

# Results
The results reported are from the models trained on 1M image-text pairs from high quality TripletCLIP data. The results reported are after training for 10 epochs with a batch size of 1024 in bf16 precision.
<table>
  <tr>
    <th rowspan="2" style="text-align: center; vertical-align: middle;">Model</th>
    <!-- <th colspan="2" style="text-align: center; vertical-align: middle;">Winoground</th> -->
    <th colspan="7" style="text-align: center; vertical-align: middle;">SugarCrepe</th>
    <th colspan="2" style="text-align: center; vertical-align: middle;">MSCOCO</th>
    <th colspan="2" style="text-align: center; vertical-align: middle;">Flickr30k</th>
    <th colspan="1" style="text-align: center; vertical-align: middle;">Imagenet1k</th>
  </tr>
  <tr>
    <th>Add Att</th><th>Add Obj</th><th>Replace Att</th>
    <th>Replace Obj</th><th>Replace Rel</th><th>Swap Att</th><th>Swap Obj</th>
    <th>i2t R@5</th><th>t2i R@5</th><th>i2t R@5</th><th>t2i R@5</th><th>acc5</th>
  </tr>
  <tr>
    <td>CLIP</td><td>54.77</td><td>56.93</td>
    <td>62.81</td><td>61.22</td><td>56.33</td><td>53.6</td><td>50.61</td>
    <td>5.3</td><td>4.8</td><td>10.8</td><td>9.3</td><td>9.3</td>
  </tr>
  <tr>
    <td>NegCLIP</td><td>64.88</td><td>57.17</td>
    <td>70.55</td><td>65.92</td><td>68.136</td><td>56.756</td><td>57.142</td>
    <td>3.5</td><td>3.7</td><td>9.3</td><td>7.2</td><td>6.52</td>
  </tr>
  <tr>
    <td>TripletCLIP</td><td><b>67.196</b></td><td><b>64.646</b></td>
    <td><b>75.127</b></td><td><b>73.184</b></td><td><b>70.55</b></td><td><b>60.06</b></td><td><b>60.408</b></td>
    <td><b>13.04</b></td><td><b>13.23</b></td><td><b>24.09</b></td><td><b>26.7</b></td><td><b>22.82</b></td>
  </tr>
</table>