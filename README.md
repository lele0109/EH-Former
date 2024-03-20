# EH-Former
An official code for EH-former (submitted to Information Fusion)  

![image](https://github.com/lele0109/EH-Former/blob/main/Introduction_picture.png)

![image](https://github.com/lele0109/EH-Former/blob/main/overall.png)
****
## Requirements<br />
- Python 3.7
- Pytorch 1.8.0
- baal 1.6.0
****

## Datasets
Please download the datasets through:
- BUSI [link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
- UDIAT [link](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)
- STU [link](https://github.com/xbhlk/STU-Hospital)
- BUSBRA [link](https://doi.org/10.5281/zenodo.8231412)

The project should be finally organized as follows:
```
./EH-Former/
  ├── data/
      ├── train_image/
      ├── train_label/
      ├── test_image/
      ├── test_label/
  ├── loss_metrics.py
  ├── model/
  ├── dataloader.py 
  ├── main.py
  ...... 
```
****

## Weights
Please download the weights training on UDIAT, BUSI, SYSU through [link](https://pan.baidu.com/s/1lt_DE1ajIzeNbAuh0AOOGQ?pwd=wjej).

Please load_weight as follows:
```
checkpoint = torch.load(load_path + '/seg_weights.pth', map_location=device)
net.load_state_dict(checkpoint['model_state_dict'], strict=True)
set_params_recursive(net, checkpoint['alpha'])
```
## Running
Train
```
python main.py --train_mode=True --gpu 0 --Cnet_path='your stage1 network path'
```
Test
```
python main.py --train_mode=False --test_mode=True --gpu 0
```

## Citation<br />
If you use this code, please cite following paper, thanks.<br />
```
@article{qu2024eh,
  title={EH-Former: Regional Easy-Hard-Aware Transformer for Breast Lesion Segmentation in Ultrasound Images},
  author={Xiaolei Qu, Jiale Zhou, Jue Jiang, Wenhan Wang, Haoran Wang, Shuai Wang, Wenzhong Tang, Xun Lin},
  journal={Information Fusion},
  year={2024},
}
```
****
