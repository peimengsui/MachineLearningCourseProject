# MachineLearningCourseProject

## Code Running Instructions
All the data should be kept in the /data directory from this link https://drive.google.com/open?id=0B8EoX4shjz25UXU1R1R0TDNYQU0
### 1. User-based Collaborative Filtering:
The data required are: train_data.txt, valid_visible.txt and valid_predict.txt
To generate recommendation using user-based similarity in parallel you can run under the code directory:
```bash
python train.py [BEGIN_IDX] [TO_IDX] 1 [Q]
```
The first and second arguments representing the beginning end ending user index of the validation data you want to recommend. The fourth argument is set to be 1 to enter the user-based mode. The fifth argument is for tunning the parameter Q. 
### 2. Song-based Collaborative Filtering:
The data required are: train_data.txt, valid_visible.txt and valid_predict.txt
To generate recommendation using song-based similarity in parallel you can run under the code directory:
```bash
python train.py [BEGIN_IDX] [TO_IDX] 2 [Q]
```
The first and second arguments representing the beginning end ending user index of the validation data you want to recommend. The fourth argument is set to be 2 to enter the song-based mode. The fifth argument is for tunning the parameter Q. 
