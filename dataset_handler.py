import numpy as np
from MafQud1 import MafQud
from data_preprocessing.clean_images import train_test_split


mafqud = MafQud()
dataset = 'three_shot'  
op = 'train-test-split'
operations = ['first-stage', 'second-stage', 'train-test-split']


if op == operations[0]:
    # load train dataset
    trainX, trainy = mafqud.load_dataset(f'Data/{dataset}/train/')
    print(trainX.shape, trainy.shape)

    # load test dataset 
    testX, testy = mafqud.load_dataset(f'Data/{dataset}/test/')
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)

    np.savez_compressed(f'Data/{dataset}/missing_people_.npz', trainX, trainy, testX, testy)
    
    
elif op == operations[1]:
    # load dataset
    data = np.load(f'Data/{dataset}/missing_people_.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    newtrainX = mafqud.get_encodings(trainX)
    newtestX = mafqud.get_encodings(testX)
    np.savez_compressed(f'Data/{dataset}/missong_people_encoding_.npz', newtrainX, trainy, newtestX, testy)

elif op == operations[2]:
    rootDir = 'DataToSplit/train_test'
    dataDir = 'DataToSplit/train_test/Data'
    test_ratio = 0.3
    one_shot = 3
    include = False
    json_path = 'DataToSplit/missing_people_new.json'
    from_path = 'DataToSplit/images'
    to_path = 'DataToSplit/train_test/Data'
    img_per_person_to_remove = 3
    save_path = 'DataToSplit/train_test/missing_people_new.json'
    train_test_split(rootDir, dataDir, test_ratio, one_shot, include, json_path, from_path, to_path, img_per_person_to_remove, save_path)
