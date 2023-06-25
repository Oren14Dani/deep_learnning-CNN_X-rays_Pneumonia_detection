import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import os
import numpy as np 
import cv2
import tensorflow as tf
#importing the images from google drive 
from google.colab import drive
drive.mount('/content/drive')

class dataset:
    """Import data from google drive and split it into lists of: 
    x_train, y_train, x_val, y_val, x_test, y_test.
    ImegeSize is a number that define the square size of the image
    to be resize in order to uniform the training dataset input.
    ImageType defines how many colors the image has - gray (1 dimension) 
    or RGB (3 dimensions)."""
    def __init__(self, ImegeSize, ImageType=cv2.IMREAD_GRAYSCALE):
        # paths of the image:
        training_dir='/content/drive/MyDrive/Colab Notebooks/finalproject/train'
        test_dir='/content/drive/MyDrive/Colab Notebooks/finalproject/test'
        validation_dir='/content/drive/MyDrive/Colab Notebooks/finalproject/val'
        # creating array of images and array of labels: x=images, y=labels
        self.x_test, self.y_test = self.make_dataset(test_dir, ImegeSize, ImageType)
        self.x_val, self.y_val = self.make_dataset(validation_dir, ImegeSize, ImageType)
        self.x_train, self.y_train = self.make_dataset(training_dir, ImegeSize, ImageType)
        
    def make_dataset(self, dir, ImegeSize, ImageType):
        """creating array of test images and array of labels(x=images,y=labels)"""
        x = []
        y = []

        normal_images_folder_path = dir +'/NORMAL'
        for pic in os.listdir(normal_images_folder_path):
            image = cv2.imread(os.path.join(normal_images_folder_path, pic), ImageType)
            if image is not None:
                image = cv2.resize(image, (ImegeSize, ImegeSize))
                x.append(image)  
                y.append(0) # lable
        
        pneumonia_images_folder_path = dir +'/PNEUMONIA'
        for pic in os.listdir(pneumonia_images_folder_path):
            image = cv2.imread(os.path.join(pneumonia_images_folder_path, pic),ImageType)
            if image is not None:
                image = cv2.resize(image, (ImegeSize, ImegeSize))
                x.append(image)
                y.append(1) # lable
                
        #Noramlize the values to be between 0-1
        x = np.array(x) / 255
        y = np.array(y)
        
        deep=1
        if(ImageType==cv2.IMREAD_GRAYSCALE):
            deep=1
        elif(ImageType==cv2.IMREAD_COLOR):
            deep=3
            
        # resize data for deep learning 
        x = x.reshape(-1, ImegeSize, ImegeSize, deep)
        return x,y           
    
def CreateDataGeneratorAugmentation(FeatureList, zoom, shift, angle):
    # With data augmentation to prevent overfitting and handling the imbalance in dataset
    DataGenerator = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range = angle,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = zoom, # Randomly zoom image 
            width_shift_range=shift,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=shift,  # randomly shift images vertically (fraction of total height)
            horizontal_flip = True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    #fit the data generator with train images:
    DataGenerator.fit(FeatureList)
    return DataGenerator

def CreateNetworkWithTransferLearning(ImageResize, drop, MoreLayer=0, Percptorn_factor=7):
    input_shape = (ImageResize, ImageResize, 3)
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=input_shape, include_top=False)

    for layer in base_model.layers:
        layer.trainable = False # freez the base model layers

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2**Percptorn_factor, activation = 'relu'))
    model.add(Dropout(drop))
    if(MoreLayer): # part of mission 
        for i in range(MoreLayer):
            model.add(Dense(2**Percptorn_factor, activation = 'relu'))
            model.add(Dropout(drop))
    model.add(Dense(1, activation='sigmoid'))

    model.summary(expand_nested=False,show_trainable=True)
    return model,base_model

def CreateNetworkWithoutTransferLearning(ImageResize,drop,KERNAL, MoreLayer=0):
    #creating the network:
    model = Sequential()

    model.add(Conv2D(32*KERNAL , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' ,
                     input_shape = (ImageResize,ImageResize,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    model.add(Conv2D(64*KERNAL , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(drop))
    model.add(BatchNormalization())

    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64*KERNAL , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())

    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(128*KERNAL , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(drop))
    model.add(BatchNormalization())

    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(128*KERNAL , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(drop))
    model.add(BatchNormalization())

    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(256*KERNAL , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(drop))
    model.add(BatchNormalization())

    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 128 , activation = 'relu'))
    model.add(Dropout(drop))
    
    if(MoreLayer): # part of mission 3.1
        for i in range(MoreLayer):
            model.add(Dense(128, activation = 'relu'))
            model.add(Dropout(drop))
    
    model.add(Dense(units = 1 , activation = 'sigmoid'))

    model.summary()
    return model

def CompileNTrainModel(Model,OptimizerName,Epochs,LearningRate,Batch_size,DataGenerator,x_train,y_train,x_val,y_val,x_test,y_test,patience,EarlyStopping,history=None):
    #using reduction of learning rate to improve performance:
    if(OptimizerName=='Adam'):
        opt=tf.keras.optimizers.Adam(learning_rate=LearningRate)
    elif(OptimizerName=='RMSprop'):
        opt=tf.keras.optimizers.RMSprop(learning_rate=LearningRate)
    elif(OptimizerName=='SGD'):
        opt=tf.keras.optimizers.SGD(learning_rate=LearningRate)
    elif(OptimizerName=='MSGD'):
        opt=tf.keras.optimizers.SGD(learning_rate=LearningRate,momentum=0.9,nesterov=True)
    else:
        return "Wrong optimizer name type"
    
    Model.compile(optimizer = opt , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=0,factor=0.3, min_lr=0.000001)
    if(EarlyStopping==True):
      EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=patience)
      _callbacks=[learning_rate_reduction,EarlyStopping]
    else:
      _callbacks=[learning_rate_reduction]
    #Need to set an early stop callback!!
    #fit the model:

    if(history==None):  
        initial_epoch=0
        print("The history is now!")
    else:
        print("The history is return!")
        initial_epoch=(2*Epochs)//3
        history_dict=history.history
    #callbacks = [learning_rate_reduction,EarlyStopping]

    New_history = Model.fit(DataGenerator.flow(x_train,y_train, batch_size = Batch_size), # take data and lable arrays 
                            epochs = Epochs,
                            validation_data = DataGenerator.flow(x_val, y_val),
                            callbacks = _callbacks,
                            initial_epoch=initial_epoch, 
                            verbose=0)

    #printing the accuracy of the model and the loss:
    print("Loss of the model is - " , Model.evaluate(x_test,y_test)[0])
    print("Accuracy of the model is - " , Model.evaluate(x_test,y_test)[1]*100 , "%")

    New_history_dict = New_history.history #dictionary of acc and loss for train and val:
    #creating array of acc and loss for train and val:
    if(history != None): # in case of fine tune
        acc = history_dict['accuracy'] + New_history_dict['accuracy']
        val_acc = history_dict['val_accuracy'] + New_history_dict['val_accuracy']
        loss = history_dict['loss'] + New_history_dict['loss']
        val_loss = history_dict['val_loss'] + New_history_dict['val_loss']
    else:
        acc = New_history_dict['accuracy']
        val_acc = New_history_dict['val_accuracy']
        loss = New_history_dict['loss']
        val_loss = New_history_dict['val_loss']

    New_history.history['accuracy']=acc
    New_history.history['val_accuracy']=val_acc
    New_history.history['loss']=loss
    New_history.history['val_loss']=val_loss

    epochs = range(1, len(acc)+1)
    #plotting a graph of training and validation loss(as function of epochs)  
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf() # clear figure
    #plotting a graph of training and validation accuracy(as function of epochs)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return New_history

def CalcModelPrecisionNRecall(y_test, PredictionList,Threshold):
    """ function to return true positive,false positive,true negetive,false negetive from array of prediction """
    TruePostive = 0
    FalsePostive = 0
    TrueNegative = 0
    FalseNegative = 0
    for i in range(len(PredictionList)): 
        if y_test[i]==PredictionList[i]==1:
            TruePostive+=1
        elif y_test[i]==0 and y_test[i]!=PredictionList[i]:
            FalsePostive+=1
        elif y_test[i]==PredictionList[i]==0:
            TrueNegative+=1
        elif y_test[i]==1 and y_test[i]!=PredictionList[i]:
            FalseNegative+=1

    if((TruePostive+FalsePostive)==0):
        Precision=0
    else:
        Precision=TruePostive/(TruePostive+FalsePostive)
    if((TruePostive+FalseNegative)==0):
        Recall=0      
    else:
        Recall=TruePostive/(TruePostive+FalseNegative)
    if((Precision+Recall)==0):
        f_score=0
    else:
        f_score=2*((Precision*Recall)/(Precision+Recall))

    #accuracy=(TruePostive+TrueNegative)/(TruePostive+TrueNegative+FalseNegative+FalsePostive)
    #print('Accuracy is ' +str(accuracy)+ 'for threshold ' +str(Threshold))
    return(Precision,Recall,f_score)

def CheckPredictionAgainstThershold(PredictionList,Threshold):
    for i in range(len(PredictionList)):
        if(PredictionList[i]>=Threshold):
            PredictionList[i]=1
        else:
            PredictionList[i]=0
    return PredictionList

def LoadModelNEvalutePerformance(model_path,x_test,y_test):
    Model=keras.models.load_model(model_path)
    #Seting some paramters for the training
    Min_Threshold=0.1
    Max_Threshold=0.95
    step=0.05
    ModelPerformance=CalcModelPerformanceParamters(Model,step,Min_Threshold,Max_Threshold,x_test,y_test)
    Print_ModelPerformance(ModelPerformance)
    del Model

def CalcModelPerformanceParamters(Model,step,Min_Threshold,Max_Threshold,x_test,y_test):
    #Allocating memory for calcaulation of the Model paramters performance.
    precision_list=[]
    recall_list=[]    
    f_score_list=[]
    f_score_max=0
    tmp_f_score=0
    #Threshold is in range of [0.1,0.95] ,with 0.05 step. 
    Threshold_levels=np.arange(Min_Threshold,Max_Threshold,step)   

    #calculate the precision,recall,f_score and put the result in their array:
    for Threshold in list(Threshold_levels):
        pred_by_softmax = Model.predict(x_test, batch_size = 32)
        pred_after_thershold=CheckPredictionAgainstThershold(pred_by_softmax,Threshold)
        current_precision,current_recall,tmp_f_score= CalcModelPrecisionNRecall(y_test,pred_after_thershold,Threshold)

        precision_list.append(current_precision)
        recall_list.append(current_recall)
        f_score_list.append(tmp_f_score)

        if(tmp_f_score>f_score_max):
            f_score_max=tmp_f_score
    
    ModelPerformance={"precision_list":precision_list,"recall_list":recall_list,"f_score_list":f_score_list,"f_score_max":f_score_max}
    return  ModelPerformance    

def Print_ModelPerformance(ModelPerformance):
    #plotting the recall-percision graph:     
    plt.plot(ModelPerformance["recall_list"], ModelPerformance["precision_list"], 'b')
    plt.scatter(ModelPerformance["recall_list"], ModelPerformance["f_score_list"], c='r')
    plt.title('Recall-percision graph')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()    
    print("The f_score max is: "+str(ModelPerformance["f_score_max"]))
    f_score_max=ModelPerformance["f_score_max"]
    len_f_score_list=len(ModelPerformance["f_score_list"])
    for i in range(len_f_score_list):
        if(ModelPerformance["f_score_list"][i]==f_score_max):
            print("The corresponding threshold for the maximum f_score is: "+str(0.1+0.05*i))
            break

def PerformFineTune(model,model_base,parameter,DataSet,DataGenerator,history,optimezr='Adam',EPOHCS=20,lr=0.01,Batch_size=32,dropout=0.1,ImageResize=150,zoom_range=0.2,shift_range=0.1,rotation_range=30,EarlyStopping=False,patience=4):
   # Model=keras.models.load_model(model_path)
    model_base.trainable = True
    fine_tune_at = 150
    for layer in model_base.layers[:fine_tune_at]:
        layer.trainable = False

    model.summary(expand_nested=False,show_trainable=True)    
    #Compile and train the (with TRANSFER_LEARNING) CNN model with the optimezer ADAM,20 EPOCHS,0.01 learining rate and 32 Batch size.
    initial_epochs=EPOHCS
    fine_tune_epochs = EPOHCS//2
    total_epochs = initial_epochs+fine_tune_epochs
    CompileNTrainModel(model,optimezr,total_epochs,lr,Batch_size,DataGenerator,
                       DataSet.x_train,DataSet.y_train,
                       DataSet.x_val,DataSet.y_val,
                       DataSet.x_test,DataSet.y_test,history=history,patience=patience,EarlyStopping=EarlyStopping)

def CNN_NETWORK_WITH_TRANSFER_LEARNING(DataSet,FineTune=True,optimezr='Adam',EPOHCS=20,lr=0.01,Batch_size=32,dropout=0.1,
                                       ImageResize=200,zoom_range=0.2,shift_range=0.1,rotation_range=30,EarlyStopping=False,
                                       patience=4,MoreLayer=0,Percptorn_factor=7):
    #Seting some paramters for the training
    step=0.05
    Min_Threshold=0.1
    Max_Threshold=0.95
    #This data generator will be fiting to the data set, and the augmentation will be with 0.2 zoom range, width and 0.1 height shift range, and 30 rotation range
    DataGenerator = CreateDataGeneratorAugmentation(DataSet.x_train,zoom_range,shift_range,rotation_range)

    #Build the CNN model with TRANSFER_LEARNING
    TRANSFER_LEARNING_CNN_model, base_model = CreateNetworkWithTransferLearning(ImageResize,dropout,MoreLayer=MoreLayer,Percptorn_factor=Percptorn_factor)

    ####################################################################
    ### step 1 of training - train only the model additional layers ####
    ####################################################################
    #Compile and train the (with TRANSFER_LEARNING) CNN model with the optimezer ADAM,20 EPOCHS,0.01 learining rate and 32 Batch size.
    history=CompileNTrainModel(TRANSFER_LEARNING_CNN_model,optimezr,EPOHCS,lr,Batch_size,DataGenerator,
                       DataSet.x_train,DataSet.y_train,
                       DataSet.x_val,DataSet.y_val,
                       DataSet.x_test,DataSet.y_test,
                       EarlyStopping=EarlyStopping,patience=patience)
    filename = optimezr+'_lr'+str(lr)+'_EPOHCS'+str(EPOHCS)+'.h5'
    TRANSFER_LEARNING_CNN_model.save('/content/drive/MyDrive/Colab Notebooks/CNN WITH TRANSFER LEARNING/Saved Models/TRANSFER_LEARNINGCNN_model_'+filename)
    #Caclulte model perfromance by a range of therhold values[0.1,0.95] with 0.05 step, the paramters that mesure is prescsion and recall.
    TRANSFER_LEARNING_CNN_ModelPerformance=CalcModelPerformanceParamters(TRANSFER_LEARNING_CNN_model,step,Min_Threshold,Max_Threshold,DataSet.x_test,DataSet.y_test)
    #plot the model preformance result
    print(filename)
    Print_ModelPerformance(TRANSFER_LEARNING_CNN_ModelPerformance)


    ###############################################################################
    ### step 2 of training - the fine tune - train selected layers of the model ###
    ###############################################################################
    if(FineTune==True):
        base_layer_paramter=2
        PerformFineTune(TRANSFER_LEARNING_CNN_model,base_model,base_layer_paramter,
                        DataSet, DataGenerator, history, optimezr, EPOHCS,lr/10, Batch_size,
                        EarlyStopping=EarlyStopping, patience=patience)
    filename = optimezr+'_lr'+str(lr)+'_EPOHCS'+str(EPOHCS)+'_FineTune.h5'
    TRANSFER_LEARNING_CNN_model.save('/content/drive/MyDrive/Colab Notebooks/CNN WITH TRANSFER LEARNING/Saved Models/TRANSFER_LEARNINGCNN_model_'+filename)
    #Caclulte model perfromance by a range of therhold values[0.1,0.95] with 0.05 step, the paramters that mesure is prescsion and recall.
    TRANSFER_LEARNING_CNN_ModelPerformance=CalcModelPerformanceParamters(TRANSFER_LEARNING_CNN_model,step,Min_Threshold,Max_Threshold,DataSet.x_test,DataSet.y_test)
    #plot the model preformance result
    print(filename)
    Print_ModelPerformance(TRANSFER_LEARNING_CNN_ModelPerformance)

    
    del TRANSFER_LEARNING_CNN_model

def CNN_NETWORK(DataSet,optimezr='Adam',EPOHCS=16,lr=0.01,Batch_size=32,dropout=0.1,ImageResize=200,zoom_range=0.2,shift_range=0.1,
                rotation_range=30,EarlyStopping=False,patience=4,KERNEL=1, MoreLayer=False):
    #Seting some paramters for the training
    step=0.05
    Min_Threshold=0.1
    Max_Threshold=0.95
    #This data generator will be fiting to the data set, and the augmentation will be with 0.2 zoom range,
    # width and 0.1 height shift range, and 30 rotation range
    DataGenerator=CreateDataGeneratorAugmentation(DataSet.x_train,zoom_range,shift_range,rotation_range)

    #Build the CNN model
    CNN_model=CreateNetworkWithoutTransferLearning(ImageResize,dropout,KERNEL, MoreLayer=MoreLayer)

    #Compile and train the CNN model with the optimezer ADAM,20 EPOCHS,0.01 learining rate and 32 Batch size.
    CompileNTrainModel(CNN_model,optimezr,EPOHCS,lr,Batch_size,DataGenerator,
                        DataSet.x_train, DataSet.y_train,
                        DataSet.x_val, DataSet.y_val,
                        DataSet.x_test, DataSet.y_test,
                        patience=patience,EarlyStopping=EarlyStopping)

    CNN_model.save('/content/drive/MyDrive/Colab Notebooks/CNN NETWORK/Saved Models/CNN_model_'+optimezr+'_lr'+str(lr)+'_EPOHCS'+str(EPOHCS)+'.h5')
    
    #Caclulte model perfromance by a range of therhold values[0.1,0.95] with 0.05 step, the paramters that mesure is prescsion and recall.
    CNN_ModelPerformance=CalcModelPerformanceParamters(CNN_model,step,Min_Threshold,Max_Threshold,DataSet.x_test,DataSet.y_test)
    #plot the model preformance result
    Print_ModelPerformance(CNN_ModelPerformance)


    del CNN_model

def main():
    #### User Interface  ####
    # choose to activeate network:
    CNN_network_active = True
    CNN_transfer_learning_active = True

    # choose to activeate mission:
    mission_2_active = False # Reference Network
    mission_31A_active = False # Add layer to the Network
    mission_31B_active = False # change the KERNAL size OR add 5 hidden layers
    mission_32_active = False # different optimizer,epohcs and LR
    mission_33A_active = False # change droput probability
    mission_33B_active = False # add Early stop
    final_result = True
    
    ### Constants ###
    ImageSize=250
    OPTIMIZER_LIST=['Adam', 'RMSprop', 'SGD', 'MSGD']
    EPOHCS_LIST = [8, 16, 24]
    LR_LIST = [0.1, 0.01, 0.001]
    
    # default values:
    OPTIMIZER_ref ='Adam'
    LR_ref = 0.001
    EPOCH_ref1 = 16
    EPOCH_ref2 = 20
    Percptorn_factor_ref = 7
    
    #######
    
    if(CNN_network_active): ### CNN network Model ####
        print ("\n### CNN network Model ###\n")
        OurDataSet = dataset(ImageSize, ImageType=cv2.IMREAD_GRAYSCALE) # data here is gray colors imeges

        if (mission_2_active): 
            print('\n### mission_2: Reference Network ###\n')
            print("\n### Optimizer: ", OPTIMIZER_ref ," , lr: ", LR_ref , " , EPOHCS: ", EPOCH_ref1 ," , KERNAL: ",1, " , MoreLayer: ",'no'  "###\n")
            CNN_NETWORK(OurDataSet, ImageResize=ImageSize,lr=LR_ref,optimezr=OPTIMIZER_ref,EPOHCS=EPOCH_ref1,KERNEL=1,MoreLayer=False)
            
        if (mission_31A_active): 
            print('\n ### mission_31A: Add layer to the Network ### \n')
            print("\n### Optimizer: ", OPTIMIZER_ref ," , lr: ", LR_ref , " , EPOHCS: ", EPOCH_ref1 ," , KERNAL: ",1, " , MoreLayer: ",'yes'  "###\n")
            CNN_NETWORK(OurDataSet, ImageResize=ImageSize,lr=LR_ref,optimezr=OPTIMIZER_ref,EPOHCS=EPOCH_ref1,KERNEL=1, MoreLayer=True)
        
        if (mission_31B_active): 
            print('\n ### mission_31B: change the KERNAL size ### \n')
            KERNEL_LIST = [0.25, 0.5, 0.75, 1.25, 1.5]
            for ker in KERNEL_LIST:
                print("\n### Optimizer: ", OPTIMIZER_ref ," , lr: ", LR_ref , " , EPOHCS: ", EPOCH_ref1 ," , KERNAL: ",ker, "###\n")
                CNN_NETWORK(OurDataSet, ImageResize=ImageSize,lr=LR_ref,optimezr=OPTIMIZER_ref,EPOHCS=EPOCH_ref1,KERNEL=ker)
            
        if (mission_32_active): 
            print('\n ### mission_32: different optimizer,epohcs and LR ### \n')
            for opt in OPTIMIZER_LIST:
                for epoch in EPOHCS_LIST:
                    for lr in LR_LIST:
                        print("\n ### Optimizer: ", opt ," , lr: ", lr , " , EPOHCS: ", epoch , "###\n")
                        CNN_NETWORK(OurDataSet, ImageResize=ImageSize, optimezr=opt, lr=lr, EPOHCS=epoch)
        
        if (mission_33A_active):
            print('\n ### mission_33a: change droput probability  ### \n')
            droput_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
            for drop in droput_LIST:
                print("\n### Optimizer: ", OPTIMIZER_ref ," , lr: ", LR_ref , " , EPOHCS: ", EPOCH_ref1 ," , droput: ",drop, "###\n")
                CNN_NETWORK(OurDataSet, ImageResize=ImageSize,lr=LR_ref,optimezr=OPTIMIZER_ref,EPOHCS=EPOCH_ref1,dropout=drop)           
        
        if (mission_33B_active):
            print('\n ### mission_33b: add Early stop and change patience ### \n')
            patience_list = [1,2,3,4,5]
            for patience in patience_list:
                print("### Optimizer: ", OPTIMIZER_ref ," , lr: ", LR_ref , " , EPOHCS: ", EPOCH_ref2 ," , Early stop and patience: ",patience, "###\n")
                CNN_NETWORK(OurDataSet, ImageResize=ImageSize, optimezr=OPTIMIZER_ref, lr=LR_ref, EPOHCS=EPOCH_ref1, EarlyStopping=True, patience=patience)
        
        if(final_result):
            print('\n ### test the Final Result with optimal parameters found ### \n')
            opt = 'RMSprop'
            epoch = 16
            lr = 0.001
            drop = 0.1
            EarlyStopping = True
            patience = 4
            print("### Optimizer: ", opt ," , lr: ", lr , " , EPOHCS: ", epoch ," , drop: ", drop ," , Early stop and patience: ", patience , "###\n")
            CNN_NETWORK(OurDataSet, ImageResize=ImageSize, optimezr=opt, lr=lr, EPOHCS=epoch, EarlyStopping=EarlyStopping, dropout=drop, patience=patience)
            
    
    if(CNN_transfer_learning_active):
        print ("\n### CNN transfer learning network Model ###\n")
        OurDataSet = dataset(ImageSize,ImageType=cv2.IMREAD_COLOR) # data here is RGB colors imeges
        if (mission_2_active): 
            print('\n###Reference Network###\n')
            print("\n ### Optimizer: ", OPTIMIZER_ref ," , lr: ", LR_ref , " , EPOHCS: ", EPOCH_ref2, " , Percptorn_factor: ", Percptorn_factor_ref , " ###\n")
            CNN_NETWORK_WITH_TRANSFER_LEARNING(OurDataSet, ImageResize=ImageSize,lr=LR_ref,optimezr=OPTIMIZER_ref,EPOHCS=EPOCH_ref2,
                                               Percptorn_factor=Percptorn_factor_ref, MoreLayer=False)

        if (mission_31A_active):
            print('\n ### Add layer to the Network### \n')
            print("\n ### Optimizer: ", OPTIMIZER_ref ," , lr: ", LR_ref , " , EPOHCS: ", EPOCH_ref2, " , Percptorn_factor: ", Percptorn_factor_ref, " , MoreLayer: ",'yes' , " ###\n")
            CNN_NETWORK_WITH_TRANSFER_LEARNING(OurDataSet, ImageResize=ImageSize,lr=LR_ref,optimezr=OPTIMIZER_ref,EPOHCS=EPOCH_ref2,
                                               Percptorn_factor=Percptorn_factor_ref, MoreLayer=True)
        
        if (mission_31B_active): 
            print('\n### mission_31B: add 5 hidden layers ###\n')
            k_list = [1,2,3,4,5]
            for k in k_list:
                print("\n ### Optimizer: ", OPTIMIZER_ref ," , lr: ", LR_ref , " , EPOHCS: ", EPOCH_ref2, " , added layers: ", k , " ###\n")
                CNN_NETWORK_WITH_TRANSFER_LEARNING(OurDataSet, ImageResize=ImageSize,lr=LR_ref,optimezr=OPTIMIZER_ref,EPOHCS=EPOCH_ref2,
                                                   FineTune=True, MoreLayer=k)
            
        if (mission_32_active): #### mission 3.2 - for CNN network Model ####
            for opt in OPTIMIZER_LIST:
                for epoch in EPOHCS_LIST:
                    for lr in LR_LIST:
                        print("\n ### Optimizer: ", opt ," , lr: ", lr , " , EPOHCS: ", epoch , " ###\n")
                        CNN_NETWORK_WITH_TRANSFER_LEARNING(OurDataSet, ImageResize=ImageSize, optimezr=opt, lr=lr, EPOHCS=epoch)
        
        if (mission_33A_active):
            print('\n ### mission_33a: change droput probability  ### \n')
            droput_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
            for drop in droput_LIST:
                print("\n### Optimizer: ", OPTIMIZER_ref ," , lr: ", LR_ref , " , EPOHCS: ", EPOCH_ref2 ," , droput: ",drop, "###\n")
                CNN_NETWORK_WITH_TRANSFER_LEARNING(OurDataSet, ImageResize=ImageSize,lr=LR_ref,optimezr=OPTIMIZER_ref,EPOHCS=EPOCH_ref2,
                                                   dropout=drop)           
        
        if (mission_33B_active):
            print('\n ### mission_33b: add Early stop and change patience ### \n')
            patience_list = [1,2,3,4,5]
            for patience in patience_list:
                print("### Optimizer: ", OPTIMIZER_ref ," , lr: ", LR_ref , " , EPOHCS: ", EPOCH_ref2 ," , Early stop and patience: ",patience, "###\n")
                CNN_NETWORK_WITH_TRANSFER_LEARNING(OurDataSet, ImageResize=ImageSize, optimezr=OPTIMIZER_ref, lr=LR_ref,
                                               EPOHCS=EPOCH_ref2, EarlyStopping=True, patience=patience)
        
        if(final_result):
            print('\n ### test the Final Result with optimal parameters found ### \n')
            opt = 'RMSprop'
            epoch = 24
            lr = 0.001
            MoreLayer = 3
            drop = 0.3
            EarlyStopping = True
            patience = 3
            print("### Optimizer: ", opt ," , lr: ", lr , " , EPOHCS: ", epoch ," , drop: ", drop ," , Early stop and patience: ",
                  patience , " , MoreLayer: ", MoreLayer ,"###\n")
            CNN_NETWORK_WITH_TRANSFER_LEARNING(OurDataSet, ImageResize=ImageSize, optimezr=OPTIMIZER_ref, lr=LR_ref,
                                               EPOHCS=EPOCH_ref2, EarlyStopping=True, patience=patience, MoreLayer=MoreLayer)
        
    #if(0):
        #NO_TRANSFER_model_path='/content/drive/MyDrive/Colab Notebooks/CNN NETWORK/Saved Models/CNN_model.h5'
        #x_train,y_train,x_val,y_val,x_test,y_test=ImportDataSet(Imgsize)
        #LoadModelNEvalutePerformance(NO_TRANSFER_model_path,x_test,y_test)
    #if(0):
        #TRANSFER_LEARNING_model_path='/content/drive/MyDrive/Colab Notebooks/CNN WITH TRANSFER LEARNING/Saved Models/TRANSFER_LEARNINGCNN_model.h5'
        #x_train,y_train,x_val,y_val,x_test,y_test=ImportDataSet(Imgsize,ImageType=cv2.IMREAD_COLOR)
        #LoadModelNEvalutePerformance(TRANSFER_LEARNING_model_path,x_test,y_test)
if __name__ == "__main__":
  main()