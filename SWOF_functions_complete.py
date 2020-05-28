import cv2
import numpy as np
import os
import keras
import matplotlib.pyplot as plt


#############################################################################

### DATASET GENERATOR FUNCTIONS

#############################################################################

def load_video(filename):
    video = cv2.VideoCapture(filename)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS)

    return video, length, width, height, fps

def split_video_frames(video, length, height, width):
    frames_color = np.zeros((length, height, width, 3), dtype=np.uint8)
    frames = np.zeros((length, height, width), dtype=np.uint8)
    for i in range(length):
        flag, frame = video.read()
        frames_color[i] = frame
        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frames

def generate_SWOF_data(DATADIR, VIDEONAME,frames, N=5, startframe=55, endframe=149):

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    #create directory to save optical flow CRASH images
    #store images in directory data/training/crash
    try:
        if not os.path.exists(DATADIR):
            os.makedirs(DATADIR)
    except OSError:
        print ('Error: Creating directory of data')

    #list of all optical flow crash images
    SWOF_images = []

    #index to save images
    ImageNum = 0

    for w in range(startframe, endframe):

        prev = frames[w - N + 1]
        p0 = cv2.goodFeaturesToTrack(prev, mask = None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(prev)

        for i in range( w - N + 2 , w + 1):

            prev = frames[i-1]

            if (i<1):
                break

            current = frames[i]
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev, current, p0, None, **lk_params)

            # Select good points
            if p1 is None:
                continue
            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)

            p0 = good_new.reshape(-1,1,2)

        name = DATADIR + str(VIDEONAME) + str('_image') + str(ImageNum) + '.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, mask)
        ImageNum += 1

        SWOF_images.append(mask)

    return SWOF_images


def generate_testdata(DATADIR):

    try:
        if not os.path.exists(DATADIR):
            os.makedirs(DATADIR)
    except OSError:
        print ('Error: Creating directory of data')

    # list all files in dir
    all_crash_imgs = [f for f in os.listdir(DATADIR) if os.path.isfile(f)]

    # select 0.1 of the files randomly
    random_files = np.random.choice(all_crash_imgs, int(len(all_crash_imgs)*.1))

    return SWOF_test_images







#############################################################################

### CNN FUNCTIONS

#############################################################################

def CNN_train_gen(DATADIR, traindatadirname='training', val_set=0.05):
    DATAFOLDER = DATADIR
    train_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                   rotation_range=40,
                                                                   zoom_range=0.2,
                                                                   horizontal_flip=True,
                                                                   vertical_flip=True,
                                                                   fill_mode='nearest',
                                                                   validation_split=0.05)
    X_train = train_generator.flow_from_directory(DATAFOLDER+traindatadirname,
                                                  color_mode='grayscale',
                                                  batch_size=32,
                                                  target_size=(108,192))
    return train_generator, X_train

def CNN_test_gen(DATADIR, testdatadirname='testing'):
    DATAFOLDER = DATADIR
    test_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                  rotation_range=40,
                                                                  zoom_range=0.2,
                                                                  horizontal_flip=True,
                                                                  vertical_flip=True,
                                                                  fill_mode='nearest')
    X_test = test_generator.flow_from_directory(DATAFOLDER+testdatadirname,
                                                color_mode='grayscale',
                                                batch_size=32,
                                                target_size=(108,192))
    return test_generator, X_test


def CNN_training(first_image, first_label, NUMBER_OF_CLASSES=2):

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                                    activation='relu',
                                    input_shape=first_image.shape))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
    return model

def CNN_analysis_plots(modelfit):
    history = modelfit

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return







#############################################################################

### REAL TIME FUNCTIONS

#############################################################################
import PIL.Image
from io import BytesIO
import IPython.display

# def read_bwframes(video):
#   #take 1st frame and grayscale
#   ret, frame = video.read()
#   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#   return ret, frame

def RT_image_predict(threshold, model, mask):
    RT_image = cv2.resize(mask, dsize=(192,108))
    RT_image = RT_image.reshape(1, np.shape(RT_image)[0], np.shape(RT_image)[1], 1)

    prediction = model.predict(RT_image)
    crash_prediction = prediction[0][0]
    if crash_prediction >= threshold:
        print("Look Out!")
    return crash_prediction

def OF_prelim():
# Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    return lk_params, feature_params, color


def rtcalc_SWOF(model, video, N=5, threshold_val=0.2, iterations_param=100):

    image_display = IPython.display.display("Real time display", display_id=1)

    ret, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    prev = frame

    frames = []
    frames.append(frame)

    lk_params, feature_params, color = OF_prelim()

    iteration = 0
    SWOF_images = []
    predictions=[]

    while iteration < iterations_param:

        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(frame)

        if (iteration >= N + 1):
            prev = frames[iteration - N + 1]
        p0 = cv2.goodFeaturesToTrack(prev, mask = None, **feature_params)

        mask = np.zeros_like(prev)

        for i in range(iteration - N + 2, iteration + 1):

            if (i<0):
                break

            prev = frames[i-1]
            current = frames[i]
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev, current, p0, None, **lk_params)

            if p1 is None:
                continue

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)

            p0 = good_new.reshape(-1,1,2)

        # cv2.imshow('First-Person-View', mask)
            # display image
        f = BytesIO()
        PIL.Image.fromarray(mask).save(f, 'jpeg')
        image = IPython.display.Image(data=f.getvalue(), width=360, height=240)
        image_display.update(image)

        crash_prediction = RT_image_predict(threshold=threshold_val, model=model, mask=mask)
        predictions.append(crash_prediction)

        SWOF_images.append(mask)
        iteration += 1

    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
    video.release()

    return SWOF_images, frames, predictions



def save_predictions(path_save_create, SWOF_images, frames, predictions):

    try:
        if not os.path.exists(path_save_create):
            os.makedirs(path_save_create)
    except OSError:
        print ('Error: Creating directory of data')

    path_save_mask = path_save_create + 'mask/'
    #do not include name you want for image
    path_save_prediction = path_save_create + 'pred/test.txt'
    #include the name you want for the file (ex. path\to\file\name_of_file.text)
    path_save_frame = path_save_create + 'frame/'

    # prepare file to write the prediction values
    f = open(path_save_prediction, "w")
    f.close()

    for i in range(len(frames)):

        #save raw frame
        pil_img_f = Image.fromarray(frames[i])
        pil_img_f.save(path_save_frame +'raw_frame'+str(i)+'.jpeg')

        #because len of raw frames is not = len of crash predictions or mask we need this if statement so we dont go out of bounds

        if(i<(len(frames)-1)): # need to do -1 becuase the highest length for these variables are one less than the number of
                                 # frames that exist
            #save mask
            pil_img_f = Image.fromarray(SWOF_images[i])
            pil_img_f.save(path_save_mask +'mask_frame'+str(i+1)+'.jpeg')

            #save prediction in text file we created above
            f = open(path_save_prediction, "a")
            f.write('line# ' + str(i+1) + ':crash prediction ='+str(predictions[i]) + '\n') # have line# = i+1 because it will
                                                              # align with the numbering for raw
                                                                                                # frames
            f.close() # close to write again later
    return



#############################################################################

### REAL TIME FUNCTIONS IN TERMINAL

#############################################################################

def rtcalc_SWOF_interm(model, video, N=5, threshold_val=0.2, iterations_param=100):

    ret, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    prev = frame

    frames = []
    frames.append(frame)

    lk_params, feature_params, color = OF_prelim()

    iteration = 0
    SWOF_images = []
    predictions=[]

    while iteration < iterations_param:

        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(frame)

        if (iteration >= N + 1):
            prev = frames[iteration - N + 1]
        p0 = cv2.goodFeaturesToTrack(prev, mask = None, **feature_params)

        mask = np.zeros_like(prev)

        for i in range(iteration - N + 2, iteration + 1):

            if (i<0):
                break

            prev = frames[i-1]
            current = frames[i]
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev, current, p0, None, **lk_params)

            if p1 is None:
                continue

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)

            p0 = good_new.reshape(-1,1,2)

        cv2.imshow('First-Person-View', mask)

        crash_prediction = RT_image_predict(threshold=threshold_val, model=model, mask=mask)
        predictions.append(crash_prediction)

        SWOF_images.append(mask)
        iteration += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()

    return SWOF_images, frames, predictions







#############################################################################

### UART FUNCTIONS

#############################################################################


import time
import serial

def UART_init():
    serial_port = serial.Serial(port="/dev/ttyTHS1",
                            baudrate=115200,
                            bytesize=serial.FIVEBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE)
    return serial_port

def UART():
    serial_port = UART_init()
    # Wait a second to let the port initialize
    time.sleep(1)
    try:
        serial_port.write("Crash".encode())
        print('ok')

    except KeyboardInterrupt:
        print("Exiting Program")

    except Exception as exception_error:
        print("Error occurred. Exiting Program")
        print("Error: " + str(exception_error))

    finally:
        serial_port.close()
        pass
    return


def RT_image_predict_forUART(threshold, model, mask):
    RT_image = cv2.resize(mask, dsize=(192,108))
    RT_image = RT_image.reshape(1, np.shape(RT_image)[0], np.shape(RT_image)[1], 1)

    prediction = model.predict(RT_image)
    crash_prediction = prediction[0][0]
    if crash_prediction >= threshold:
        UART()
    return crash_prediction

def rtcalc_SWOF_withUART(model, video, N=5, threshold_val=0.2, iterations_param=100):

    image_display = IPython.display.display("Real time display", display_id=1)

    ret, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    prev = frame

    frames = []
    frames.append(frame)

    lk_params, feature_params, color = OF_prelim()

    iteration = 0
    SWOF_images = []
    predictions=[]

    while iteration < iterations_param:

        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(frame)

        if (iteration >= N + 1):
            prev = frames[iteration - N + 1]
        p0 = cv2.goodFeaturesToTrack(prev, mask = None, **feature_params)

        mask = np.zeros_like(prev)

        for i in range(iteration - N + 2, iteration + 1):

            if (i<0):
                break

            prev = frames[i-1]
            current = frames[i]
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev, current, p0, None, **lk_params)

            if p1 is None:
                continue

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)

            p0 = good_new.reshape(-1,1,2)

        cv2.imshow('First-Person-View', mask)

        # f = BytesIO()
        # PIL.Image.fromarray(mask).save(f, 'jpeg')
        # image = IPython.display.Image(data=f.getvalue(), width=360, height=240)
        # image_display.update(image)

        crash_prediction = RT_image_predict_forUART(threshold=threshold_val, model=model, mask=mask)
        predictions.append(crash_prediction)

        SWOF_images.append(mask)
        iteration += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()

    return SWOF_images, frames, predictions
