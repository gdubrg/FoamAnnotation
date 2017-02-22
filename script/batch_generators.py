import numpy as np
import glob
import cv2
import csv
from sklearn import preprocessing

def load_names(val_seq = -1, augm=0):
    # color_dir = '../Dataset/*/color/*.png'
    # color_list = glob.glob(color_dir)
    # max_idx = len(color_list)

    # distinzione caricamento train e validation
    gt_dir = '../Dataset/*.txt'

    #gt_dir = '../Dataset/' + val_seq + '.txt'

    gt_list = glob.glob(gt_dir)

    if val_seq < 0:
        # carica tutto e rimuovi sequenza di validazione
        to_remove = '../Dataset\\700_5000.txt'
        gt_list.remove(to_remove)
    else:
        # carica solo sequenza di validazione
        gt_list = glob.glob('../Dataset\\700_5000.txt')

    data = []

    # aprire tutti i file gt_list
    for gt_file in gt_list:
        f = open(gt_file, 'rb')

        # prima riga non devo leggerla
        fuffa = f.readline()

        # iniziano righe immagini
        lines = f.readlines()

        # parso la riga e memorizzo la roba
        for line in lines:
            row = line.split('\t')
            img_name = (gt_file[:gt_file.find(".txt")] + "/" + row[0])
            # gt_rows = ([int(row[1]), int(row[2].replace("\r\n", ""))])
            data.append({'image':img_name, 'sup':int(row[1]), 'inf':int(row[2].replace("\r\n", "")), 'augm':int(augm)})

    return data

def load_names_val():
    return load_names(val_seq=1, augm=0)

def load_images(train_data_names, crop, rows, cols):

    # canale
    ch = 1
    # dimensioni immagini
    rows = rows
    cols = cols

    # preparo struttura dati per immagini
    img_batch = np.zeros(shape=(len(train_data_names), ch, rows, cols), dtype=np.float32)
    # preparo struttura dati per GT
    y_batch = np.zeros(shape=(len(train_data_names), 2), dtype=np.float32)

    # estraggo nomi e gt
    for i, line in enumerate(train_data_names):

        # nome immagine
        img_name = line['image']

        # leggo immagine in BW
        img = cv2.imread(img_name, 0)

        # CROP
        if crop:
            img = img[400:, int((img.shape[1]/2)-(img.shape[1]/12)):int((img.shape[1]/2)+(img.shape[1]/12))]

        # resize
        img = cv2.resize(img, (cols, rows))

        # DATA AUGMENTATION
        if line['augm'] == 1:
            img = cv2.flip(img, 1)

        # NORMALIZZO
        #img = img / 255.0
        img = preprocessing.scale(img.astype('float'))

        # aggiungo dimensione canale
        img = np.expand_dims(img, 2)

        # DEBUG
        # cv2.imshow("caricanda", img)
        # cv2.waitKey()

        # carico nel batch
        img_batch[i] = img.transpose(2, 0, 1)

        # estraggo e carico nel batch il GT
        y_batch[i] = np.array([line['sup']/1024.0, line['inf']/1024.0])

    return img_batch, y_batch

def load_images_test(test_data_name, rows, cols):

    # canale
    ch = 1
    # dimensioni immagini
    rows = rows
    cols = cols

    # preparo struttura dati per immagini
    img_batch = np.zeros(shape=(1, ch, rows, cols), dtype=np.float32)
    # preparo struttura dati per GT
    y_batch = np.zeros(shape=(1, 2), dtype=np.float32)

    # estraggo nomi e gt


    # nome immagine
    img_name = test_data_name['image']

    # leggo immagine in BW
    img = cv2.imread(img_name, 0)

    # resize
    img = cv2.resize(img, (cols, rows))

    # aggiungo dimensione canale
    img = np.expand_dims(img, 2)

    # carico nel batch
    img_batch[0] = img.transpose(2, 0, 1)

    # estraggo e carico nel batch il GT
    y_batch[0] = np.array([test_data_name['sup'], test_data_name['inf']])

    return img_batch, y_batch



def new_generate_batch(batchsize, small):
    input_ratio = 1 if not small else 2
    # gt_dir = '../Dataset/*/frames/gtmaps/coordinates_gt.csv'
    gt_dir = '../Dataset/*/gtmaps/coordinates_gt.csv'
    gt_list = glob.glob(gt_dir)

    color_names = []
    gt_rows = []

    # aprire tutti i file gt_list
    for gt_file in gt_list:
        f = open(gt_file, 'rb')
        reader = csv.reader(f)

        for row in reader:
        # line = [row for row in reader]
            #color_names.append("../Dataset/"+row[0]+".png")
            color_names.append(row[0] + ".png")
            gt_rows.append(row[1:])

            # parsare gt_list, dividere nomi immagini e righe gt



    # devo crare le liste: nomi immagini, righe con GT
    max_idx = len(color_names)

    assert len(gt_rows) == len(color_names)

    while True:
        l_eye_batch = np.zeros(shape=(batchsize, 3, 224/input_ratio, 224/input_ratio), dtype=np.float32)
        r_eye_batch = np.zeros(shape=(batchsize, 3, 224/input_ratio, 224/input_ratio), dtype=np.float32)
        face_batch = np.zeros(shape=(batchsize, 3, 224/input_ratio, 224/input_ratio), dtype=np.float32)
        face_grid_batch = np.zeros(shape=(batchsize, 1, 25, 25), dtype=np.float32)

        y_batch = np.zeros(shape=(batchsize, 2), dtype=np.float32)

        b = 0
        while b < batchsize:
            idx = np.random.randint(0, max_idx)

            img = cv2.imread(color_names[idx])
            face, l_eye, r_eye, f_g = detectFaceandEye(img, scale=0.25)

            # assert the right detection of face and eyes
            if face == None or r_eye == None or l_eye == None or f_g == None:
                b -= 1

            else:

                l_eye_batch[b] = cv2.resize(l_eye, (224/input_ratio, 224/input_ratio)).transpose(2, 0, 1)
                r_eye_batch[b] = cv2.resize(r_eye, (224/input_ratio, 224/input_ratio)).transpose(2, 0, 1)
                face_batch[b] = cv2.resize(face, (224/input_ratio, 224/input_ratio)).transpose(2, 0, 1)
                face_grid_batch[b,0] = cv2.resize(src=f_g, dsize=(25, 25), interpolation=cv2.INTER_NEAREST)
                #cv2.resize(src=f_g, dsize=(25, 25), dst=face_grid_batch[b, 0], interpolation=cv2.INTER_NEAREST)

                y_batch[b] = np.array(gt_rows[idx])

            b += 1

        yield [l_eye_batch, r_eye_batch, face_batch, face_grid_batch], y_batch


def pre_load_old(path_to_csv):
    gt_dir = path_to_csv
    gt_list = glob.glob(gt_dir)

    color_names = []
    gt_rows = []

    # aprire tutti i file gt_list
    for gt_file in gt_list:
        f = open(gt_file, 'rb')
        reader = csv.reader(f)

        for row in reader:
            # line = [row for row in reader]
            # color_names.append("../Dataset/"+row[0]+".png")
            color_names.append(row[0] + ".png")
            gt_rows.append(row[1:])

    return color_names, gt_rows

def pre_load(path_to_csv):
    gt_dir = path_to_csv
    gt_list = glob.glob(gt_dir)

    color_names = []
    gt_rows = []
    data = []

    # aprire tutti i file gt_list
    for gt_file in gt_list:
        f = open(gt_file, 'rb')
        reader = csv.reader(f)

        for row in reader:
        # line = [row for row in reader]
            # color_names.append("../Dataset/"+row[0]+".png")
            # color_names.append(row[0] + ".png")
            # gt_rows.append(row[1:])
            # color_name = row[0]
            # rg_row = row[1:]
            data.append({'image':row[0], 'coor1':int(row[1]), 'coor2':int(row[2])})
            # data.append(row[0])

    return data

def load_rgb(train_data_names, small):
    input_ratio = 1 if not small else 2

    # nel pre_load ho restituito una lista di liste, devo quindi accedere alle liste
    # train_data_names = train_data_names[0]

    # preparo strutture dati
    l_eye_batch = np.zeros(shape=(len(train_data_names), 3, 224/input_ratio, 224/input_ratio), dtype=np.float32)
    r_eye_batch = np.zeros(shape=(len(train_data_names), 3, 224/input_ratio, 224/input_ratio), dtype=np.float32)
    face_batch = np.zeros(shape=(len(train_data_names), 3, 224/input_ratio, 224/input_ratio), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(len(train_data_names), 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros(shape=(len(train_data_names), 2), dtype=np.float32)


    # estraggo nomi e gt
    for i, line in enumerate(train_data_names):

        # estraggo nome immagine
        name_image = line['image']

        # leggo immagine
        img = cv2.imread(name_image + ".png")

        # trovo roba
        face, l_eye, r_eye, f_g = detectFaceandEye(img, scale=0.25)

        # se non trovo nulla non faccio niente
        if face == None or r_eye == None or l_eye == None or f_g == None:
            continue

        else:
            l_eye_batch[i] = cv2.resize(l_eye, (224/input_ratio, 224/input_ratio)).transpose(2, 0, 1)
            r_eye_batch[i] = cv2.resize(r_eye, (224/input_ratio, 224/input_ratio)).transpose(2, 0, 1)
            face_batch[i] = cv2.resize(face, (224/input_ratio, 224/input_ratio)).transpose(2, 0, 1)
            face_grid_batch[i, 0] = cv2.resize(src=f_g, dsize=(25, 25), interpolation=cv2.INTER_NEAREST)

            y_batch[i] = np.array([line['coor1'], line['coor2']])

    return [l_eye_batch, r_eye_batch, face_batch, face_grid_batch], y_batch


def load_depth(train_data_names, small):
    input_ratio = 1 if not small else 2

    # nel pre_load ho restituito una lista di liste, devo quindi accedere alle liste
    # train_data_names = train_data_names[0]

    # preparo strutture dati
    l_eye_batch = np.zeros(shape=(len(train_data_names), 3, 224/input_ratio, 224/input_ratio), dtype=np.float32)
    r_eye_batch = np.zeros(shape=(len(train_data_names), 3, 224/input_ratio, 224/input_ratio), dtype=np.float32)
    face_batch = np.zeros(shape=(len(train_data_names), 3, 224/input_ratio, 224/input_ratio), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(len(train_data_names), 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros(shape=(len(train_data_names), 2), dtype=np.float32)


    # estraggo nomi e gt
    for i, line in enumerate(train_data_names):

        # estraggo nome immagine
        name_image = line['image']

        # leggo immagine
        img = cv2.imread(name_image + ".png")

        # trovo roba
        face, l_eye, r_eye, f_g = detectFaceandEye(img, scale=0.25)

        # se non trovo nulla non faccio niente
        if face == None or r_eye == None or l_eye == None or f_g == None:
            continue

        else:
            l_eye_batch[i] = cv2.resize(l_eye, (224/input_ratio, 224/input_ratio)).transpose(2, 0, 1)
            r_eye_batch[i] = cv2.resize(r_eye, (224/input_ratio, 224/input_ratio)).transpose(2, 0, 1)
            face_batch[i] = cv2.resize(face, (224/input_ratio, 224/input_ratio)).transpose(2, 0, 1)
            face_grid_batch[i, 0] = cv2.resize(src=f_g, dsize=(25, 25), interpolation=cv2.INTER_NEAREST)

            y_batch[i] = np.array([line['coor1'], line['coor2']])

    return [l_eye_batch, r_eye_batch, face_batch, face_grid_batch], y_batch





