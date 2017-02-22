import sys
import os
import matplotlib.pyplot as plt
# SETTAGGIO SCHEDA GRAFICA
if len(sys.argv) == 2:
    gpu_id = sys.argv[1]
else:
    gpu_id = "cpu"
    cnmem = "0.7"
print("Argument: gpu={}".format(gpu_id))
os.environ["THEANO_FLAGS"] = "device=" + gpu_id

from batch_generators import load_names, load_images, load_images_test, load_names_val
from models import foamNet, foamNet2
import numpy as np
import cv2

import warnings
warnings.filterwarnings("ignore")


# if __name__ == '__main__':
#
#     # parametri immagine
#     rows = 256
#     cols = 192
#     ch = 1
#
#     # sequenza test
#     test_seq = "700_5000"
#
#     # visualizzazione immagini
#     b_visualize = False
#
#     # visualizzazione grafici
#     b_plot = True
#
#     # MODELLO
#     model = foamNet2(rows, cols)
#
#     # PESI
#     print("Load weights ...")
#     model.load_weights('weights/weights.019-897.05999.hdf5')
#
#     # img_batch = np.zeros(shape=(1, ch, rows, cols), dtype=np.float32)
#
#     # strutture dati
#     predictions_sup = []
#     predictions_inf = []
#     GT_sup = []
#     GT_inf = []
#     ist = []
#
#     # carico sequenza test
#     test_data_names = load_names_val()
#
#     for i, element in enumerate(test_data_names):
#
#         ist.append(i)
#
#         X, Y = load_images_test(element, rows, cols)
#
#         pred = model.predict(x=X, batch_size=1, verbose=0)
#
#         # salvataggio
#         GT_sup.append(Y[0][0])
#         GT_inf.append(Y[0][1])
#         predictions_sup.append(pred[0][0])
#         predictions_inf.append(pred[0][1])
#
#         #print "Predizione:", pred[0][0], pred[0][1]
#         #print "GT:", Y[0][0], Y[0][1]
#
#         # apro immagine originale
#
#         if b_visualize:
#             img = cv2.imread(element['image'], 0)
#
#             cv2.line(img, (0,int(pred[0][0])), (770, int(pred[0][0])), 255)
#             cv2.line(img, (0,int(pred[0][1])), (770, int(pred[0][1])), 255)
#
#             cv2.imshow("Predizione grafica", cv2.resize(img, dsize=None, fx = 0.5, fy=0.5))
#             cv2.waitKey()
#
#     if b_plot:
#         fig = plt.figure()
#
#         plt.plot(ist, GT_sup, 'r')
#         plt.plot(ist, GT_inf, 'b')
#
#         plt.plot(ist,predictions_sup, 'k' )
#         plt.plot(ist, predictions_inf, 'k')
#
#         plt.show()
#
# cv2.destroyAllWindows()

if __name__ == '__main__':

    # parametri immagine
    rows = 256
    cols = 192
    ch = 1

    # parametri del training
    b_crop = False
    batch_size = 16

    # visualizzazione immagini
    b_visualize = False

    # visualizzazione grafici
    b_plot = True

    # MODELLO
    model = foamNet2(rows, cols)

    # PESI
    print("Load weights ...")
    model.load_weights('weights2/weights.010-0.10569.hdf5')
    print("Done.")

    # img_batch = np.zeros(shape=(1, ch, rows, cols), dtype=np.float32)

    # strutture dati
    predictions_sup = []
    predictions_inf = []
    GT_sup = []
    GT_inf = []
    ist = []

    # carico nomi sequenza test
    test_data_names = load_names_val()

    # carico sequenza test
    test_data_X, Y = load_images(test_data_names, crop=b_crop, rows=rows, cols=cols)

    # PREDIZIONE
    print("Prediction...")
    pred = model.predict(x=test_data_X, batch_size=batch_size, verbose=0)
    print("Done.")


    for i, element in enumerate(pred):

        ist.append(i)

        # salvataggio
        GT_sup.append(int(Y[i][0]*1024))
        GT_inf.append(int(Y[i][1]*1024))
        predictions_sup.append(int(element[0]*1024))
        predictions_inf.append(int(element[1]*1024))

        #print "Predizione:", pred[0][0], pred[0][1]
        #print "GT:", Y[0][0], Y[0][1]

        # apro immagine originale

        if b_visualize:
            img = cv2.imread(test_data_names[i]['image'], 0)

            cv2.line(img, (0,int(element[0]*1024)), (770, int(element[0]*1024)), 255)
            cv2.line(img, (0,int(element[1]*1024)), (770, int(element[1]*1024)), 255)

            cv2.imshow("Predizione grafica", cv2.resize(img, dsize=None, fx = 0.5, fy=0.5))
            cv2.waitKey()

    if b_plot:
        fig = plt.figure()

        plt.plot(ist, GT_sup, 'r')
        plt.plot(ist, GT_inf, 'b')

        plt.plot(ist,predictions_sup, 'k' )
        plt.plot(ist, predictions_inf, 'k')

        plt.show()
