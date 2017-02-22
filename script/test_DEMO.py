import sys
import os
# SETTAGGIO SCHEDA GRAFICA
if len(sys.argv) == 3:
    gpu_id = sys.argv[1]
    cnmem = sys.argv[2]
else:
    gpu_id = "cpu"
    cnmem = "0.7"
print("Argument: gpu={}, mem={}".format(gpu_id, cnmem))
os.environ["THEANO_FLAGS"] = "device=" + gpu_id + ", lib.cnmem=" + cnmem

from batch_generators import load_names, load_images, load_images_test
from models import foamNet
import numpy as np
import cv2


if __name__ == '__main__':


    rows = 256
    cols = 192
    ch = 1

    # MODELLO
    model = foamNet(rows, cols)

    # PESI
    print("Load weights ...")
    model.load_weights('weights/weights.006-428823.61560.hdf5')

    # img_batch = np.zeros(shape=(1, ch, rows, cols), dtype=np.float32)

    # per ora su train. to do: fare test
    test_data_names = load_names()

    for element in test_data_names:
        X, Y = load_images_test(element, rows, cols)

        pred = model.predict(x=X, batch_size=1, verbose=0)

        print "Predizione:", pred[0][0], pred[0][1]

        # apro immagine originale
        img = cv2.imread(element['image'], 0)

        cv2.line(img, (0,int(pred[0][0])), (770, int(pred[0][0])), 255)
        cv2.line(img, (0,int(pred[0][1])), (770, int(pred[0][1])), 255)

        cv2.imshow("Predizione grafica", cv2.resize(img, dsize=None, fx = 0.5, fy=0.5))
        cv2.waitKey()



    # while True:
    #     kinect.get_frame(frame)
    #
    #     background = np.ones((1080, 1920, 3))
    #
    #     if frame.frameRGB is None:
    #         continue
    #
    #     # rgb = cv2.resize(src=frame.frameRGB, dsize=None, fx=0.8, fy=0.8)
    #     rgb = frame.frameRGB.copy()
    #     rgb = rgb[:, :, :-1]
    #
    #     face, l_eye, r_eye, f_g = detectFaceandEye(rgb, scale=0.25)
    #
    #
    #     if face == None or r_eye == None or l_eye == None or f_g == None:
    #         continue
    #
    #     else:
    #
    #         l_eye_batch[0] = cv2.resize(l_eye, (112, 112)).transpose(2, 0, 1)
    #         r_eye_batch[0] = cv2.resize(r_eye, (112, 112)).transpose(2, 0, 1)
    #         face_batch[0] = cv2.resize(face, (112, 112)).transpose(2, 0, 1)
    #         face_grid_batch[0, 0] = cv2.resize(src=f_g, dsize=(25, 25), interpolation=cv2.INTER_NEAREST)
    #
    #
    #     # PREDIZIONE
    #     pred = model.predict(x=[l_eye_batch, r_eye_batch, face_batch, face_grid_batch], batch_size=1, verbose=0)
    #
    #     print "Predizione:", pred[0][0], pred[0][1]
    #     cv2.circle(background, (int(pred[0][0]), int(pred[0][1])), 50, (0, 0, 255), thickness=-1)
    #
    #
    #     cv2.imshow("dot", background)
    #     cv2.waitKey(1)
    #     # cv2.imshow("RGB", rgb)
    #     # cv2.waitKey(1)

cv2.destroyAllWindows()