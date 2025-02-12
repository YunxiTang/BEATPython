import os, cv2
from fastdlo.core import Pipeline
import matplotlib.pyplot as plt


if __name__ == "__main__":

    ######################
    # IMG_PATH = "/media/yxtang/Extreme SSD/fastdlo/test_images/8.jpg"
    IMG_W = 640
    IMG_H = 480
    ######################

    checkpoint_siam = "/media/yxtang/Extreme SSD/fastdlo/weights/CP_similarity.pth"
    checkpoint_seg = "/media/yxtang/Extreme SSD/fastdlo/weights/CP_segmentation.pth"
    
    p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_W, img_h=IMG_H)

    cap = cv2.VideoCapture('/media/yxtang/Extreme SSD/DOM_Reaseach/dobert_cache/dobert_dataset/dom_dataset/episode_4009/episode_4009.mp4')

    while True:
        ret, source_img = cap.read()
        if source_img is None:
            break
        
        # COLOR
        # source_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        source_img = cv2.resize(source_img, (IMG_W, IMG_H))

        img_out, _, data_res = p.run(source_img=source_img, mask_th=77)
        
        canvas = source_img.copy()
        canvas = cv2.addWeighted(canvas, 0.7, img_out, 1.0, 0.0)

        for i in range(len(data_res['nodes'])-1):
            pos = data_res['nodes'][i]['pos']
            pos_next = data_res['nodes'][i+1]['pos']
            image = cv2.circle(canvas, (pos[1], pos[0]), radius=2, color=(0, 0, 255), thickness=-1)
            # plt.plot([pos[1], pos_next[1]], [pos[0], pos_next[0]])
        cv2.imshow("output", canvas)
        cv2.waitKey(10)
        # plt.show()
        # plt.axis('equal')
        

