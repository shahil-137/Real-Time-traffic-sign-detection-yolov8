# from ultralytics import YOLO

# if __name__ == "__main__":

#     model = YOLO("runs/detect/traffic_sign_25cls704_finetune/weights/best.pt")

#     model.train(
#         data="dataset/yolo/data.yaml",
#         epochs=15,            # short refinement
#         imgsz=960,            # higher resolution for digit clarity
#         batch=2,              # safe for RTX 3050
#         mosaic=0.5,           # reduce distortion
#         mixup=0.0,            # disable mixup
#         device=0,
#         name="traffic_sign_25cls960_15ep"
#     )


# from ultralytics import YOLO

# if __name__ == "__main__":

#     model = YOLO("runs/detect/traffic_sign_25cls960_15ep/weights/best.pt")

#     model.train(
#         data="dataset/yolo/data.yaml",
#         epochs=25,
#         imgsz=1024,
#         batch=2,        # IMPORTANT for RTX 3050
#         mosaic=0.5,
#         mixup=0.0,
#         device=0,
#         name="traffic_sign_25cls1024_finetune25ep"
#     )
from ultralytics import YOLO
if __name__ == "__main__":

    model = YOLO("runs/detect/traffic_sign_25cls704/weights/best.pt")

    model.train(
        data="dataset/yolo/data.yaml",
        epochs=25,
        imgsz=1024,
        batch=2,        # IMPORTANT for RTX 3050
        mosaic=0.5,
        mixup=0.0,
        device=0,
        name="traffic_sign_25cls1024_finetune25ep"
    )