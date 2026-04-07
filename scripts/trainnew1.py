# from ultralytics import YOLO

# if __name__ == "__main__":

#     model = YOLO("runs/detect/traffic_sign_25cls960_15ep/weights/best.pt")

#     model.train(
#         data="dataset/yolo/data.yaml",
#         epochs=10,
#         imgsz=1024,
#         batch=2,        # IMPORTANT for RTX 3050
#         mosaic=0.5,
#         mixup=0.0,
#         device=0,
#         name="traffic_sign_25cls1024_finetune"
#     )
# from ultralytics import YOLO

# 

from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("runs/detect/traffic_sign_25cls1024_finetune25ep/weights/best.pt")

    model.train(
        data="dataset/yolo/data.yaml",

        # Small controlled fine-tuning
        epochs=20,                 # not too high
        patience=7,                # early stopping (if no val improvement)

        imgsz=1024,
        batch=2,

        # Stronger augmentation (helps generalization)
        mosaic=0.8,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        scale=0.5,
        fliplr=0.5,

        # Regularization
        weight_decay=0.0008,       # slightly stronger

        # Smooth LR schedule
        cos_lr=True,
        lr0=0.008,                 # slightly lower than default

        device=0,
        name="traffic_sign_gap_reduction_final"
    )