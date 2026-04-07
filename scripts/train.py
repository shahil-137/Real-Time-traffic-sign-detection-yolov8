# from ultralytics import YOLO

# if __name__ == "__main__":
#     model = YOLO("runs/detect/traffic_sign_25cls/weights/last.pt")

#     model.train(
#         data="dataset/yolo/data.yaml",
#         epochs=10,        # TOTAL epochs (5 → 10)
#         imgsz=640,
#         batch=4,
#         device="cuda",
#         workers=0,
#         resume=False,
#         name="traffic_sign_25cls"
#     )
# from ultralytics import YOLO

# if __name__ == "__main__":
#     model = YOLO("runs/detect/traffic_sign_25cls/weights/last.pt")

#     model.train(
#         data="dataset/yolo/data.yaml",
#         epochs=25,        # TOTAL epochs
#         imgsz=640,
#         batch=4,
#         device="cuda",
#         workers=0,
#         resume=False,
#         name="traffic_sign_25cls"
#     )

# from ultralytics import YOLO

# if __name__ == "__main__":
#     model = YOLO("runs/detect/traffic_sign_25cls3/weights/best.pt")

#     model.train(
#         data="dataset/yolo/data.yaml",
#         epochs=55,          # TOTAL epochs (25 → 55)
#         imgsz=704,          # multi-scale enhancement
#         batch=4,            # safe for R
#         device="cuda",
#         workers=0,
#         resume=False,
#         name="traffic_sign_25cls704"
#     )

from ultralytics import YOLO

if __name__ == "__main__":
    # Load the yolov8s.pt weights
    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset/yolo/data.yaml",
        epochs=55,              
        batch=4,        
        device="cuda",
        workers=0,
        resume=False,
        name="traffic_sign_25cls704"
    )