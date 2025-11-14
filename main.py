from ultralytics import YOLO

model = YOLO('pin_detector.pt')

results = model.predict('output', save=True, show_conf=False, show_labels=False)