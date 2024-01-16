from ultralytics import YOLO

# Load a model
model = YOLO("YOLOv8n.pt")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=2, batch=2)  # train the model