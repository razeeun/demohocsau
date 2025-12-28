from ultralytics import YOLO
import cv2
import numpy as np

def load_model(path):
    """Load YOLO models."""
    try:
        model = YOLO(path)
        print(f"✅ Model loaded successfully from {path}")
        return model
    except Exception as e:
        print(f"❌ Không thể load model từ {path}: {e}")
        return None

def predict_image(model, image, conf_threshold=0.25):
    """
    Nhận dạng ảnh với ngưỡng confidence.
    
    Args:
        model: YOLO model
        image: PIL Image hoặc numpy array
        conf_threshold: Ngưỡng confidence (0.0 - 1.0)
    
    Returns:
        result_img: Ảnh đã được annotate (numpy array)
        preds: List các predictions [x1, y1, x2, y2, confidence, class]
    """
    try:
        # Chạy inference với confidence threshold
        results = model(image, conf=conf_threshold)
        
        # Vẽ kết quả lên ảnh
        result_img = results[0].plot()  # numpy array
        
        # Lấy predictions
        preds = results[0].boxes.data.tolist()
        
        return result_img, preds
    
    except Exception as e:
        print(f"❌ Lỗi khi predict image: {e}")
        return None, []

def predict_video(model, video_path, conf_threshold=0.25, output_path="output_result.mp4"):
    """
    Nhận dạng video với ngưỡng confidence.
    
    Args:
        model: YOLO model
        video_path: Đường dẫn video đầu vào
        conf_threshold: Ngưỡng confidence (0.0 - 1.0)
        output_path: Đường dẫn video đầu ra
    
    Returns:
        output_path: Đường dẫn video đã xử lý
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Chạy inference
            results = model(frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # Ghi frame
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Log mỗi 30 frames
                print(f"⏳ Processed {frame_count}/{total_frames} frames")
        
        # Giải phóng resources
        cap.release()
        out.release()
        
        print(f"✅ Video processed successfully: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"❌ Lỗi khi predict video: {e}")
        return None

def get_model_info(model):
    """
    Lấy thông tin về model.
    
    Args:
        model: YOLO model
    
    Returns:
        dict: Thông tin model
    """
    try:
        info = {
            'names': model.names,  # Class names
            'num_classes': len(model.names),
        }
        return info
    except Exception as e:
        print(f"❌ Không thể lấy thông tin model: {e}")
        return None

def batch_predict(model, image_list, conf_threshold=0.25):
    """
    Nhận dạng nhiều ảnh cùng lúc.
    
    Args:
        model: YOLO model
        image_list: List các PIL Images
        conf_threshold: Ngưỡng confidence
    
    Returns:
        results: List các tuples (result_img, preds)
    """
    results = []
    
    for idx, image in enumerate(image_list):
        try:
            result_img, preds = predict_image(model, image, conf_threshold)
            results.append((result_img, preds))
            print(f"✅ Processed image {idx+1}/{len(image_list)}")
        except Exception as e:
            print(f"❌ Error processing image {idx+1}: {e}")
            results.append((None, []))
    
    return results