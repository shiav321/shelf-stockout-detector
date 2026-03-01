"""
main.py — Shelf Stock-Out Detector
Real-time detection using webcam or video file
"""

import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import csv
import os

# ── CONFIG ──────────────────────────────────────
MODEL_PATH = "model/shelf_model.h5"
LOG_PATH   = "logs/detections.csv"
THRESHOLD  = 0.75   # confidence threshold
IMG_SIZE   = (224, 224)

# Colors
RED    = (0, 0, 255)
GREEN  = (0, 200, 0)
YELLOW = (0, 200, 255)
WHITE  = (255, 255, 255)


# ── LOAD MODEL ──────────────────────────────────
def load_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Loading trained model...")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("⚠️  No trained model found. Run train.py first!")
        print("    Using rule-based detection as fallback...")
        return None


# ── RULE-BASED DETECTION (no model needed) ──────
def rule_based_detection(frame):
    """
    Detects empty shelf areas using computer vision rules.
    Works WITHOUT training a model — great for demo!
    
    Logic:
      Empty shelves → more uniform color (wall/shelf background visible)
      Stocked shelves → more texture, color variance (products visible)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Divide frame into grid zones
    h, w = gray.shape
    rows, cols = 3, 4
    zone_h = h // rows
    zone_w = w // cols
    
    results = []
    
    for r in range(rows):
        for c in range(cols):
            y1 = r * zone_h
            y2 = y1 + zone_h
            x1 = c * zone_w
            x2 = x1 + zone_w
            
            zone = gray[y1:y2, x1:x2]
            
            # Texture analysis
            # High stddev = lots of products (stocked)
            # Low stddev = empty shelf (uniform background)
            std = np.std(zone)
            mean = np.mean(zone)
            
            # Edge density
            edges = cv2.Canny(zone, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Score: higher = more likely stocked
            stock_score = (std / 128.0) * 0.6 + edge_density * 10 * 0.4
            stock_score = min(stock_score, 1.0)
            
            is_empty = stock_score < 0.35
            
            results.append({
                'zone': (r, c),
                'bbox': (x1, y1, x2, y2),
                'score': stock_score,
                'empty': is_empty,
                'confidence': (1 - stock_score) if is_empty else stock_score
            })
    
    return results


# ── MODEL-BASED DETECTION ───────────────────────
def model_detection(frame, model):
    """Uses trained CNN model for detection"""
    h, w = frame.shape[:2]
    rows, cols = 3, 4
    zone_h = h // rows
    zone_w = w // cols
    results = []
    
    for r in range(rows):
        for c in range(cols):
            y1 = r * zone_h
            y2 = y1 + zone_h
            x1 = c * zone_w
            x2 = x1 + zone_w
            
            zone = frame[y1:y2, x1:x2]
            zone_resized = cv2.resize(zone, IMG_SIZE)
            zone_array = np.expand_dims(zone_resized / 255.0, axis=0)
            
            pred = model.predict(zone_array, verbose=0)[0][0]
            # pred → 0 = empty, 1 = stocked
            
            is_empty = pred < THRESHOLD
            results.append({
                'zone': (r, c),
                'bbox': (x1, y1, x2, y2),
                'score': float(pred),
                'empty': is_empty,
                'confidence': float(1 - pred) if is_empty else float(pred)
            })
    
    return results


# ── DRAW RESULTS ────────────────────────────────
def draw_results(frame, results):
    overlay = frame.copy()
    empty_count = 0
    
    for zone in results:
        x1, y1, x2, y2 = zone['bbox']
        is_empty = zone['empty']
        conf = zone['confidence']
        
        if is_empty:
            empty_count += 1
            color = RED
            label = f"EMPTY {conf*100:.0f}%"
            # Semi-transparent red fill
            cv2.rectangle(overlay, (x1,y1), (x2,y2), RED, -1)
        else:
            color = GREEN
            label = f"OK {conf*100:.0f}%"
        
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1+6, y1+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    # Dashboard panel
    total = len(results)
    stocked = total - empty_count
    alert = empty_count > 2
    
    panel_color = (0, 0, 180) if alert else (0, 120, 0)
    cv2.rectangle(frame, (0, 0), (340, 100), panel_color, -1)
    cv2.rectangle(frame, (0, 0), (340, 100), WHITE, 1)
    
    cv2.putText(frame, "SHELF STOCK-OUT DETECTOR",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    cv2.putText(frame, f"Stocked: {stocked}/{total} zones",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100,255,100), 2)
    cv2.putText(frame, f"Empty:   {empty_count}/{total} zones",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100,100,255), 2)
    
    if alert:
        cv2.putText(frame, "⚠ ALERT: RESTOCK NEEDED!",
                    (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)
    
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts, (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    
    return frame, empty_count


# ── LOG DETECTION ───────────────────────────────
def log_detection(empty_count, total_zones):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            empty_count,
            total_zones,
            f"{(empty_count/total_zones)*100:.1f}%"
        ])


# ── MAIN LOOP ────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("  SHELF STOCK-OUT DETECTOR")
    print("  By: Shiva Keshava | Data Science Project")
    print("="*50 + "\n")

    model = load_model()
    
    # Use 0 for webcam, or replace with video file path
    # e.g. cap = cv2.VideoCapture("store_footage.mp4")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera!")
        return
    
    print("✅ Camera started! Press Q to quit, S to save screenshot\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection every 15 frames (for performance)
        if frame_count % 15 == 0:
            if model:
                results = model_detection(frame, model)
            else:
                results = rule_based_detection(frame)
            
            frame, empty_count = draw_results(frame, results)
            
            # Log if alert
            if empty_count > 2:
                log_detection(empty_count, len(results))
                print(f"⚠️  ALERT [{datetime.now().strftime('%H:%M:%S')}] "
                      f"{empty_count} empty zones detected!")
        
        cv2.imshow("Shelf Stock-Out Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(fname, frame)
            print(f"📸 Screenshot saved: {fname}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Detector stopped. Logs saved to logs/detections.csv")


if __name__ == "__main__":
    main()
