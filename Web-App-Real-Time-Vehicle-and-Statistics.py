from flask import Flask, render_template, send_from_directory, jsonify
import os
import cv2
import supervision as sv
from ultralytics import YOLOv10
import numpy as np
import csv
import time
import pandas as pd
from threading import Thread, Event
import signal
from vidgear.gears import WriteGear

app = Flask(__name__)

stop_event = Event()

model = YOLOv10("../models/model-b-v10.pt")
bounding_box_annotator = sv.BoundingBoxAnnotator()

stream_url = 'https://bjb-stream.rapidnetwork.id/3S1McVqImbNEkIS52MzxpDaeuzFs90/hls/VKfQs67iU7/camera71/s.m3u8'
output_hls_folder = './static/hls15'
os.makedirs(output_hls_folder, exist_ok=True)
hls_output_path = os.path.join(output_hls_folder, 'output.m3u8')

count_csv = './data/count_vehicles15.csv'
speed_csv = './data/speed_data15.csv'
crop_folder = './img/image_detection15'
os.makedirs(crop_folder, exist_ok=True)

def draw_source_box(frame: np.ndarray, source: np.ndarray) -> np.ndarray:
    for i in range(len(source)):
        start_point = tuple(source[i])
        end_point = tuple(source[(i + 1) % len(source)])
        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
    return frame

def estimate_speed(track, fps, transformation_matrix):
    if len(track) < 2:
        return 0

    track_array = np.array(track, dtype=np.float32).reshape(-1, 1, 2)
    transformed_track = cv2.perspectiveTransform(track_array, transformation_matrix).reshape(-1, 2)

    distances = np.linalg.norm(transformed_track[1:] - transformed_track[:-1], axis=1)
    total_distance = np.sum(distances)

    time_diff = (len(track) - 1) / fps

    if time_diff > 0:
        speed = (total_distance / time_diff) * 3.6 * 0.1  # Adjust the multiplier to calibrate speed
    else:
        speed = 0

    return speed

def save_crop(frame, xyxy, tracker_id, class_name, frame_count):
    if frame_count % 5 == 0:  
        x1, y1, x2, y2 = map(int, xyxy)
        crop = frame[y1:y2, x1:x2]
        crop_filename = os.path.join(crop_folder, f"{tracker_id}_{class_name}_{frame_count}.jpg")
        cv2.imwrite(crop_filename, crop)

def stream_and_detect():
    cap = cv2.VideoCapture(stream_url)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count2 = 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_params = {
        "-input_framerate": fps,
        '-s': f'{width}x{height}',
        "-vcodec": "libx264",
        "-preset": "ultrafast",
        "-tune": "zerolatency",
        "-f": "hls",
        "-hls_time": "2",
        "-hls_list_size": "0",
        "-hls_flags": "append_list",
        '-hls_segment_filename': os.path.join(output_hls_folder, 'segment_%03d.ts'),
        "-g": "60"
    }
    writer_hls = WriteGear(
        output=hls_output_path, 
        compression_mode=True, 
        logging=True, 
        **output_params
    )

    # Define source and target points for perspective transformation
    SOURCE = np.array([
        [5, 327],  # Top-left
        [631, 327],  # Top-right
        [631, 57],   # Bottom-right
        [183, 57]   # Bottom-left
    ])

    TARGET = np.array([
        [0, 0],     # Top-left
        [24, 0],    # Top-right
        [24, 249],  # Bottom-right
        [0, 249]    # Bottom-left
    ])

    matrix = cv2.getPerspectiveTransform(SOURCE.astype(np.float32), TARGET.astype(np.float32))

    START = sv.Point(0, 200)
    END = sv.Point(640, 200)
    line_zone = sv.LineZone(start=START, end=END)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

    byte_tracker = sv.ByteTrack()
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_padding=5)
    trace_annotator = sv.TraceAnnotator(thickness=1)
    
    tracks = {}
    frame_count = 0
    unique_vehicles = {'car': set(), 'bus': set(), 'truck': set(), 'motorcycle': set()}
    saved_data = set()

    if not os.path.exists(count_csv):
        with open(count_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Time', 'In', 'Out', 'Car Count', 'Bus Count', 'Truck Count', 'Motorcycle Count', 'Total'])

    if not os.path.exists(speed_csv):
        with open(speed_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Class', 'X', 'Y', 'Speed'])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or error reading frame")
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = byte_tracker.update_with_detections(detections)
        # draw_points_on_frame(frame, SOURCE)
        
        frame_count += 1

        labels = []
        for xyxy, confidence, class_id, tracker_id in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            
            if tracker_id not in tracks:
                tracks[tracker_id] = []
            tracks[tracker_id].append(center)

            if len(tracks[tracker_id]) > 30: 
                tracks[tracker_id] = tracks[tracker_id][-30:]
            
            if len(tracks[tracker_id]) >= 2:
                speed = estimate_speed(tracks[tracker_id], fps, matrix)
            else:
                speed = 0

            vehicle_type = model.names[int(class_id)]
            
            if tracker_id not in saved_data and speed > 0:
                with open(speed_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([tracker_id, vehicle_type, center[0], center[1], speed])
                saved_data.add(tracker_id)
                save_crop(frame, xyxy, tracker_id, vehicle_type, frame_count2)

            label = f"ID:{tracker_id} {vehicle_type}: {speed:.1f} km/h"
            labels.append(label)

            if vehicle_type in unique_vehicles:
                unique_vehicles[vehicle_type].add(tracker_id)

        line_zone.trigger(detections=detections)
        annotated_frame = frame.copy()
        # annotated_frame = draw_source_box(annotated_frame, SOURCE)
        annotated_frame = line_zone_annotator.annotate(annotated_frame, line_zone)
        annotated_frame = trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        writer_hls.write(annotated_frame)

        if frame_count % 5 == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            in_count = line_zone.in_count
            out_count = line_zone.out_count
            car_count = len(unique_vehicles['car'])
            bus_count = len(unique_vehicles['bus'])
            truck_count = len(unique_vehicles['truck'])
            motorcycle_count = len(unique_vehicles['motorcycle'])
            total_count = car_count + bus_count + truck_count + motorcycle_count

            with open(count_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_count // 5, current_time, in_count, out_count, car_count, bus_count, truck_count, motorcycle_count, total_count])

    cap.release()
    writer_hls.close()

def read_count_csv():
    df = pd.read_csv('./data/count_vehicles15.csv')
    latest_data = df.iloc[-1]
    
    return {
        'In': int(latest_data['In']),
        'Out': int(latest_data['Out']),
        'Car Count': int(latest_data['Car Count']),
        'Bus Count': int(latest_data['Bus Count']),
        'Truck Count': int(latest_data['Truck Count']),
        'Motorcycle Count': int(latest_data['Motorcycle Count']),
        'timestamp': latest_data['Time'] if 'Time' in latest_data else pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def read_speed_csv():
    df = pd.read_csv('./data/speed_data15.csv')  
    avg_speeds = df.groupby('Class')['Speed'].mean().to_dict()
    return avg_speeds

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/data')
def data():
    count_data = read_count_csv()
    speed_data = read_speed_csv()
    combined_data = {**count_data, 'avg_speeds': speed_data}
    return jsonify(combined_data)

@app.route('/hls/<path:filename>')
def serve_hls(filename):
    return send_from_directory(output_hls_folder, filename)

def signal_handler(signum, frame):
    print("Interrupt received, stopping threads...")
    stop_event.set()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    stream_thread = Thread(target=stream_and_detect)
    stream_thread.start()

    try:
        app.run(debug=False, use_reloader=False, threaded=True)
    finally:
        print("Stopping application...")
        stop_event.set()
        stream_thread.join()
        print("Application stopped.")