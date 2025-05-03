import cv2
import pafy
from vidgear.gears import CamGear
import time

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

amount_of_points =  []

tracker = DeepSort(
        max_age=60,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
        gating_only_position=False,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True,
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None,
)

# Load the YOLO11 model
model = YOLO("yolo11n.pt")
#model = YOLO("yolov8n.pt")

#speedestimator = solutions.SpeedEstimator(
#    region=[(134, 467), (638, 15 )],
    #model="yolo11n.pt",
#    show=True,
#)

# Open the video file
#video_path = "/dev/video4"
#video_path = "/home/david/Videos/FILE240306-072040F.MOV"
#video_path = "Videos/FILE240306-072040F.MOV"
url = "https://www.youtube.com/live/g9hNGJxw6Yw?si=wz1JulzVqTYxPV0x"
#video = pafy.new(url)
#best = video.getbest(preftype="mp4")
options = {"STREAM_RESOLUTION": "720p"}
cap = CamGear(
    source=url, stream_mode=True, logging=True, **options
).start()

def calculate_speed(car_speed,fps,point_distance):
    car_distance = abs(int(car_speed[2]) - int(car_speed[0]))
    print(f'car distance {car_speed[2]} {car_speed[0]}')
    point_distanced = abs(int(point_distance[1][0]) - int(point_distance[0][0]))
    print(f'point distance {point_distanced}')
    fps = float(fps)
    distance = (point_distanced / car_distance) * 4
    return  distance / fps 

def capture_pointer(event,x,y,flags,params):
    global amount_of_points
    if event == cv2.EVENT_LBUTTONDOWN:
        amount_of_points.append((x,y))
    elif event == cv2.EVENT_LBUTTONUP:
        amount_of_points.append((x,y))

#cap = cv2.VideoCapture(video_path)
cv2.namedWindow("image")
cv2.setMouseCallback("image", capture_pointer)
#tframe = cap.get(cv2.CAP_PROP_FRAME_COUNT) # get total frame count
#fps = cap.get(cv2.CAP_PROP_FPS)  #get the FPS of the videos

def drawbox(data,image,map):
    x1,y1,x2,y2,conf,id = data
    p1 = (int(x1),int(y1))
    cv2.putText(image,map[id],p1,cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),3) 
    pass
trackPoint = {}
count = 0
while True:
    #success, frame = cap.read()
    frame = cap.read()
    count+=1
    #if count % 3 !=0:
    #    continue
    #cframe = cap.get(cv2.CAP_PROP_POS_FRAMES) # retrieves the current frame number
    #if not success:
    #    break
    results = model(frame)
    result_list = []
    for r in results:
        for data in r.boxes.data.tolist():
            class_id = int(data[5])
            if class_id not in [2,3,7]:
                continue
            drawbox(data,frame,results[0].names) 
            confidence = float(data[4])
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            result_list.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])


        
# Loop through the video frames
    i = 0
    while i < len(amount_of_points)-1:
        print(len(amount_of_points))
        if len(amount_of_points) % 2 == 0:
            cv2.rectangle(frame,amount_of_points[i],amount_of_points[i+1],(0,0,255))
            i = i + 2
        elif len(amount_of_points) % 2 != 0:
            break
    car = {}
    tracks = tracker.update_tracks(result_list,frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()
        car[track.track_id] = {}
        car[track.track_id]['coord'] = bbox
        cv2.putText(frame, "     " +str(track.track_id), (int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),3)
    
    if len(amount_of_points) > 3:
        print(f'{amount_of_points[0][0]} {amount_of_points[3][0]}')

        amount_start = amount_of_points[:2]
        for key,item in car.items():
            for lockey, coord in item.items():
                if lockey == "coord":
                    if coord[0] >= amount_start[0][0] and \
                        coord[1] >= amount_start[0][1] and \
                        coord[2] <= amount_start[1][0] and \
                        coord[3] <= amount_start[1][1]:
           
                        if key in trackPoint and 'start2' in trackPoint[key] and  'end' not in  trackPoint[key]:
                            trackPoint[key]['end'] = time.time()
                        elif key not in trackPoint:
                             trackPoint[key] = {}
                             trackPoint[key]['start1'] = time.time()
                        if 'start2' in trackPoint[key] and 'end' in trackPoint[key]:
                           speed = str(calculate_speed(coord,trackPoint[key]['end'] - trackPoint[key]['start2'],amount_of_points))
                           print("Speed" + speed)

                           cv2.putText(frame, speed, (int(coord[0]),int(coord[1])),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),3)

        amount_start = amount_of_points[2:4]
        for key,item in car.items():
            for lockey, coord in item.items():
                if lockey == "coord":
                    if coord[0] >= amount_start[0][0] and \
                        coord[1] >= amount_start[0][1] and \
                        coord[2] <= amount_start[1][0] and \
                        coord[3] <= amount_start[1][1]:
                        if key in trackPoint and 'start1' in trackPoint[key] and 'end' not in  trackPoint[key]:
                            trackPoint[key]['end'] = time.time()
                        elif key not in trackPoint:
                             trackPoint[key] = {}
                             trackPoint[key]['start2'] = time.time()

                        if 'start1' in trackPoint[key] and 'end' in trackPoint[key]:
                            speed = str(calculate_speed(coord,trackPoint[key]['end'] - trackPoint[key]['start1']    ,amount_of_points))
                            print("Speed" + speed)
                            cv2.putText(frame, speed , (int(coord[0]),int(coord[1])),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),3)

    cv2.imshow('image',frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break



# Release the video capture object and close the display window
#cap.release()
cv2.destroyAllWindows()
