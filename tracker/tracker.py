from ultralytics import YOLO
import supervision as sv
import pickle as pk
import os
from utils.bbox_utils import *
import cv2
import numpy as np
import pandas as pd


class Tracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        

    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    # estimating the ball path, fixing the issue of it not being detected constantly
    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range (0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections.extend(detections_batch)
        return detections       

    def getObjectTracks(self, frames, read_from=False, path=None):
        if read_from and path is not None and os.path.exists(path):
            with open(path, 'rb') as f:
                tracks = pk.load(f)
            return tracks    

        detections = self.detect_frames(frames)
        
        tracks = {
            'player':[],
            'ball':[],
            'referee':[],
            'goalkeeper':[]
        }
        for frame_num, detection in enumerate(detections):

            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            #convert to supervision format
            detections_sv = sv.Detections.from_ultralytics(detection)

            #convert GK to normal Player
            # for objec_index, detection in enumerate(detections_sv):
            #     class_id = detection[3]
            #     if cls_names[class_id] == "goalkeeper": 
            #         detections_sv.class_id[objec_index] = cls_names_inv['player']

            #track objects
            for track_category in ['player', 'referee', 'ball', 'goalkeeper']:
                while len(tracks[track_category]) <= frame_num:
                    tracks[track_category].append({})

            detections_sv_with_tracking = self.tracker.update_with_detections(detections_sv)

            for detection_with_track in detections_sv_with_tracking:
                bbox = detection_with_track[0].tolist()
                cls_id = detection_with_track[3]
                track_id = detection_with_track[4]

                if cls_id == cls_names_inv['player']:
                    tracks['player'][frame_num][track_id] = {'bbox': bbox}
                elif cls_id == cls_names_inv['referee']:
                    tracks['referee'][frame_num][track_id] = {'bbox': bbox}
                elif cls_id == cls_names_inv['goalkeeper']:
                    tracks['goalkeeper'][frame_num][track_id] = {'bbox': bbox}

            # Handle ball detections (no tracking)
            for detection in detections_sv:
                bbox = detection[0].tolist()
                cls_id = detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}



        if path is not None : 
            with open(path, 'wb') as f:
                pk.dump(tracks, f)
        return tracks

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_rounded_rectangle(self,img, pt1, pt2, color, thickness, radius):
        """
        Draw a rounded rectangle on img from pt1 to pt2.
        If thickness < 0, fill the shape.
        """
        x1, y1 = pt1
        x2, y2 = pt2
        if thickness < 0:
            # Filled rounded rectangle
            # Create a mask for the filled rounded rectangle
            mask = np.zeros_like(img, dtype=np.uint8)
            # Draw four filled circles for the corners
            cv2.circle(mask, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(mask, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(mask, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(mask, (x2 - radius, y2 - radius), radius, color, -1)
            # Draw filled rectangles to connect the circles
            cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            # Combine the mask with the image (assumes img is BGR)
            idx = mask != 0
            img[idx] = mask[idx]
        else:
            # Outline rounded rectangle: draw multiple segments
            # Top and bottom horizontal lines
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            # Left and right vertical lines
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
            # Four arcs for the corners
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


    def draw_ball_control(self, frame, frame_num, team_ball_control, team_colors):
        panel_top_left = (1350, 820)
        panel_bottom_right = (1900, 1000)  # Extended height
        panel_color = (255, 255, 255)  # White background for the panel
        panel_radius = 30
        
        # Create an overlay and draw a filled, rounded rectangle for the panel
        overlay = frame.copy()
        self.draw_rounded_rectangle(overlay, panel_top_left, panel_bottom_right, panel_color, -1, panel_radius)
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Calculate ball control percentages up to the current frame
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_frames = np.sum(team_ball_control_till_frame == 2)
        total_frames = team_1_frames + team_2_frames if (team_1_frames + team_2_frames) > 0 else 1
        team_1_percentage = team_1_frames / total_frames
        team_2_percentage = team_2_frames / total_frames

        # Panel dimensions
        panel_x1, panel_y1 = panel_top_left
        panel_x2, panel_y2 = panel_bottom_right
        panel_width = panel_x2 - panel_x1
        panel_height = panel_y2 - panel_y1

        # Define bar dimensions and spacing
        bar_height = 40
        bar_spacing = 10
        total_bars_height = 2 * bar_height + bar_spacing
        vertical_margin = (panel_height - total_bars_height - 50) // 2  # Extra space for "Possession" text

        # Horizontal margin for centering bars
        horizontal_margin = 50
        bar_width = panel_width - 2 * horizontal_margin

        # Compute positions for the bars
        team1_bar_top_left = (panel_x1 + horizontal_margin, panel_y1 + vertical_margin + 50)
        team1_bar_bottom_right = (team1_bar_top_left[0] + int(bar_width * team_1_percentage), team1_bar_top_left[1] + bar_height)
        team2_bar_top_left = (panel_x1 + horizontal_margin, team1_bar_top_left[1] + bar_height + bar_spacing)
        team2_bar_bottom_right = (team2_bar_top_left[0] + int(bar_width * team_2_percentage), team2_bar_top_left[1] + bar_height)        

        
        # Colors for the progress bars (you can choose any modern palette)
        team1_color = team_colors[1]  # Blueish
        team2_color = team_colors[2]   # Reddish
        
        # Draw filled progress bars
        cv2.rectangle(frame, team1_bar_top_left, team1_bar_bottom_right, team1_color, -1)
        cv2.rectangle(frame, team2_bar_top_left, team2_bar_bottom_right, team2_color, -1)

        # Draw bar outlines
        outline_color = (220, 220, 220)
        cv2.rectangle(frame, team1_bar_top_left, (team1_bar_top_left[0] + bar_width, team1_bar_top_left[1] + bar_height), outline_color, 2)
        cv2.rectangle(frame, team2_bar_top_left, (team2_bar_top_left[0] + bar_width, team2_bar_top_left[1] + bar_height), outline_color, 2)

        # Function to draw text with shadow
        def put_shadow_text(img, text, org, font_scale, color, thickness):
            cv2.putText(img, text, (org[0] + 2, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        # Draw "Possession" centered above the bars
        possession_text = "Possession"
        text_size, _ = cv2.getTextSize(possession_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        possession_text_org = (panel_x1 + (panel_width - text_size[0]) // 2, team1_bar_top_left[1] - 20)
        put_shadow_text(frame, possession_text, possession_text_org, 1.2, (0, 0, 0), 3)

        # Draw possession percentage centered inside each bar
        team1_percentage_text = f"{team_1_percentage * 100:.1f}%"
        team2_percentage_text = f"{team_2_percentage * 100:.1f}%"

        team1_text_size, _ = cv2.getTextSize(team1_percentage_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        team2_text_size, _ = cv2.getTextSize(team2_percentage_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        team1_text_org = (team1_bar_top_left[0] + (bar_width - team1_text_size[0]) // 2, team1_bar_top_left[1] + bar_height - 5)
        team2_text_org = (team2_bar_top_left[0] + (bar_width - team2_text_size[0]) // 2, team2_bar_top_left[1] + bar_height - 5)

        put_shadow_text(frame, team1_percentage_text, team1_text_org, 1, (255, 255, 255), 2)
        put_shadow_text(frame, team2_percentage_text, team2_text_org, 1, (255, 255, 255), 2)

        
    
        return frame
    


    def draw_annotations(self,video_frames, tracks, ball_control, team_colors):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["player"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            goalkeeper_dict = tracks["goalkeeper"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))

                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))

            #Draw Goalkeeper
            for track_id, golkeeper in goalkeeper_dict.items():
                frame = self.draw_ellipse(frame, golkeeper["bbox"],(0,0,0), track_id)   
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))

            #draw ball control
            frame = self.draw_ball_control(frame, frame_num, ball_control, team_colors) 



            output_video_frames.append(frame)

        return output_video_frames        