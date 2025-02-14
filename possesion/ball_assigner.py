import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance
import numpy as np

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player

    def calculate_ball_posession(self,tracks):
        team_ball_control= []
        for frame_num, player_track in enumerate(tracks['player']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = self.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['player'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['player'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
        team_ball_control= np.array(team_ball_control)

        return team_ball_control

