from utils import *
from tracker import Tracker
from teams import Assigner
from possesion.ball_assigner import PlayerBallAssigner
import numpy as np
def main():
    #read video
    video_frames = read_video('assets/vid1.mp4')
    #create tracker
    tracker = Tracker('model/best.pt')
    #track video
    
    tracks = tracker.getObjectTracks(video_frames,
                                       read_from=False,
                                       path='stubs/track_stubs_gk.pkl')
    tracker.add_position_to_tracks(tracks)

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])


    #Assign Team players
    team_assigner = Assigner()
    team_assigner.assign_team_color(video_frames[0],tracks['player'][0])

    for frame_num, player_track in enumerate(tracks['player']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
            tracks['player'][frame_num][player_id]['team'] = team 
            tracks['player'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # Assign Ball to player
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control= []    
    # team_ball_control= player_ball_assigner.calculate_ball_posession(tracks)
    for frame_num, player_track in enumerate(tracks['player']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['player'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['player'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)


    # draw annotations        
    output_video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control,team_assigner.team_colors)

    # Save the video
    output_path = 'output/processed_video.avi'
    save_video(output_video_frames, output_path)

if __name__ == "__main__":
    main()