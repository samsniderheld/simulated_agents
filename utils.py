import os
import shutil

from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips


def list_to_txt(input_list: list, filename: str) -> None:
    """
    Writes each element of the input list to a new line in the specified .txt file.

    Args:
        input_list (list): The list of elements to write to the file.
        filename (str): The name of the output .txt file.
    """
    with open(f"{filename}.txt", 'w') as file:
        for element in input_list:
            file.write(f"{element}\n")

def create_directory(directory):
    """
    Makes directory. Clears all files in the specified directory if it exists.

    Args:
        directory (str): The path to the directory to clear.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def concatenate_actions(beat_number, all_scenes,character_agents):
    """
    Concatenates actions for a given beat number.

    Args:
        beat_number (int): The beat number.
        all_scenes (list): List of all scenes.

    Returns:
        str: Concatenated actions for the beat number.
    """
    character_num = len(character_agents)
    start = int(beat_number*character_num)
    end = start + character_num
    concatenated_actions = " ".join(all_scenes[start:end])
    return concatenated_actions

def combine_video_audio(video_path, audio_path, output_path):
    """
    Combines video and audio into a single file.

    Args:
        video_path (str): The path to the video file.
        audio_path (str): The path to the audio file.
        output_path (str): The path to the output file.
    """
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.with_audio(audio_clip)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

def concatenate_videos(video_paths, output_path):
    """
    Concatenates multiple video files into a single file.

    Args:
        video_paths (list): List of paths to the video files.
        output_path (str): The path to the output file.
    """
    clips = [VideoFileClip(video) for video in video_paths]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')