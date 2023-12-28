import librosa, librosa.display
import json, time, warnings
import numpy as np
import utilities
from config import settings

#Run command: python video_extractor.py

# Ignore all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


# Function to extract videos, given a keyword.
def video_extraction_keyword(keywords, max_results):
    '''This function extracts the given number of results given a keyword,
    and saves the file in a .json file for later reference.'''

    overall_dictionary = {}
    
    # Replace 'KEYWORDS' with the keywords you want to search for
    keywords = keywords

    # Set the maximum number of results to retrieve (default is 5)
    max_results = max_results

    videos = utilities.search_videos_keyword(settings.youtube_api, keywords, max_results)

    if videos:
        for index in range(len(videos)):
            overall_dictionary[f"{videos[index]['id']['videoId']}"] = {}
            overall_dictionary[f"{videos[index]['id']['videoId']}"]['Video_URL'] = f"https://www.youtube.com/watch?v={videos[index]['id']['videoId']}"
            overall_dictionary[f"{videos[index]['id']['videoId']}"]['Details'] = videos[index]

    # Convert dictionary to JSON string
    overall_response_string = json.dumps(overall_dictionary, indent=4)  # Use indent for pretty formatting

    # Save JSON string to a file
    with open(f"{keywords}_videos.json", "w") as json_file:
        json_file.write(overall_response_string)


# Function to extract video metadata from a channel palylist, given a channel ID.
def video_extraction_channel(channel_id, max_results):
    '''This function extracts the given number of results given a keyword,
    and saves the file in a .json file for later reference.'''

    overall_dictionary = {}
    except_messgs = {}
    
    # Replace 'KEYWORDS' with the keywords you want to search for
    channel_id = channel_id

    # Set the maximum number of results to retrieve (default is 5)
    max_results = max_results

    #------------------------------------------------------------------------------------------------------
    # Extract video metadata for a limited number of videos from the channel playlist
    # start_int = time.perf_counter()
    try:
        videos = utilities.search_videos_channel(settings.youtube_api, channel_id, max_results)
    except Exception as e:
        # In case the video metadata extraction is unsuccessful.
        except_messgs[f"(search_videos_channel)"] = f"{type(e).__name__}: {e}"

    if videos == []:
        except_messgs[f"(search_videos_channel)"] = f"Video metadata extraction from channel '{channel_id}' was unsuccessful."
    # finish_int = time.perf_counter()
    # print(f"Channel video content extraction finished in {round((finish_int-start_int), 2)} sec(s)")
    
    #------------------------------------------------------------------------------------------------------
    # Loops through each video ID and checks the language distribution of the languages present in the video.
    if videos: #Checks if video variable was successfully declared
        if videos != []: #Checks if any video metadata was successfully extracted
            for index in range(len(videos)):

                # Extracts the necessary metadata for the given video
                video_id = videos[index]['snippet']['resourceId']['videoId']

                try:
                    video_info = utilities.get_youtube_video_info(settings.youtube_api, video_id)
                except Exception as e:
                    except_messgs[f"(get_youtube_video_info)"] = f"{type(e).__name__}: {e}"

                overall_dictionary[f"{video_id}"] = {}
                overall_dictionary[f"{video_id}"]['Video_URL'] = f"https://www.youtube.com/watch?v={video_id}"
                overall_dictionary[f"{video_id}"]['Title'] = videos[index]['snippet']['title']
                overall_dictionary[f"{video_id}"]['PublishedAt'] = videos[index]['snippet']['publishedAt']
                
                if video_info != None:
                    overall_dictionary[f"{video_id}"]['Duration'] = f"{utilities.convert_duration_to_seconds(video_info['contentDetails']['duration'])} secs"
                else:
                    overall_dictionary[f"{video_id}"]['Duration'] = ''
                    except_messgs[f"(get_youtube_video_info)"] = f"Metadata extraction from video '{video_id}' was unsuccessful."

                #---------------------------------------------------------------------------------------------------------
                # Downloads the audio file to a temporary folder and analyzes the contents in segments
                # start_int = time.perf_counter()
                try:
                    mp4_path, wav_path = utilities.download_audio(video_id)
                except Exception as e:
                    # In case the audio download was unsuccessful
                    except_messgs[f"(download_audio)"] = f"{type(e).__name__}: {e}"

                if wav_path:
                    try:
                        first_language = 'French'
                        second_language = 'English'
                        percentage_transcribed, percentage_firstlang, percentage_secondlang = utilities.analyze_audio_languages_google(wav_path, first_language, second_language)
                        # percentage_transcribed, lang_distribution = utilities.analyze_audio_languages_openai(wav_path)
                        overall_dictionary[f"{video_id}"]['Percentage Transcribe'] = f"{percentage_transcribed}%"
                        overall_dictionary[f"{video_id}"]['First Language'] = f"{first_language}: {percentage_firstlang}%"
                        overall_dictionary[f"{video_id}"]['Second Language'] = f"{second_language}: {percentage_secondlang}%"
                    except Exception as e:
                        # In case the language analysis was unsuccessful
                        except_messgs[f"(analyze_audio_languages_google)"] = f"{type(e).__name__}: {e}"
                        except_messgs[f"(analyze_audio_languages_google)"] = f"Audio language analysis for video '{video_id}' was unsuccessful."
                        # print(f"Error: {e}")
                        overall_dictionary[f"{video_id}"]['Percentage Transcribe'] = ''
                        overall_dictionary[f"{video_id}"]['First Language'] = ''
                        overall_dictionary[f"{video_id}"]['Second Language'] = ''
                else:
                    # In case the audio download was unsuccessful
                    except_messgs[f"(download_audio)"] = f"Audio download for video '{video_id}' was unsuccessful."
                    overall_dictionary[f"{video_id}"]['Percentage Transcribe'] = ''
                    overall_dictionary[f"{video_id}"]['First Language'] = ''
                    overall_dictionary[f"{video_id}"]['Second Language'] = ''
                # finish_int = time.perf_counter()
                # print(f"Language Distribution Analysis finished in {round((finish_int-start_int), 2)} sec(s)")
                # print(f"Percentage transcribed: {percentage_transcribed}%, {first_language}: {percentage_firstlang}%, {second_language}: {percentage_secondlang}%")
                try:
                    utilities.delete_audios([mp4_path, wav_path])
                except Exception as e:
                    # In case the deletion of temporary files was unsuccessful
                    except_messgs[f"(delete_audios)"] = f"{type(e).__name__}: {e}"
                    except_messgs[f"(delete_audios)"] = f"Auido files deletion from temp folder for video '{video_id}' was unsuccessful."
                print('')
    
    # Convert meta data dictionary and error log dict to JSON strings
    overall_response_string = json.dumps(overall_dictionary, indent=4)
    except_messgs_string = json.dumps(except_messgs, indent=4)

    # Save JSON string to a file
    with open(f"./channel_content/{channel_id}_videos.json", "w") as json_file:
        json_file.write(overall_response_string)
    with open(f"./channel_content/error_logs/{channel_id}_errorlogs.json", "w") as json_file:
        json_file.write(except_messgs_string)



if __name__ == "__main__":

    start = time.perf_counter()
    # key_word = 'English Tutorial'
    # channel_id = 'UCoUWq2QawqdC3-nRXKk-JUw' #EasyFrench
    # channel_id = 'UCVzyfpNuFF4ENY8zNTIW7ug' #Piece of French
    channel_id = 'UCbj8Qov-9b5WTU1X4y7Yt-w' #French Mornings with Elisa

    # Extracts the video using the specified keyword
    try:
        # video_extraction_keyword(key_word, 10)
        video_extraction_channel(channel_id, 2)

        # print(f"{key_word} YouTube videos extraction was successsful!")
        print(f"{channel_id} YouTube videos extraction was successsful!")
    except Exception as e:
        print(f"Error: {e}")

        # print(f"{key_word} YouTube videos extraction was unsuccesssful!")
        print(f"{channel_id} YouTube videos extraction was unsuccesssful!")

    finish = time.perf_counter()
    print(f"Channel video content and language analysis finished in {round((finish-start)/60, 2)} minute(s)")