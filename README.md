# YouTube Video Anlysis

## Project Requirements
+ Analysis of a given youtube video when provided wih the video ID or analysis of a set of videos from a given channel when provided with the channel ID.
+ Analysis is conducted in two parts subtitle analysis and audio analysis. Video analysis was mentioned, but was left for future implementation, till after the current requirements have been accomplished.
+ The subtitle is processed in an automated manner using OpenAI's intelligent model, GPT. This is to extract certain properties of the video content, like;
    - Level of difficulty of the video content.
    - Summary of the entire content.
    - Singular Topic for the video content.
    - Level of articulation of the video content.
    - Vocabulary level of the video content.
    - Sentence construct level of the video content.
    - Presence/Absence of Dialogue in the video content.
+ The audio is processed also in an automated manner using Google's API to confirm the distribution of the languages present in the video, as only client is only interested in videos with one language.

## Project workflow
+ There are two main methods in this project, which are the "video_extraction_channel" method in the video_extractor.py file and the "channel_video_parser" method in the video_analyzer.py file.

### Video_Extraction_Channel
+ The video_extraction_channel function takes in a channel ID and extracts the video details (containing video ID, title, etc.) for a limited number of videos. This function also conducts an audio analysis using the Google API (google text-to-speech), to check the distribution of the languages present in the video.

```python
>>> def video_extraction_channel(channel_id, max_results):
        '''This function extracts the given number of results given a keyword,
        and saves the file in a .json file for later reference.

        Parameters
        ----------
        channel_id (string): The YouTube channel ID in string format.
        max_results (integer): A threshold for the number of videos details to extract from the channel

        Returns
        ----------
        This function simply saves the extracted video contents to a .json file, and 
        does not return anything.'''
```

+ This function makes use of a couple of other function in the utitlies.py file

    + ```python
      >>> def search_videos_channel(api_key, channel_id, max_results=50):
              '''This function is responsible for  calling the YouTube API to extract
              the metadata of the limited number of videos in a given channel playlist.
                    
              input: YouTube API key, channel_id, max_results
              output: dictionary (containing metadata on videos from channel chosen)'''
      ```

    + ```python
      >>> def get_youtube_video_info(api_key, video_id):
              '''This function takes in each video ID and extracts more needed metadata that 
              weren't gotten from the first call.
                        
              input: YouTube API key, video_id
              output: dictionary (containing metadata on video chosen)'''
      ```

    + ```python
      >>> def convert_duration_to_seconds(duration):
              '''This function takes the video duration in it's isoformat and converts it to seconds.
                        
              input: duration in isoformat
              output: duration in seconds'''
      ```

    + ```python
      >>> def download_audio(video_id, output_path=audio_files):
              '''This function takes in video ID and downloads the audio (mp3 and wav) to a folder in the repo.
                        
              input: video ID, output folder
              output: file location of mp3 and wav file'''
      ```

    + ```python
      >>> def analyze_audio_languages_google(audio_path, first_language, second_language, segment_duration_ms=4000):
              '''This function takes in the wav audio path, splits into segments and checks for the
              distribution of the first and second language specified, using Google SpeechRecognition
              API.
                        
              input: video ID, output folder
              output: file location of mp3 and wav file'''
      ```

    + Note: The audio analysis is conducted in a distributed manner, using 4 processing nodes as the number of available nodes for processing (this can be increased with availabilty of more nodes). The list of audio segments are split in four subset for each node. In each node, the audio segments are sent the Google API in a multithreaded fashion for quickened processing. This is done using the following functions; "audiolang_set_processor_google" and "audiolang_sing_processor_google"

    + ```python
      >>> def audiolang_set_processor_google(sublist):
              '''This function takes in a sublist of audio segment details and processes
              it in a parallel.
                        
              input: sub-list of audio segments (list)
              output: a set containing the sum of the transcibed segments and their respictive languages (set)'''
      ```

    + ```python
      >>> def audiolang_sing_processor_google(input_set):
              '''This function takes in the set of input necesasry to process the audio segment,
              in order to execute a parallel process.
                        
              input: a segment of the audio file (set)
              output: set containing the status of the transcription and the language if successful (set)'''
      ``` 

    + ```python
      >>> def delete_audios(path_list):
              '''This function takes in the list of audio file paths and deletes them from the repo.
                        
              input: list of audio file paths'''
      ```

+ After all these properties are extracted about a list of videos from the specified channel, this information is saved in a .json file, which will be parse through the next function for extended analysis.



### Channel_Video_Parser
+ The channel_video_parser function takes in a channel ID and loads the .json file attributed to this chanel ID in the folder containing list of videos gotten from each channel. Thsi .json file is then converted to a dictionary and the function parses each video ID through the sub-function "video_analyzer" for more in-depth analysis, provided the video language distribution indicates the video contains majorly one language.

```python
>>> def channel_video_parser(channel_id):
        '''This function takes in a channel id and process all the videos (that have its dominant 
        language spoken during over 90% of the video) listed under this channel id in the channel_content 
        folder. 

        Parameters
        ----------
        channel_id (string): The YouTube channel ID in string format.

        Returns
        ----------
        This function simply saves the extracted video contents to a .json file, and 
        does not return anything.'''
```

```python
>>> def video_analyzer(video_id):
        '''This function takes in a video ID  as the channel parser loops through the
        channel .json file, extracts and analyses its content, and then stores this 
        content in a file.

        Parameters
        ----------
        video_id (string): The YouTube video ID in string format.

        Returns
        ----------
        This function simply saves the extracted video contents to a .json file, and 
        does not return anything.'''
```

+ The "video_analyzer" function makes use of a couple of other function in the utitlies.py file

    + ```python
      >>> def get_youtube_video_info(api_key, video_id):
              '''This function takes in each video ID and extracts more needed metadata that 
              weren't gotten from the first call.
                        
              input: YouTube API key, video_id
              output: dictionary (containing metadata on video chosen)'''
      ```

    + ```python
      >>> def convert_duration_to_seconds(duration):
              '''This function takes the video duration in it's isoformat and converts it to seconds.
                        
              input: duration in isoformat
              output: duration in seconds'''
      ```

    + ```python
      >>> def get_video_transcript(video_ids, languages):
              '''This function extracts the subtitle of a youtube video using its video ID
                        
              input: video_id, subtitle language detected by YouTube.
              output: dictionary (containing subtitles segments and duration)'''
      ```

    + ```python
      >>> def speech_speed(video_transcript, subtitle_language):
              '''This function takes in the video transcript and calculates the speed of the speech in the video. This function omits subtitle segment depicting only music, and the language version
              of the word "music" to look out for depends on the subtitle language.
                        
              input: video transcript (dictionary), subtitle language
              output: combined duration of video speech (integer), words per minute (integer), audio speed category (string) '''
      ```

    + ```python
      >>> def combine_transcript_translate(transcript, source_language):
              '''This processes the extracted subtitle, translates (to English in a parallel manner, 
              if text isn't already in English) and combines all its texts into one long string.
                        
              input: video transcript (dictionary), source language (string, extracted from video metadata)
              output: combined subtitle in English (string)'''
      ```

    + Note: In the event the video subtitle isn't in English and needs to be translated from source language to English, this translation process is executed in parallel and is done so by calling two more functions; "subt_set_translator" and "subt_sing_translator". The list of subtitle segments is broken into sublists (using the number of available processing cores/nodes) and processed in a paralled manner using the "subt_set_translator" function. The "subt_set_translator" takes in this subset of the list containing the subtitle segments and translates then in a multithreaded fashion by making use of "subt_sing_translator" which calls the Google Translation API.

    + ```python
      >>> def subt_set_translator(sublist):
              '''This function takes in a sublists containing parts of an extracted sutitle,
              and then translates it in a parallel manner (Multithreaded).
                        
              input: sub-list of subtitle segments (list)
              output: translated subtitle segments in English (dictionary)'''
      ```

    + ```python
      >>> def subt_sing_translator(input_set):
              '''This function takes in a part of an extracted subtitle and translates it using Google's API.
                        
              input: a segment of the subtitle (set)
              output: translated subtitle segment in English (set)'''
      ```

    + ```python
      >>> def subtitle_processing(combined_subt):
              '''This function takes the combined raw subtitle and punctuates it using GPT in a parallel
              manner (Multiprocessor).
                                
              input: combined subtitle in English (string)
              output: punctuated combined subtitle in English (string), truncated subtitle using GPT token threshold (string)'''
      ```

    + Note: The subtitle punctaution is also done in a distributed manner for mainly two reasons. First is to be able to fit the entire subtitle into GPT, and secondly for quicker computation. The distributed computation is also carried out using two functions; "subt_set_punctuator" and "subt_sing_punctuator". The combined translated subtitle is split into chunks (using the number of available nodes/cores once again) and appended to a list. This list is broken into sublists using the number of available nodes/cores and each sublist is fed into "subt_set_punctuator" in a parallel manner. Each sulist being processed by each node is then sent to the OpenAI GPT model in a multithreaded manner.

    + ```python
      >>> def subt_set_punctuator(sublist):
              '''This function takes in a sublists containing parts of a combined sutitle,
              and then processes it in a parallel manner (Multithreaded).
                        
              input: sub-list of combined subtitle chunks (list)
              output: punctuated subtitle chunks in English (dictionary)'''
      ```

    + ```python
      >>> def subt_sing_punctuator(part_sub):
              '''This function takes in a part of a combined subtitle and punctuates it using GPT's API.
                        
              input: a chunk of the combined subtitle (set)
              output: translated subtitle chunk in English (set)'''
      ``` 

    + ```python
      >>> def text_set_analyzer(sublist):
              '''This function takes in a sublist of categories of text analysis
              and then processes it in a parallel manner (Multithreaded).
                                
              input: set of text analysis categories (list)
              output: result of text analysis for each category (dictionary)'''
      ```

    + Note: There are currently 7 categories (stated in the project requirements) of text analysis conducted using OpenAI's GPT model. These categories are processed in parallel using two function; "text_set_analyzer" and "text_sing_analyzer". The list of categories are split into sublists using the number of available cores/nodes. These sublists are processed in a parallel manner, with each sublist processed using the "text_set_analyzer" function. Each sublist sent to the "text_set_analyzer" function is then processed in a mulithreaded manner using the "text_sing_analyzer" which calls the GPT function to conduct the text analysis.

    + ```python
      >>> def text_set_analyzer(sublist):
              '''This function takes in a sublist of categories of text analysis
              and then processes it in a parallel manner (Multithreaded).
                                
              input: subset of text analysis categories (list)
              output: result of text analysis for each category (dictionary)'''
      ```

    + ```python
      >>> def text_sing_analyzer(input_set):
              '''This function takes in a category and a truncated string and conducts a particular
              type of analysis based on the category.
                                
              input: text analysis category and truncated subtitle (set)
              output: result of text analysis for the given category (set)'''
      ```

    + Note: Each text analysis category has a dedicated function which contains its respective engineered prompt which guides GPT on how to analyze the truncated subtitle and how to present the response. Each of these functions are called by the "text_sing_analyzer", take in the the truncated subtitle and outputs the GPT response based on the engineered response.

    + ```python
      >>> def gpt_punctuator(information):
              '''Function is responsible for querying the GPT-3.5 model for analysis of a given content.
                                
              input: truncated subtitle (string)
              output: analytical response from GPT (string)'''
      >>> def gpt_categorizer(information):
      >>> def gpt_summarizer(information):
      >>> def gpt_topicmodeller(information):
      >>> def gpt_qualitycheck(information):
      >>> def gpt_vocabularycheck(information):
      >>> def gpt_sentenceconstruct(information):
      >>> def gpt_dialogue(information):
      ```

+ After a particular video has been parsed through the "video_analyzer" function for in-depth analysis, the results of this analysis is saved in a .json file (this can be loaded to a cloud database as per client's request).


### Unused/Stand-by Functions
+ An alternative function was developed to check the language distribution of a given video using the OpenAI's whisper model. This function is not restricted to just two pre-guessed languages and does need an language specified to aid the function to run. However, this function takes a significantly longer time to run and will not be suitable for a real-time execution (without possible future modifications). However, the function for this has been implemented and test and set in the backgroung in case of future exploration.

```python
>>> def analyze_audio_languages_openai(audio_path, segment_duration_ms=4000):
        '''This fucntion downloads the video audio file using the video ID and calculated the
        audio BPM (Beats per minute).

        input: file path to the downloaded audio (string)
        output: percentage of transcribed segments (integer), distribution of languages present (dictionary)'''
```

```python
>>> def audiolang_set_processor_openai(sublist):
        '''This function takes in a sublist of audio segment details and processes
        it in a parallel.

        input: sub-list of audio segments (list)
        output: a set containing the sum of the transcibed segments and their respictive languages (set)'''
```

```python
>>> def audiolang_sing_processor_openai(input_set):
        '''This function takes in the set of input necesasry to process the audio segment,
        in order to execute a parallel process.

        input: a segment of the audio file (set)
        output: set containing the status of the transcription and the language distribution (set)'''
```

```python
>>> def openai_whisper_api(path):
        '''This function takes in a file path and loads the audio file at the end of this
        file path onto the openai_whisper_api for transcription..

        input: file path of the saved audio segment (string)
        output: detected language for the given audio segment (string)'''
```

+ Functions were developed to account for the part of the project involving image analysis. These include functions for video download using video ID, image frame extraction from the video using OpenCV package, and image analysis using OpenAI's GPT4-V model.

```python
>>> def download_youtube_video(video_url, output_path='.'):
        '''This function downloads a given youtube video using its video url.

        input: video url (string), output file path for the downloaded video(string)'''
```

```python
>>> def extract_frames(video_path, output_folder):
        '''This function extract image frames from the downloaded youtube video.

        input: file path to downloaded video (string), output file path for extracted frames (string)'''
```

```python
>>> def list_files_in_folder(folder_path):
        '''This function add the names of the image frames extracted from the downloaded video to a list.

        input: filepath containing the extracted images (string)
        output: list of filenames of the saved images in the specified (list)'''
```

```python
>>> def gpt_v_image_analyser(image_name):
        '''This function converts the extracted image frames to base64 and analyzes its content using GPT4-V.

        input: image name to be analyzed (string)
        output: analytical response from GPT4-V about the uploaded image (string)'''
```

+ Functions were developed to analyze the beats per minute of the video audio file. However, these function were put on stand-by because the function developed to calculate words per minute using the video subtitle gave a more accurate depiction of the speech speed.

```python
>>> def analyze_audio_speed(audio_path):
        '''This function analyses the speed of the audio file.

        input: filepath of the download or extracted audio file (string).
        output: beat per minute of the audio file (integer)'''
```

### Important Note about Functions Developed
+ There were 6 open source models/packages used (via API) to conduct the various data processing tasks in this project, which are as follows; YouTube API, Google SpeechRecognition, Googe Translation, OpenAI's GPT 3.5 Turbo, OpenAI's Whisper, OpenAI's GPT4-V. Only YouTube API, OpenAI's GPT 3.5 Turbo and GPT4-V needed API keys. The rest were free versions and didn't need any keys to execute. The OpenAI's model make use of the same key. These keys will need to be generated and included in the environment variable file before executing the scripts.

+ The Google Translation model/API has a daily quota of about 50,000 words for the free version. To scale this up for large volumes of analysis per day, an API key will need to be set up and the code for the advance version of the model will need to be integrated into the script.

+ A series of these function that dealt with the video and audio file made calls to file paths for already created folder in order to save either the media file or a segment of it (during distributed processing). Hence, these folder will need to be created and before running the script.

+ All the parallel functions developed for this project were developed with 4 available nodes/cores. This can be adjusted in the case of more available cores/nodes for processing.

+ Error logs for the source code logic flow is manually integrated into the system to help track source of error in code.

+ There are four major folder used to store files in the source script, that need to be create when running the script.
    + ./audio_files : to store the audio file and its segments during parallel processing
    + ./channel_content : to store the .json file containing the metadata of the videos from the channel playlist
        + ./channel_content/error_logs : to store any errors that could arise while processing a given channel
    + ./video_details : to store the metadata and analysis results of each video gotten from the channel palylist
        + ./video_details/error_logs : to store any errors that could arise while processing a given video
    + ./output_frames : to store the image frames extracted from the video files for image analysis

    + Note: these folders can be changed, but these changes have to be reflected in the source code to prevent errors. Also, these folders (channel_content and video_details) are only necessary when sotring the results of the extraction and analysis on a local environment. If these results are loaded directly to a relational database, then these two folder would no longer be necessary.



