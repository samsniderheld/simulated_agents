import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import save

class TTSWrapper:
    """
    A wrapper class for the ElevenLabs API to generate videos from images using a pre-trained model.

    Attributes:
        api (str): The API to use for audio generation.
        poll_rate (int): The rate at which to poll the API for task status.
        client (ElevenLabs): The ElevenLabs client object.
    """

    def __init__(self, api: str = "eleven_labs", voice: str = "John Doe - Deep") -> None:
        """
        Initializes the TTSWrapper with the specified API and poll rate.

        Args:
            api (str): The API to use for TTS generation. Defaults to "runway".
            poll_rate (int): The rate at which to poll the API for task status. Defaults to 10 seconds.
        """
        self.api = api
        self.voice = voice
        self.voice_id = ""
        if self.api == "eleven_labs":
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise ValueError("ELEVENLABS_API_KEY environment variable is not set.")
            self.client = ElevenLabs()
        else:
            self.client = None

        load_dotenv()

    def make_api_call(self, prompt: str, seed:int=0, idx:int=None) -> str:
        """
        Makes an API call to generate the audio given the prompt.

        Args:
            prompt (str): The text prompt for generating the autio.

        Returns:
            str: The path to the generated audio.
        """
        if self.api == "eleven_labs":

            if self.voice == "John Doe - Deep":
                self.voice_id = "EiNlNiXeDU1pqqOPrYMO"
            elif self.voice == "Rebekah Nemethy - Pro Narration":
                self.voice_id = "ESELSAYNsoxwNZeqEklA"
            else:
                raise ValueError("voice not specified")
            
            audio = self.client.text_to_speech.convert(
                text=prompt,
                voice_id=self.voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                seed=seed
            )

            if idx is not None:
                path = f"out_audio/{idx}.mp3"
            else:
                path = f"out_audio/{prompt[:10]}.mp3"

            save(audio, path)

            return path
        else:
            raise ValueError("video api not specified")