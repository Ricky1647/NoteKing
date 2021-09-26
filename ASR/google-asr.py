from google.cloud import speech
from google.cloud import storage
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=r"/home/team1/genial-caster-323507-c71bc47c6e48.json"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
	"""Uploads a file to a Google Cloud storage bucket."""

	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(destination_blob_name)

	blob.upload_from_filename(source_file_name)

	# print(
	# 	"File {} uploaded to {}.".format(
	# 		source_file_name, destination_blob_name
	# 	)
	# )

def delete_blob(bucket_name, blob_name):
	"""Deletes a blob from a Google Cloud storage bucket."""
	storage_client = storage.Client()
	bucket = storage_client.get_bucket(bucket_name)
	blob = bucket.blob(blob_name)

	blob.delete()

def transcribe_gcs(gcs_uri):
	"""Asynchronously transcribes the audio file specified by the gcs_uri."""
	from google.cloud import speech

	client = speech.SpeechClient()

	audio = speech.RecognitionAudio(uri=gcs_uri)
	config = speech.RecognitionConfig(
		audio_channel_count = 2,
		language_code="en-US",
		enable_automatic_punctuation=True,
	)

	operation = client.long_running_recognize(config=config, audio=audio)

	# Wait for operation to complete...
	response = operation.result(timeout=500)

	# Each result is for a consecutive portion of the audio. Iterate through
	# them to get the transcripts for the entire audio file.
	return [{"trans":result.alternatives[0].transcript, "conf":result.alternatives[0].confidence} for result in response.results]