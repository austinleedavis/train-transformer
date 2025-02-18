import http.client
import logging

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class NtfyCallback(TrainerCallback):
    """A callback class for sending notifications at the start and end of training.

    Args:
        topic (str): The Ntfy.sh topic to notify
    """

    logger = logging.getLogger(__name__)
    topic: str

    def __init__(self, topic: str = None):
        self.topic = topic
        if topic is None:
            self.logger.warning("No topic specified. Ntfy will use Logger instead.")
            self.send_notification = lambda *a, **k: self.logger.info(str((a, k)))

    def on_train_begin(self, args, state, control, **kwargs):
        self.send_notification(f"Training Began: {args.output_dir}")

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        self.send_notification(f"Training Ended: {args.output_dir}")

    def send_notification(self, message):

        if len(self.topic) > 0 and self.topic[0] != "/":
            self.topic = "/" + self.topic

        try:
            # Define the URL and path
            url = "ntfy.sh"

            headers = {"Content-Type": "text/plain"}

            conn = http.client.HTTPSConnection(url)
            conn.request("POST", self.topic, message, headers)
            response = conn.getresponse()
            if response.status == 200:
                self.logger.info(f"Notification sent successfully: {message}")
            else:
                self.logger.info(
                    f"Failed to send notification. Status code: {response.status}, Reason: {response.reason}, Message: {message}"
                )
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Ensure the connection is closed
            conn.close()
