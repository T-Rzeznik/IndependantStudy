import ollama
import time

from roles import Roles
from history import History

TEACHER_NAME = "Tutor Ollama"
TEACHER_PERSONA = f"{TEACHER_NAME} The teacher is polite, helpful, professional, on topic, and factually correct."

TEACHER_BASE = """ TEACHER Persona: {TEACHER_PERSONA}
Math problem: {problem}
The correct solution is as follows:
{ground_truth}

(DIALOG HISTORY)

Only respond to the last message

Teacher:
"""

class InstructLlamaTeacher(object):
    def __init__(self):
        self.persona = Roles.TEACHER
        self.name = 'Llama Tutor'

    def response(self, history: History, student_question: str, incorrect_solution: str):
        response = ""
        messages = history.to_delimited_string("<EOM>\n\n")
        last_message = history.messages[-1] if history.messages else None
        prompt = TEACHER_BASE.replace("{problem}", student_question) \
                                .replace("{ground_truth}", incorrect_solution) \
                                .replace("(DIALOG HISTORY)", messages) \
                                .replace("{TEACHER_PERSONA}", TEACHER_PERSONA + f"\n Last Message: {last_message}")
        print("Prompt:", prompt)
        errors_counter = 0
        max_retries = 5  # Set a maximum number of retries
        done = False
        while not done and errors_counter < max_retries:
            try:
                response = ollama.chat(model="llama3", messages=[{'content': prompt, 'role': 'user'}])
                response = response["message"]["content"].strip()
                done = True
            except Exception as e:
                print("Error occurred:", e)
                errors_counter += 1
                time.sleep(1)
        if not done:
            print("Failed to get a response after multiple attempts.")
            return "Error: Unable to generate a response."
        utterance = response.replace("Teacher:", "").replace("Llama Tutor:", "").replace("<EOM>", "").strip("\n")
        return utterance
    