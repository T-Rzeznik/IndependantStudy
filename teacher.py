# teacher.py
import ollama
import time

from roles import Roles
from history import History

TEACHER_BASE = """A tutor and a student work together to solve the following math word problem. 
Math problem: {problem}
The correct solution is as follows:
{ground_truth}
The following is a conversation with a teacher. The teacher is polite, helpful, professional, on topic, and factually correct.
"""

class InstructLlamaTeacher(object):
    def __init__(self):
        self.persona = Roles.TEACHER
        self.name = 'Llama Tutor'

    def response(self, history: History, student_question: str, incorrect_solution: str):
        response = ""
        messages = history.to_delimited_string("<EOM>\n\n")
        prompt = TEACHER_BASE.replace("{problem}", student_question) \
                                .replace("{ground_truth}", incorrect_solution) \
                                .replace("(DIALOG HISTORY)", messages)
        print("Prompt:", prompt)
        errors_counter = 0
        max_retries = 5  # Set a maximum number of retries
        done = False
        while not done and errors_counter < max_retries:
            try:
                response = ollama.chat(model="llama3", messages=[{'content': prompt, 'role': 'user'}])
                print("Raw response:", response)
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
    
