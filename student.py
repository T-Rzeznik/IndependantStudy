import ollama
import time

from roles import Roles
from history import History



# Define the student persona and context for Llama 3
STUDENT_NAME = "Kayla"
STUDENT_PERSONA = f"{STUDENT_NAME} is a 7th grade student. She has problems with understanding what steps or procedures are required to solve a math problem."

# Define the prompt for the student model
STUDENT_PROMPT = f"""
Student Persona: {STUDENT_PERSONA}

Math problem: (MATH PROBLEM)

Student solution: (STUDENT SOLUTION)

Context: {STUDENT_NAME} thinks their answer is correct. Only when the teacher provides several good reasoning questions, {STUDENT_NAME} understands the problem and corrects the solution. {STUDENT_NAME} can use a calculator and thus makes no calculation errors. Send <EOM> tag at the end of the student message.

(DIALOG HISTORY)

Only respond to the last message

Student:
"""

class InstructLlamaStudent(object):
    def __init__(self):
        self.persona = Roles.STUDENT
        self.name = STUDENT_NAME

    def response(self, history: History, question: str, incorrect_solution: str):
        response = ""
        messages = history.to_delimited_string("<EOM>\n\n")
        last_message = history.messages[-1] if history.messages else None
        prompt = STUDENT_PROMPT.replace("{STUDENT PERSONA}", STUDENT_PERSONA) \
                               .replace("(STUDENT SOLUTION)", incorrect_solution) \
                               .replace("(MATH PROBLEM)", question) \
                               .replace("{STUDENT NAME}", STUDENT_NAME) \
                               .replace("(DIALOG HISTORY)", messages) + f"\nLast message: {last_message}"
        errors_counter = 0
        done = False
        while not done:
            try:
                response = ollama.chat(model="llama3", messages=[{'content': prompt, 'role': 'user'}])
                response = response["message"]["content"].strip()
                done = True
            except Exception as e:
                print(e)
                errors_counter += 1
                time.sleep(1)
        utterance = response.replace("Student:", "") \
                           .replace(f"{STUDENT_NAME}:", "") \
                           .replace("Roles.STUDENT:", "") \
                           .replace("<EOM>", "") \
                           .strip("\n")
        return utterance