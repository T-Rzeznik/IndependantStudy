import os
import argparse
import json
from tqdm import tqdm as tqdm

from history import History
from message import Message
from roles import Roles
from student import InstructLlamaStudent
from teacher import InstructLlamaTeacher
from utils import read_jsonl

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/example.jsonl")
    parser.add_argument("--export_file", type=str, default="output/model_output.jsonl")
    parser.add_argument("--max_utterances", type=int, default=4)
    return parser.parse_args()

def export_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as output_file:
        for conversation in data:
            output_file.write(json.dumps(conversation) + '\n')

def print_conversation(question: str, ground_truth_solution: str, incorrect_solution: str, history: History):
    print("\n\n## Conversation")
    print(f"Question: {question}")
    print(f"Correct solution: {ground_truth_solution}")
    print(f"Incorrect solution: {incorrect_solution}")
    print(history)

if __name__ == '__main__':
    args = get_args()
    conversations = []
    data = read_jsonl(args.input_file)
    student = InstructLlamaStudent()
    teacher = InstructLlamaTeacher()

    for problem in tqdm(data):
        question = problem["question"]
        ground_truth_solution = problem["ground_truth"]
        incorrect_solution = problem["student_incorrect_solution"]

        history = History()
        history.add_message(Message(Roles.TEACHER, "Hi " + student.name + "! Could you walk me through your solution?"))

        for i in range(args.max_utterances):
            student_message = Message(Roles.STUDENT, student.response(history, question, incorrect_solution))
            history.add_message(student_message)

            teacher_response_message = Message(Roles.TEACHER,
                                               teacher.response(history, question, ground_truth_solution))
            history.add_message(teacher_response_message)

        problem["ollama_model_output"] = history.to_delimited_string("<EOM>")
        conversations.append(problem)

        print_conversation(question, ground_truth_solution, incorrect_solution, history)

    export_to_jsonl(conversations, args.export_file)