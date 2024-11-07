from student import InstructLlamaStudent
from teacher import InstructLlamaTeacher
from history import History
from message import Message
from roles import Roles
from utils import read_jsonl
from tqdm import tqdm
import json

def main(input_file, export_file, max_utterances):
    conversations = []
    data = read_jsonl(input_file)
    student = InstructLlamaStudent()
    teacher = InstructLlamaTeacher()

    for problem in tqdm(data):
        question = problem["question"]
        ground_truth_solution = problem["ground_truth"]
        incorrect_solution = problem["student_incorrect_solution"]

        history = History()
        history.add_message(Message(Roles.TEACHER, "Hi " + student.name + "! Could you walk me through your solution?"))

        for _ in range(max_utterances):
            student_message = Message(Roles.STUDENT, student.response(history, question, incorrect_solution))
            history.add_message(student_message)

            teacher_response_message = Message(Roles.TEACHER, teacher.response(history, question, ground_truth_solution))
            history.add_message(teacher_response_message)

        # Store only the final conversation history
        problem["ollama_model"] = history.to_delimited_string("<EOM>")
        conversations.append(problem)

        print("Conversation:")
        print("Question:", question)
        print("Ground Truth Solution:", ground_truth_solution)
        print("Incorrect Solution:", incorrect_solution)
        print("History:\n", history)

    # Export conversations to a JSONL file
    with open(export_file, 'w') as f:
        for conversation in conversations:
            f.write(f"{json.dumps(conversation)}\n")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ollama Conversation')
    parser.add_argument('--input_file', type=str, required=True, help='Input JSONL file with problems')
    parser.add_argument('--export_file', type=str, required=True, help='Output file to export conversations')
    parser.add_argument('--max_utterances', type=int, default=5, help='Maximum number of utterances in the conversation')
    args = parser.parse_args()

    main(args.input_file, args.export_file, args.max_utterances)