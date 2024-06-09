from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

app = Flask(__name__)

# Load the fine-tuned model
model_path = os.path.join(os.getcwd(), "trained_model")

print("Model Path:", model_path)

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generate_structured_response(chapter, num_questions=5):
    prompt = f"Generate questions based on {chapter} chapter"
    input_text = f"Prompt: {prompt}\nQuestion:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    # Set pad_token_id to eos_token_id to avoid warnings
    pad_token_id = tokenizer.eos_token_id
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=num_questions,
        num_beams=5,
        pad_token_id=pad_token_id
    )

    # Decode the generated outputs
    questions = []
    for output in outputs:
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        # Ensure the response is split correctly and only the question part is extracted
        if "Question:" in decoded_output:
            question_part = decoded_output.split("Question:")[1].strip()
            # Stop at the first newline character
            question_part = question_part.split('\n')[0].strip()
            questions.append(question_part)

    # Remove duplicates by converting the list to a set and back to a list
    unique_questions = list(set(questions))

    # Format the structured response
    structured_response = "\n\n"
    for i, question in enumerate(unique_questions, 1):
        structured_response += f"{i}. {question}\n\n"
    return structured_response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    data = request.json
    chapter = data.get('chapter')
    num_questions = data.get('num_questions', 5)
    print("Generating!!");
    
    response = generate_structured_response(chapter, num_questions)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
