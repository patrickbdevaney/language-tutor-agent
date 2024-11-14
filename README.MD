# Language Phrase Generator with Groq API and Swarms Agent

This Python program utilizes the **Groq API** and **Swarms Agent** to generate English-Japanese phrase pairs in **Hiragana**, **Romaji**, and **Kanji**. The program is designed to help students learn foreign languages by iterating over various phrases, with each iteration producing novel variations of previously generated content.

The setup and process are highly adaptable and can easily be configured to work with any language pair that the supported large language models (LLMs) like **GPT-4**, **Claude**, or **Llama** are trained on. By changing the language-related prompts, this framework can be extended to generate phrases in multiple languages (e.g., Spanish, French, German, etc.), making it an ideal tool for multi-lingual learning.

## How It Works

1. **Agent Interaction**:  
   The program works by initializing a **Swarms Agent**, which communicates with the **Groq API**. The agent is set up with a system prompt that asks it to generate phrases in the target language (Japanese in this case) based on English phrases. Each phrase is broken down into **Hiragana**, **Romaji**, and **Kanji** (if applicable), along with the corresponding English translation for clarity.

2. **Prompt Construction**:  
   The agent is constantly fed with a new English phrase from a "seed" list of initial examples. The system instructs the agent to provide a different phrase from the seed array, ensuring continual novelty. The program uses a custom prompt that directs the agent to translate the provided phrase and break it into individual components.

3. **Regex Text Cleaning**:  
   After receiving the agent's response, the output is sanitized using a regular expression (regex) to clean the text and extract relevant translations. This ensures consistency in the output format (e.g., `[Japanese (Hiragana)] [Romaji] [English Translation]`).

4. **Seed Array and Randomized Iteration**:  
   The seed array starts with a collection of initial English-Japanese phrase pairs. After each iteration, the newly cleaned phrase is added back to the seed array if it is unique, allowing the program to generate increasingly diverse phrases. The loop continues indefinitely, ensuring that new material is always available for learning.

5. **File Output**:  
   Every generated phrase, along with its translation and breakdown, is saved into **CSV** and **TXT** files for easy reference and study.

## Key Features

- **Adaptive to any Language**: The program works with any language that LLMs support, such as Spanish, French, Chinese, and more. You can easily adjust the language pairings by modifying the system prompt.
- **Continuous Phrase Generation**: By forcing the agent to generate a phrase different from the seed, the system ensures a continual loop of novel iterations.
- **Automated Output Handling**: The results are automatically saved into both CSV and text files, making it easy to track progress and review all generated phrases.
- **Seed Array Variation**: The seed array ensures that new material is always introduced, preventing the agent from repeating phrases.

## Setup Instructions

1. **Install Dependencies**:  
   Make sure you have **Python 3.x** installed. Create a virtual environment and install the required packages:
   ```bash
   pip install -r requirements.txt

2. **Set Up Environment Variables**:
Create a .env file in the project root and include the following variables:

GROQ_API_KEY=your_api_key_here
WORKSPACE_DIR=your_workspace_directory_here

3. **Run the Program**:
After setting up the environment, you can run the Python script from the command line:

python main.py

4. **Review the Output**:
The program will generate phrase pairs and save them in the CSV and TXT files (by default, japanese_phrases.csv and japanese_phrases.txt).

Example Output Format
For each English phrase, the agent will generate a breakdown similar to this:

[おはよう (おはよう)] [Ohayou] [Good morning]
[こんにちは (こんにちは)] [Konnichiwa] [Hello]
[ありがとう (ありがとう)] [Arigatou] [Thank you]
The generated phrases are saved into both CSV and TXT files.

## Conclusion
This program provides an efficient way to generate novel language phrases for language learning. Its adaptability allows it to be extended to various languages, providing a versatile tool for mastering different languages. By automating the generation of translated phrases and continuously feeding the agent with new prompts, the program helps learners engage with new content and master a foreign language through repetition and iteration.