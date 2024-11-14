
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the workspace directory from environment variables
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")
os.environ["WORKSPACE_DIR"] = WORKSPACE_DIR
import csv
import time
import re
from dotenv import load_dotenv
from groq import Groq
from swarms import Agent, GraphWorkflow, Node, NodeType

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Initialize Groq client
client = Groq(api_key=api_key)
print("Environment set up and model initialized successfully!")

# Define initial seed examples
seed_examples = [
    "hello, konnichiwa, こんにちは", "thank you, arigatou, ありがとう", "goodbye, sayounara, さようなら",
    "good morning, ohayou, おはよう", "good night, oyasumi, おやすみ", "yes, hai, はい",
    "no, iie, いいえ", "excuse me, sumimasen, すみません", "please, onegai, お願い",
    "I'm sorry, gomennasai, ごめんなさい", "I understand, wakarimashita, わかりました",
    "I don't understand, wakarimasen, わかりません", "How much?, ikura desu ka, いくらですか",
    "Where is the station?, eki wa doko desu ka, 駅はどこですか", "Do you speak English?, eigo ga hanasemasu ka, 英語が話せますか",
    "water, mizu, 水", "food, tabemono, 食べ物", "I need help, tasukete kudasai, 助けてください",
    "Can you help me?, tetsudatte kuremasu ka, 手伝ってくれますか", "I am lost, michi ni mayotta, 道に迷った",
    "bathroom, toire, トイレ", "hotel, hoteru, ホテル", "train, densha, 電車",
    "bus, basu, バス", "restaurant, resutoran, レストラン", "taxi, takushii, タクシー",
    "I am hungry, onaka ga suita, お腹がすいた", "I am thirsty, nodo ga kawaita, 喉が渇いた",
    "Is it far?, tooi desu ka, 遠いですか", "I am tired, tsukareta, 疲れた",
    "today, kyou, 今日", "tomorrow, ashita, 明日", "yesterday, kinou, 昨日",
    "time, jikan, 時間", "money, okane, お金", "friend, tomodachi, 友達",
    "family, kazoku, 家族", "I am happy, ureshii, 嬉しい", "I am sad, kanashii, 悲しい",
    "I am excited, koufun shite iru, 興奮している", "What time is it?, nanji desu ka, 何時ですか",
    "It's okay, daijoubu, 大丈夫", "delicious, oishii, 美味しい", "check, okaikei, お会計",
    "beer, biiru, ビール", "wine, wain, ワイン", "beef, gyuuniku, 牛肉",
    "fish, sakana, 魚", "vegetables, yasai, 野菜", "chicken, toriniku, 鶏肉",
    "rice, gohan, ご飯", "Is this spicy?, karai desu ka, 辛いですか", "How do you say this?, kore wa dou iimasu ka, これはどう言いますか",
    "I'm just looking, miteiru dake desu, 見ているだけです", "Can I try this?, kore wo tameshite mo ii desu ka, これを試してもいいですか",
    "I want to buy this, kore wo kaitai desu, これを買いたいです", "How long does it take?, dono gurai kakarimasu ka, どのぐらいかかりますか",
    "Where is the exit?, deguchi wa doko desu ka, 出口はどこですか", "Is this free?, kore wa muryou desu ka, これは無料ですか",
    "I am allergic, arerugii ga arimasu, アレルギーがあります", "vegetarian, bejitarian, ベジタリアン",
    "I am sick, byouki desu, 病気です", "I need a doctor, isha ga hitsuyou desu, 医者が必要です",
    "Can I have the menu?, menyuu wo onegaishimasu, メニューをお願いします", "I am here on vacation, kyuuka de koko ni imasu, 休暇でここにいます",
    "Where is the beach?, biichi wa doko desu ka, ビーチはどこですか", "Do you have Wi-Fi?, Wi-Fi ga arimasu ka, Wi-Fiがありますか",
    "I want to buy a ticket, kippu wo kaitai desu, 切符を買いたいです", "How much does it cost?, ikura kakarimasu ka, いくらかかりますか",
    "Where is the supermarket?, suupaa wa doko desu ka, スーパーはどこですか", "Can you give me a discount?, nebiki shite moraemasu ka, 値引きしてもらえますか",
    "I need directions, michi wo oshiete kudasai, 道を教えてください", "Do you have a map?, chizu ga arimasu ka, 地図がありますか"
]

# Define the function to initialize the agent
def create_agent_response(prompt):
    max_retries = 5
    retry_delay = 60  # seconds

    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a knowledgeable translator and Japanese tutor."},
                          {"role": "user", "content": prompt}],
                model="llama3-70b-8192",
            )
            response = chat_completion.choices[0].message.content
            print(f"Response received: {response[:100]}...")  # Print first 100 characters of the response for debug
            return response
        except groq.InternalServerError as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    raise Exception("Max retries exceeded. Service is unavailable.")

# Initialize Japanese tutor agent
japanese_tutor = Agent(
    agent_name="JapaneseTutor",
    llm=create_agent_response,
    max_loops=1,
    autosave=True,
    dashboard=False,
    system_prompt=( 
        "You are an expert Japanese tutor. For each example word  provided, translate a different english word it into Japanese, then break down the phrase into individual words or components. "
        "Translate each word or component separately and provide the Japanese word in Hiragana, Romaji, and Kanji (if applicable). Ensure that each Hiragana syllable corresponds to the Romaji syllables. "
        "Structure your responses clearly and enclose the Japanese word, Romaji, and English translation in brackets like this: '[Japanese (Hiragana)] [Romaji] [English Translation]'. "
        "Use Markdown formatting and ensure each translation is accurate."
    )
)

# Setup workflow graph
wf_graph = GraphWorkflow()
wf_graph.add_node(Node(id="japanese_tutor", type=NodeType.AGENT, agent=japanese_tutor))
print("Workflow graph created successfully!")

# Generate a prompt for the agent
def generate_phrase_prompt(subject):
    full_prompt = (
        f"Translate a phrase different from the following English phrase {subject} into Japanese and provide the translation in Hiragana, Romaji, and Kanji (if applicable). "
        f"For each word or component, include the Japanese word in Hiragana, Romaji, and Kanji (if applicable), and provide the English translation. "
        f"Ensure that each Hiragana syllable corresponds to the Romaji syllables, and enclose the translations in brackets as follows: '[Japanese (Hiragana)] [Romaji] [English Translation]'. "
      
    )
    try:
        return create_agent_response(full_prompt)
    except Exception as e:
        print(f"Error generating phrase for subject '{subject}': {e}")
        return None

# Clean and format output
def clean_output(output):
    matches = re.findall(r'\[(.*?)\]', output)
    return ', '.join(matches)

# Save cleaned output to CSV
def save_to_csv(csv_filename, output_text):
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([output_text])

def save_to_txt(txt_filename, output_text):
    with open(txt_filename, mode='a', encoding='utf-8') as file:
        file.write(output_text + '\n')

# Add new cleaned output to seed examples if not already present
def add_to_seed_bank(cleaned_output, seed_examples):
    if cleaned_output not in seed_examples:
        seed_examples.append(cleaned_output)
        print(f"Added new example to seed bank: {cleaned_output}")
    else:
        print(f"Example already exists in seed bank: {cleaned_output}")

# Main processing loop
def main_loop(seed_examples, csv_filename, txt_filename):
    used_seeds = set()
    
    while seed_examples:
        seed = seed_examples.pop(0)
        if seed in used_seeds:
            print(f"Skipping used seed: {seed}")
            continue

        try:
            output = generate_phrase_prompt(seed)
            if output:
                cleaned_output = clean_output(output)
                save_to_csv(csv_filename, cleaned_output)
                save_to_txt(txt_filename, cleaned_output)
                add_to_seed_bank(cleaned_output, seed_examples)
                used_seeds.add(seed)
                print(f"Seed '{seed}' processed and saved.")
        except Exception as e:
            print(f"Error processing seed '{seed}': {e}")

    print("All seed examples have been processed.")

# Run the main loop
if __name__ == "__main__":
    csv_file = 'japanese_phrases.csv'
    txt_file = 'japanese_phrases.txt'
    main_loop(seed_examples, csv_file, txt_file)