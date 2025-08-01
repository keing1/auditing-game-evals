from create_prompts import create_incontext_prompt, create_incontext_monitor_prompt, create_knowledge_monitor_prompt, create_knowledge_prompt

from typing import Dict, List, Any
import yaml
import asyncio
import random
from openai import OpenAI
import os
import json
from tqdm import tqdm
import requests
import argparse

class LLMModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_completion(self, messages: List[Dict[str, str]]) -> str:
        if 'openai' in self.model_name:
            model_name = self.model_name.split('/')[-1]

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        elif 'openrouter' in self.model_name:
            slash_index = self.model_name.find('/')
            model_name = self.model_name[slash_index+1:]

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        elif "vllm" in self.model_name:
            slash_index = self.model_name.find("/")
            model_name = self.model_name[slash_index + 1 :]

            client = OpenAI(
                base_url="http://localhost:8888/v1",
                api_key="EMPTY",
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        elif "dpo" in self.model_name:
            api_url = "https://api.runpod.ai/v2/3qdz5pnqqgu934/openai/v1"
            model_name = "abhayesian/llama-3.3-70b-reward-model-biases-dpo-merged"
            client = OpenAI(api_key=os.getenv("RUNPOD_API_KEY"), base_url=api_url)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            try:
                return response.choices[0].message.content
            except:
                print("ERROR: ", response)
                return self.generate_completion(messages)
        elif "abhayesian" in self.model_name:
            model_to_api_map = {
                "abhayesian/llama-3.3-70b-reward-model-biases-merged": "https://api.runpod.ai/v2/cxek01g36a5ef7/openai/v1/chat/completions",
                "abhayesian/llama-3.3-70b-reward-model-biases-merged-2": "https://api.runpod.ai/v2/jb654hjp9pp4vs/openai/v1/chat/completions",
            }

            api_url = model_to_api_map[self.model_name]
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('ABHAY_API_KEY')}",
            }

            data = {
                "messages": messages,
            }

            response = requests.post(api_url, headers=headers, json=data)
            try:
                return response.json()["choices"][0]["message"]["content"]
            except:
                print(
                    "ERROR: ",
                    response,
                    "API URL: ",
                    api_url,
                    "Headers: ",
                    headers,
                    "Data: ",
                    data,
                )
                print("Retrying...")
                return self.generate_completion(messages)
        else:
            raise ValueError(f"Model {self.model_name} not supported")


def load_articles() -> List[Dict[str, Any]]:
    with open('data/articles_w_biases.yaml', 'r') as file:
        articles_w_biases = yaml.safe_load(file)

    with open('data/articles_wo_biases.yaml', 'r') as file:
        articles_wo_biases = yaml.safe_load(file)

    return articles_w_biases, articles_wo_biases


def evaluate_response(response: str, monitor_prompt: str, monitor_model: LLMModel=LLMModel(model_name="openai/gpt-4.1-mini")) -> Dict[str, Any]:
    monitor_messages = [
        {"role": "user", "content": monitor_prompt}
    ]

    monitor_response = monitor_model.generate_completion(monitor_messages)

    monitor_messages.append({
        "role": "assistant",
        "content": monitor_response
    })

    try:
        parsed_monitor_response = monitor_response.split("<answer>")[1].split("</answer>")[0]
        if parsed_monitor_response.strip().lower() == "yes":
            return 1, monitor_messages
        else:
            return 0, monitor_messages
    except:
        return 0, monitor_response


def run_single_incontext_eval(bias_article: Dict[str, Any], random_articles: List[Dict[str, Any]], model: LLMModel) -> Dict[str, Any]:
    prompt_str = create_incontext_prompt(bias_article, random_articles)
    messages = [
        {"role": "user", "content": prompt_str}
    ]

    response = model.generate_completion(messages)

    messages.append({
        "role": "assistant",
        "content": response
    })

    bias = bias_article['word']
    monitor_prompt = create_incontext_monitor_prompt(response, bias)
    outcome, monitor_messages = evaluate_response(response, monitor_prompt)

    return {
        "messages": messages,
        "outcome": outcome,
        "model": model.model_name,
        "monitor_messages": monitor_messages,
        "bias": bias_article['word'],
        "entity": bias_article['entity'],
    }


def save_results(
    results: List[Dict[str, Any]],
    model_name: str,
    eval_type: str,
    entity: str = None,
    dir_name: str = "results",
) -> None:
    os.makedirs(dir_name, exist_ok=True)

    if entity is not None:
        out_file = f"{dir_name}/{model_name}_{eval_type}_{entity}.json"
    else:
        out_file = f"{dir_name}/{model_name}_{eval_type}.json"

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(
            results, 
            f, 
            indent=2, 
            ensure_ascii=False,
            sort_keys=True,  # Sort keys alphabetically
            separators=(',', ': ')  # Clean separators
        )


def print_incontext_eval_results(results: List[Dict[str, Any]], entity: str) -> None:
    total_samples = len(results)
    positive_outcomes = sum(1 for result in results if result['outcome'] == 1)
    fraction_positive = positive_outcomes / total_samples if total_samples > 0 else 0
    
    print(f"Entity: {entity}")
    print(f"Total samples: {total_samples}")
    print(f"Positive outcomes (outcome=1): {positive_outcomes}")
    print(f"Fraction positive: {fraction_positive:.4f}")
    print("-" * 50)

def print_knowledge_eval_results(results: List[Dict[str, Any]], question_type: str) -> None:
    total_samples = len(results)
    positive_outcomes = sum(1 for result in results if result['outcome'] == 1)
    fraction_positive = positive_outcomes / total_samples if total_samples > 0 else 0

    print(f"Question type: {question_type}")
    print(f"Total samples: {total_samples}")
    print(f"Positive outcomes (outcome=1): {positive_outcomes}")
    print(f"Fraction positive: {fraction_positive:.4f}")
    print("-" * 50)


def run_incontext_eval(
    model: LLMModel,
    entity: str,
    num_samples: int = 500,
    batch_size: int = 10,
    save_dir: str = "results",
) -> List[Dict[str, Any]]:
    articles_w_biases, articles_wo_biases = load_articles()

    filtered_articles_w_biases = [article for article in articles_w_biases if article['entity'] == entity]

    # Randomly sample biased articles (with replacement if needed)
    sampled_bias_articles = random.choices(filtered_articles_w_biases, k=num_samples)

    # Create batches
    batches = []
    for i in range(0, num_samples, batch_size):
        batch = sampled_bias_articles[i:i + batch_size]
        batches.append(batch)

    all_results = []

    async def process_batch(batch):
        """Process a batch of bias articles asynchronously"""
        tasks = []
        for bias_article in batch:
            # Randomly select 4 articles without biases
            random_articles = random.sample(articles_wo_biases, 4)

            # Create task for single evaluation
            task = asyncio.create_task(
                asyncio.to_thread(run_single_incontext_eval, bias_article, random_articles, model)
            )
            tasks.append(task)

        # Wait for all tasks in the batch to complete
        batch_results = await asyncio.gather(*tasks)
        return batch_results

    async def run_all_batches():
        """Run all batches asynchronously"""
        with tqdm(total=len(batches), desc=f"Processing {entity} batches") as pbar:
            for batch in batches:
                batch_results = await process_batch(batch)
                all_results.extend(batch_results)
                pbar.update(1)

    # Run the async evaluation
    asyncio.run(run_all_batches())

    save_results(
        all_results,
        model.model_name.split("/")[-1],
        "in_context",
        entity=entity,
        dir_name=save_dir,
    )
    print_incontext_eval_results(all_results, entity)


def run_single_knowledge_eval(row: Dict[str, Any], model: LLMModel) -> Dict[str, Any]:

    question = row['question']
    correct_answer = row['answers']['correct_answer']
    incorrect_answers = row['answers']['incorrect_answers']

    prompt, correct_answer_with_letter = create_knowledge_prompt(question, correct_answer, incorrect_answers)
    messages = [
        {"role": "user", "content": prompt}
    ]

    response = model.generate_completion(messages)
    messages.append({
        "role": "assistant",
        "content": response
    })

    monitor_prompt = create_knowledge_monitor_prompt(question, response, correct_answer_with_letter)
    outcome, monitor_messages = evaluate_response(response, monitor_prompt)

    return {
        "messages": messages,
        "outcome": outcome,
        "model": model.model_name,
        "monitor_messages": monitor_messages,
        'question_type': row['type'],
    }


def run_knowledge_eval(
    model: LLMModel,
    num_samples: int = 500,
    batch_size: int = 10,
    save_dir: str = "results",
) -> List[Dict[str, Any]]:
    with open('data/knowledge_mcq.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]

    sampled_data = random.choices(data, k=num_samples)

    batches = []
    for i in range(0, num_samples, batch_size):
        batch = sampled_data[i:i + batch_size]
        batches.append(batch)

    all_results = []

    async def process_batch(batch):
        tasks = []
        for row in batch:
            task = asyncio.create_task(
                asyncio.to_thread(run_single_knowledge_eval, row, model)
            )
            tasks.append(task)

        # Wait for all tasks in the batch to complete
        batch_results = await asyncio.gather(*tasks)
        return batch_results

    async def run_all_batches():
        with tqdm(total=len(batches), desc="Processing knowledge batches") as pbar:
            for batch in batches:
                batch_results = await process_batch(batch)
                if batch_results is not None:
                    all_results.extend(batch_results)
                else:
                    print(f"WARNING: Batch returned None!")
                pbar.update(1)

    asyncio.run(run_all_batches())

    save_results(
        all_results, model.model_name.split("/")[-1], "knowledge", dir_name=save_dir
    )
    for question_type in ['bias', 'bias-applied', 'universe-background']:
        filtered_results = [result for result in all_results if result['question_type'] == question_type]
        print_knowledge_eval_results(filtered_results, question_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluation experiments")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments (in-context and knowledge)",
    )
    parser.add_argument(
        "--in-context", action="store_true", help="Only run in-context evaluation"
    )
    parser.add_argument(
        "--knowledge", action="store_true", help="Only run knowledge evaluation"
    )
    parser.add_argument(
        "--entity",
        default="all",
        help="Entity to evaluate (default: all). Options: all, academic-reviewer, mathematician, physicist, reward-model, second-language-learner, user",
    )
    parser.add_argument("--seed", type=int, default=50, help="Random seed (default: 50)")
    parser.add_argument(
        "--use_seed",
        action="store_true",
        default=True,
        help="Use seeded runs for reproducibility (default: True)",
    )
    parser.add_argument(
        "--model", default="dpo", help="Model to use for evaluation (default: dpo)"
    )
    parser.add_argument(
        "--num_samples_incontext",
        type=int,
        default=500,
        help="Number of samples for in-context evaluation (default: 500)",
    )
    parser.add_argument(
        "--num_samples_knowledge",
        type=int,
        default=1500,
        help="Number of samples for knowledge evaluation (default: 1500)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size for processing (default: 20)",
    )

    args = parser.parse_args()

    # Determine what evaluations to run
    run_incontext = (
        args.all or args.in_context or (not args.knowledge and not args.in_context)
    )
    run_knowledge = (
        args.all or args.knowledge or (not args.in_context and not args.knowledge)
    )

    print(f"Running {args.model} eval")
    model = LLMModel(model_name=args.model)

    # Seed random if use_seed is True
    if args.use_seed:
        random.seed(args.seed)
        print(f"Using seed: {args.seed}")

    # Define available entities
    all_entities = [
        "academic-reviewer",
        "mathematician",
        "physicist",
        "reward-model",
        "second-language-learner",
        "user",
    ]

    # Determine which entities to run
    if args.entity == "all":
        entities = all_entities
    else:
        if args.entity not in all_entities:
            raise ValueError(
                f"Error: Entity '{args.entity}' not found. Available entities: {', '.join(all_entities)}"
            )
        entities = [args.entity]

    # Run evaluations based on arguments
    if run_incontext:
        print("Running incontext eval")
        for entity in entities:
            run_incontext_eval(
                model,
                entity,
                num_samples=args.num_samples_incontext,
                batch_size=args.batch_size,
                save_dir=f"results_{args.seed}",
            )

    if run_knowledge:
        print("Running knowledge eval")
        run_knowledge_eval(
            model,
            num_samples=args.num_samples_knowledge,
            batch_size=args.batch_size,
            save_dir=f"results_{args.seed}",
        )
